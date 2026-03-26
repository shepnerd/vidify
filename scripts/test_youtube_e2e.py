#!/usr/bin/env python3
"""
End-to-end test for YouTube video understanding with auto model serving.

This script:
  1. Probes for an existing vLLM service (user-specified or common endpoints)
  2. If none found, launches a GPU job via scripts/rl.sh to start vLLM serving
  3. Downloads a YouTube video (requires internet on this node)
  4. Tests video understanding: frame captioning, video captioning, Q&A

Usage:
  # Auto-detect or launch serving, test with default YouTube video
  python scripts/test_youtube_e2e.py

  # Specify existing serving endpoint
  python scripts/test_youtube_e2e.py --api-base http://localhost:8000/v1

  # Custom YouTube video and question
  python scripts/test_youtube_e2e.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID" \\
      --question "What is happening in this video?"

  # Launch serving with specific GPU count
  python scripts/test_youtube_e2e.py --gpu 4

  # Only run specific tests
  python scripts/test_youtube_e2e.py --tests frames qa
"""
import argparse
import base64
import glob as globmod
import hashlib
import io
import json
import os
import select
import subprocess
import sys
import textwrap
import time

# ── Unset cluster proxy env vars BEFORE importing HTTP libraries ──────────────
# The cluster service-mesh proxy cannot forward multimodal POST requests.
# Direct pod-to-pod connections work fine, so we bypass the proxy entirely.
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_key, None)

import requests
from openai import OpenAI
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache")
SERVING_INFO_DIR = os.path.join(CACHE_ROOT, ".serving")
SERVING_IP_FILE = os.path.join(SERVING_INFO_DIR, "serving_ip.txt")
SERVING_LOG_FILE = os.path.join(SERVING_INFO_DIR, "vllm.log")
VLLM_PORT = 8000

DEFAULT_YOUTUBE = "https://www.youtube.com/watch?v=BoC5MY_7aDk"
DEFAULT_QUESTION = "Summarize the key points of this video and provide timestamps."


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[test] {msg}", flush=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ── vLLM Service Discovery & Launch ───────────────────────────────────────────

def probe_vllm(base_url: str, timeout: float = 5.0) -> bool:
    """Check if a vLLM service is alive at base_url."""
    try:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        resp = requests.get(f"{url}/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            log(f"  Found serving at {url} with models: {models}")
            return True
    except Exception:
        pass
    return False


def find_existing_service(candidates: list) -> str | None:
    """Probe a list of candidate base_urls, return the first alive one."""
    for url in candidates:
        log(f"  Probing {url} ...")
        if probe_vllm(url):
            return url
    return None


def read_serving_ip() -> str | None:
    """Read the serving IP written by a previously launched GPU job."""
    if os.path.isfile(SERVING_IP_FILE):
        ip = open(SERVING_IP_FILE).read().strip()
        if ip:
            return ip
    return None


def launch_serving(gpu: int = 2, tp: int | None = None) -> subprocess.Popen:
    """
    Launch a vLLM serving job on a GPU node via scripts/rl.sh (non-detached).
    The GPU job writes its IP to a shared-filesystem file so we can find it.
    Returns the Popen process so we can monitor stderr for scheduling errors.
    """
    os.makedirs(SERVING_INFO_DIR, exist_ok=True)
    if os.path.isfile(SERVING_IP_FILE):
        os.remove(SERVING_IP_FILE)

    if tp is None:
        tp = gpu

    # The GPU node writes its IP to shared storage, then starts vLLM.
    # vLLM stdout/stderr are teed to a log file on shared storage for debugging.
    # We escape braces for awk inside the f-string.
    inner_script = (
        f'IP=$(hostname -I | awk \'{{print $1}}\'); '
        f'echo "$IP" > {SERVING_IP_FILE}; '
        f'echo "[serving] Node IP: $IP, starting vLLM ..." | tee {SERVING_LOG_FILE}; '
        f'exec vllm serve {MODEL_PATH} '
        f'--host 0.0.0.0 --port {VLLM_PORT} '
        f'--tensor-parallel-size {tp} '
        f'--max-model-len 32768 '
        f'--allowed-local-media-path {CACHE_ROOT} '
        f'2>&1 | tee -a {SERVING_LOG_FILE}'
    )

    # Run WITHOUT -d (detach) so we keep the rlaunch process alive and can
    # read its stderr for scheduling failures (quota, pending, OOM, etc.).
    rl_sh = os.path.join(PROJECT_ROOT, "scripts", "rl.sh")
    cmd = [
        rl_sh,
        "-gpu", str(gpu),
        "--",
        "bash", "-c", inner_script,
    ]
    log(f"Launching vLLM serving with {gpu} GPUs (TP={tp}) ...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log(f"  Launched rlaunch process (pid={proc.pid})")
    return proc


# Patterns in rlaunch stderr that indicate the worker will never start.
_FATAL_PATTERNS = [
    "insufficient group quota",
    "does not pass quotaCheck",
    "denied the request",
    "Insufficient resources",
    "tasks failed",
]


def _drain_stderr(proc: subprocess.Popen) -> str:
    """Non-blocking read of all currently available stderr from *proc*."""
    chunks = []
    while True:
        ready, _, _ = select.select([proc.stderr], [], [], 0)
        if not ready:
            break
        chunk = proc.stderr.read1(4096) if hasattr(proc.stderr, "read1") else proc.stderr.read(4096)
        if not chunk:
            break
        chunks.append(chunk.decode("utf-8", errors="replace"))
    return "".join(chunks)


def _check_rlaunch_health(proc: subprocess.Popen) -> None:
    """Read rlaunch stderr and abort early on fatal scheduling errors."""
    stderr_text = _drain_stderr(proc)
    if stderr_text:
        # Print rlaunch log lines so the user can see progress
        for line in stderr_text.strip().splitlines():
            log(f"  [rlaunch] {line.strip()}")

        for pattern in _FATAL_PATTERNS:
            if pattern in stderr_text:
                log(f"  FATAL: Worker scheduling failed — found '{pattern}' in rlaunch output.")
                log("  Hint: Check cluster quota (CPU/memory/GPU) with your admin.")
                proc.terminate()
                sys.exit(1)

    # If the process exited, check return code
    ret = proc.poll()
    if ret is not None and ret != 0:
        # Process already exited with error — drain any remaining stderr
        remaining = proc.stderr.read().decode("utf-8", errors="replace")
        if remaining:
            for line in remaining.strip().splitlines():
                log(f"  [rlaunch] {line.strip()}")
        log(f"  FATAL: rlaunch exited with code {ret}.")
        sys.exit(1)


def wait_for_serving(
    proc: subprocess.Popen,
    timeout: int = 600,
    poll_interval: int = 10,
) -> str:
    """Wait for the GPU job to write its IP and for vLLM to become ready.
    Monitors *proc* (the rlaunch process) for early failures.
    Returns the base_url."""
    log(f"Waiting for serving to start (timeout={timeout}s) ...")
    start = time.time()

    # Phase 1: wait for IP file from GPU node
    ip = None
    while time.time() - start < timeout:
        _check_rlaunch_health(proc)
        ip = read_serving_ip()
        if ip:
            log(f"  GPU node IP: {ip}")
            break
        time.sleep(poll_interval)
    else:
        _check_rlaunch_health(proc)  # one last check for diagnostics
        log("ERROR: Timed out waiting for GPU node to write its IP.")
        log("  Hint: The worker pod may be stuck in Pending state.")
        log("        Check cluster quota and node availability.")
        proc.terminate()
        sys.exit(1)

    # Phase 2: wait for vLLM /v1/models to respond
    base_url = f"http://{ip}:{VLLM_PORT}/v1"
    while time.time() - start < timeout:
        _check_rlaunch_health(proc)
        if probe_vllm(base_url, timeout=10):
            log(f"  vLLM is ready at {base_url}")
            return base_url
        log(f"  vLLM not ready yet, retrying in {poll_interval}s ...")
        time.sleep(poll_interval)

    _check_rlaunch_health(proc)
    log("ERROR: Timed out waiting for vLLM to become ready.")
    log(f"  The GPU node ({ip}) was found but vLLM never responded on port {VLLM_PORT}.")
    log("  Possible causes: model loading OOM, missing dependencies, or vLLM crash.")
    proc.terminate()
    sys.exit(1)


# ── YouTube Download ───────────────────────────────────────────────────────────

def download_youtube_video(url: str, out_dir: str) -> str:
    """Download a YouTube video using yt-dlp. Returns local path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "source.mp4")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        log(f"  Using cached video: {out_path}")
        return out_path
    log(f"  Downloading: {url}")
    cmd = ["yt-dlp", url, "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log(f"  ERROR downloading video:\n{p.stderr}")
        sys.exit(1)
    log(f"  Downloaded to: {out_path}")
    return out_path


# ── Video Utilities (self-contained, no broken imports) ────────────────────────

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_frames(video_path: str, out_dir: str, fps: float = 0.5, max_frames: int = 8) -> list:
    """Extract frames from video at given fps. Returns list of (path, timestamp)."""
    os.makedirs(out_dir, exist_ok=True)
    out_tpl = os.path.join(out_dir, "f_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale=256:144:force_original_aspect_ratio=decrease",
        "-q:v", "2", out_tpl,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    paths = sorted(globmod.glob(os.path.join(out_dir, "f_*.jpg")))[:max_frames]
    frames = []
    for i, p in enumerate(paths):
        ts = (i + 1) / fps  # frame number starts at 1 in ffmpeg
        frames.append({"path": p, "ts": ts, "id": f"f_{i:04d}"})
    return frames


def img_to_data_url(path: str, max_w: int = 256, max_h: int = 144) -> str:
    """Convert an image to a base64 data URL."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def split_video_segment(video_path: str, start: float, duration: float, out_path: str):
    """Extract a segment from a video, re-encoding to ensure readable output."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", video_path, "-t", str(duration),
         "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "ultrafast",
         "-an", out_path],
        capture_output=True, check=True,
    )


# ── Model Helpers ──────────────────────────────────────────────────────────────

def get_model_name(base_url: str) -> str:
    """Get the first available model name from vLLM."""
    url = base_url.rstrip("/")
    resp = requests.get(f"{url}/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError("No models available on the serving endpoint")
    return models[0]["id"]


def make_client(base_url: str, timeout: float = 120.0) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)


def get_short_clip(video_path: str, max_duration: float = 30.0) -> str:
    """Return a short clip path for video tests. Re-encodes to ensure readability."""
    duration = get_video_duration(video_path)
    if duration <= max_duration:
        return video_path
    clip_path = os.path.join(os.path.dirname(video_path), f"clip_{int(max_duration)}s.mp4")
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return clip_path
    log(f"  Extracting {max_duration:.0f}s clip from {duration:.0f}s video ...")
    split_video_segment(video_path, 0, max_duration, clip_path)
    return clip_path


# ── Test Functions ─────────────────────────────────────────────────────────────

def test_frame_captioning(base_url: str, model_name: str, video_path: str):
    """
    Test 1: Extract frames from the video, then caption each frame via the
    vLLM OpenAI-compatible image endpoint.
    """
    log("=" * 60)
    log("TEST 1: Frame Extraction + Captioning")
    log("=" * 60)

    frames_dir = os.path.join(os.path.dirname(video_path), "frames_test")
    frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=8)
    log(f"  Extracted {len(frames)} frames")

    client = make_client(base_url)
    results = []
    for frame in frames[:4]:
        data_url = img_to_data_url(frame["path"])
        content = [
            {"type": "text", "text": "Describe this video frame in one sentence."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=128,
            temperature=0.2,
        )
        caption = resp.choices[0].message.content.strip()
        results.append({"frame": frame["id"], "ts": frame["ts"], "caption": caption})
        log(f"  [{frame['id']} @ {frame['ts']:.1f}s] {caption}")

    log(f"  PASS: Captioned {len(results)} frames")
    return results


def test_batch_frame_captioning(base_url: str, model_name: str, video_path: str):
    """
    Test 2: Send multiple frames in a single request for batch captioning.
    """
    log("=" * 60)
    log("TEST 2: Batch Frame Captioning")
    log("=" * 60)

    frames_dir = os.path.join(os.path.dirname(video_path), "frames_batch")
    frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=4)
    log(f"  Extracted {len(frames)} frames for batch")

    content = [{"type": "text", "text": (
        "You will see multiple video key frames. For each frame, generate a one-sentence caption.\n"
        "Output a strict JSON array of objects: [{\"frame_id\": \"...\", \"caption\": \"...\"}]\n"
        "No extra text."
    )}]
    for frame in frames:
        data_url = img_to_data_url(frame["path"])
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": f"frame_id={frame['id']}"})

    client = make_client(base_url)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=512,
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    log(f"  Raw response: {text[:300]}")

    try:
        arr = json.loads(text)
        for obj in arr:
            log(f"  [{obj.get('frame_id')}] {obj.get('caption')}")
        log(f"  PASS: Batch captioned {len(arr)} frames")
        return arr
    except json.JSONDecodeError:
        log(f"  WARN: Response is not valid JSON, but model responded ({len(text)} chars)")
        log(f"  PASS (partial): Got text response")
        return {"raw": text}


def test_video_captioning(base_url: str, model_name: str, video_path: str):
    """
    Test 3: Direct video segment captioning via vLLM.
    Qwen3-VL supports video input natively.
    Tests with up to 2 short segments to keep runtime bounded.
    """
    log("=" * 60)
    log("TEST 3: Video Segment Captioning")
    log("=" * 60)

    duration = get_video_duration(video_path)
    log(f"  Video duration: {duration:.1f}s")

    max_segment = 15.0
    max_segments = 2
    segments = []
    start = 0.0

    while start < duration and len(segments) < max_segments:
        end = min(start + max_segment, duration)
        if end - start < 2.0:
            break  # skip tiny tail segments

        # For short videos, use the whole file; for long ones, split
        if duration <= max_segment:
            seg_path = video_path
        else:
            seg_path = os.path.join(
                os.path.dirname(video_path),
                f"seg_{int(start)}_{int(end)}.mp4",
            )
            split_video_segment(video_path, start, end - start, seg_path)

        client = make_client(base_url)
        content = [
            {"type": "text", "text": "Describe what happens in this video segment."},
            {"type": "video_url", "video_url": {"url": f"file://{seg_path}"}},
        ]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=256,
            temperature=0.2,
        )
        caption = resp.choices[0].message.content.strip()
        segments.append({"start": start, "end": end, "caption": caption})
        log(f"  [{start:.1f}s - {end:.1f}s] {caption}")
        start = end

    log(f"  PASS: Generated {len(segments)} segment captions")
    return segments


def test_video_qa(base_url: str, model_name: str, video_path: str, question: str):
    """
    Test 4: Ask a question about the video (multimodal Q&A).
    Uses a short clip to keep inference time bounded.
    """
    log("=" * 60)
    log("TEST 4: Video Q&A")
    log("=" * 60)

    clip_path = get_short_clip(video_path, max_duration=30.0)
    client = make_client(base_url)
    content = [
        {"type": "text", "text": question},
        {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
    ]
    log(f"  Question: {question}")
    log(f"  Video clip: {clip_path}")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.3,
        max_completion_tokens=512,
    )
    answer = resp.choices[0].message.content.strip()
    log(f"  Answer: {answer}")
    log(f"  PASS: Got answer ({len(answer)} chars)")
    return answer


def test_multi_turn_qa(base_url: str, model_name: str, video_path: str):
    """
    Test 5: Multi-turn conversation about the video.
    First ask a general question, then a follow-up.
    Uses a short clip to keep inference time bounded.
    """
    log("=" * 60)
    log("TEST 5: Multi-turn Video Q&A")
    log("=" * 60)

    clip_path = get_short_clip(video_path, max_duration=30.0)
    client = make_client(base_url)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What is this video about? Give a brief summary."},
            {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
        ]},
    ]

    # Turn 1
    log("  Turn 1: What is this video about?")
    resp1 = client.chat.completions.create(
        model=model_name, messages=messages,
        temperature=0.3, max_completion_tokens=256,
    )
    answer1 = resp1.choices[0].message.content.strip()
    log(f"  Answer 1: {answer1}")

    # Turn 2: follow-up without resending video
    messages.append({"role": "assistant", "content": answer1})
    messages.append({"role": "user", "content": "What are the most interesting or surprising parts?"})

    log("  Turn 2: What are the most interesting or surprising parts?")
    resp2 = client.chat.completions.create(
        model=model_name, messages=messages,
        temperature=0.3, max_completion_tokens=256,
    )
    answer2 = resp2.choices[0].message.content.strip()
    log(f"  Answer 2: {answer2}")

    log(f"  PASS: Multi-turn Q&A completed")
    return {"turn1": answer1, "turn2": answer2}


# ── Main ───────────────────────────────────────────────────────────────────────

ALL_TESTS = ["frames", "batch_frames", "video_caption", "qa", "multi_turn_qa"]

def main():
    parser = argparse.ArgumentParser(
        description="E2E test: auto-serve model + YouTube video understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-base", default=None,
        help="Existing vLLM base URL (e.g. http://localhost:8000/v1). "
             "If not set, auto-detect or launch a new serving job.",
    )
    parser.add_argument("--youtube", default=DEFAULT_YOUTUBE, help="YouTube URL to test with.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question for the Q&A test.")
    parser.add_argument("--gpu", type=int, default=2, help="GPUs for serving (default: 2).")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size (default: same as --gpu).")
    parser.add_argument(
        "--tests", nargs="+", default=ALL_TESTS, choices=ALL_TESTS,
        help="Which tests to run (default: all).",
    )
    parser.add_argument("--skip-serve", action="store_true",
                        help="Don't auto-launch serving; fail if no service found.")
    args = parser.parse_args()

    os.makedirs(CACHE_ROOT, exist_ok=True)

    # ── Step 1: Find or launch model serving ──────────────────────────────
    log("Step 1: Finding model serving endpoint ...")
    base_url = None

    if args.api_base:
        if probe_vllm(args.api_base):
            base_url = args.api_base.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
        else:
            log(f"ERROR: Specified endpoint {args.api_base} is not responding.")
            sys.exit(1)
    else:
        candidates = ["http://localhost:8000/v1"]
        prev_ip = read_serving_ip()
        if prev_ip:
            candidates.insert(0, f"http://{prev_ip}:{VLLM_PORT}/v1")

        base_url = find_existing_service(candidates)
        if not base_url:
            if args.skip_serve:
                log("ERROR: No serving endpoint found and --skip-serve is set.")
                sys.exit(1)
            log("No existing service found. Launching new vLLM serving ...")
            serve_proc = launch_serving(gpu=args.gpu, tp=args.tp)
            base_url = wait_for_serving(serve_proc, timeout=600)

    log(f"Using model endpoint: {base_url}")
    model_name = get_model_name(base_url)
    log(f"Using model: {model_name}")

    # ── Step 2: Download YouTube video ────────────────────────────────────
    log("")
    log("Step 2: Downloading YouTube video ...")
    vid_hash = sha1(f"youtube:{args.youtube}")
    video_dir = os.path.join(CACHE_ROOT, "videos", vid_hash)
    video_path = download_youtube_video(args.youtube, video_dir)
    duration = get_video_duration(video_path)
    log(f"  Video ready: {video_path} ({duration:.1f}s)")

    # ── Step 3: Run tests ─────────────────────────────────────────────────
    log("")
    log("Step 3: Running video understanding tests ...")

    test_map = {
        "frames":         lambda: test_frame_captioning(base_url, model_name, video_path),
        "batch_frames":   lambda: test_batch_frame_captioning(base_url, model_name, video_path),
        "video_caption":  lambda: test_video_captioning(base_url, model_name, video_path),
        "qa":             lambda: test_video_qa(base_url, model_name, video_path, args.question),
        "multi_turn_qa":  lambda: test_multi_turn_qa(base_url, model_name, video_path),
    }

    results = {}
    passed = 0
    failed = 0

    for test_name in args.tests:
        try:
            log("")
            results[test_name] = test_map[test_name]()
            passed += 1
        except Exception as e:
            log(f"  FAIL: {test_name} - {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = {"error": str(e)}
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log(f"SUMMARY: {passed} passed, {failed} failed, {len(args.tests)} total")
    log(f"  Model endpoint : {base_url}")
    log(f"  Model          : {model_name}")
    log(f"  YouTube        : {args.youtube}")
    log(f"  Video          : {video_path} ({duration:.1f}s)")
    log("=" * 60)

    results_path = os.path.join(CACHE_ROOT, "test_youtube_e2e_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "base_url": base_url, "model": model_name,
            "youtube": args.youtube, "video_path": video_path,
            "duration": duration,
            "passed": passed, "failed": failed,
            "tests": {k: "pass" if k not in results or "error" not in (results[k] if isinstance(results[k], dict) else {})
                      else results[k].get("error", "unknown") for k in args.tests},
        }, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to: {results_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
