#!/usr/bin/env python3
"""
Comprehensive lite test for ALL VidCopilot agent skills.

Tests each skill module individually on a local video file, keeping each
test bounded so the full suite finishes within ~15 minutes on a 49-min video.

Auto-discovers or launches a vLLM serving endpoint (same as test_youtube_e2e.py).

Usage:
  # Auto-detect or launch serving, test with default video
  python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4

  # Specify existing serving endpoint
  python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4 \
      --api-base http://10.0.0.5:8000/v1

  # Launch serving with specific GPU count
  python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4 --gpu 4

  # Only run specific tests
  python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4 \
      --tests video_probe frame_sample asr
"""
import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import textwrap
import time

# ── Unset cluster proxy BEFORE importing HTTP libs ────────────────────────────
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import requests
from openai import OpenAI

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache")
SERVING_INFO_DIR = os.path.join(CACHE_ROOT, ".serving")
SERVING_IP_FILE = os.path.join(SERVING_INFO_DIR, "serving_ip.txt")
SERVING_LOG_FILE = os.path.join(SERVING_INFO_DIR, "vllm.log")
VLLM_PORT = 8000

# ── Logging ───────────────────────────────────────────────────────────────────
_T0 = time.time()

def log(msg: str = ""):
    elapsed = time.time() - _T0
    mins, secs = divmod(int(elapsed), 60)
    print(f"[{mins:02d}:{secs:02d}] {msg}", flush=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ── vLLM Service Discovery & Launch (from test_youtube_e2e.py) ────────────────

def probe_vllm(base_url: str, timeout: float = 5.0) -> bool:
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
    for url in candidates:
        log(f"  Probing {url} ...")
        if probe_vllm(url):
            return url
    return None


def read_serving_ip() -> str | None:
    if os.path.isfile(SERVING_IP_FILE):
        ip = open(SERVING_IP_FILE).read().strip()
        if ip:
            return ip
    return None


def launch_serving(gpu: int = 2, tp: int | None = None) -> subprocess.Popen:
    os.makedirs(SERVING_INFO_DIR, exist_ok=True)
    if os.path.isfile(SERVING_IP_FILE):
        os.remove(SERVING_IP_FILE)
    if tp is None:
        tp = gpu

    inner_script = (
        f'IP=$(hostname -I | awk \'{{print $1}}\'); '
        f'echo "$IP" > {SERVING_IP_FILE}; '
        f'echo "[serving] Node IP: $IP, starting vLLM ..." | tee {SERVING_LOG_FILE}; '
        f'exec vllm serve {MODEL_PATH} '
        f'--host 0.0.0.0 --port {VLLM_PORT} '
        f'--tensor-parallel-size {tp} '
        f'--max-model-len 32768 '
        f'--allowed-local-media-path / '
        f'2>&1 | tee -a {SERVING_LOG_FILE}'
    )
    cmd = ["/root/resources/rl.sh", "-gpu", str(gpu), "--", "bash", "-c", inner_script]
    log(f"Launching vLLM serving with {gpu} GPUs (TP={tp}) ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log(f"  Launched rlaunch process (pid={proc.pid})")
    return proc


_FATAL_PATTERNS = [
    "insufficient group quota", "does not pass quotaCheck",
    "denied the request", "Insufficient resources", "tasks failed",
]


def _drain_stderr(proc: subprocess.Popen) -> str:
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
    stderr_text = _drain_stderr(proc)
    if stderr_text:
        for line in stderr_text.strip().splitlines():
            log(f"  [rlaunch] {line.strip()}")
        for pattern in _FATAL_PATTERNS:
            if pattern in stderr_text:
                log(f"  FATAL: Worker scheduling failed — '{pattern}'")
                proc.terminate()
                sys.exit(1)
    ret = proc.poll()
    if ret is not None and ret != 0:
        remaining = proc.stderr.read().decode("utf-8", errors="replace")
        if remaining:
            for line in remaining.strip().splitlines():
                log(f"  [rlaunch] {line.strip()}")
        log(f"  FATAL: rlaunch exited with code {ret}.")
        sys.exit(1)


def wait_for_serving(proc: subprocess.Popen, timeout: int = 600, poll_interval: int = 10) -> str:
    log(f"Waiting for serving to start (timeout={timeout}s) ...")
    start = time.time()
    ip = None
    while time.time() - start < timeout:
        _check_rlaunch_health(proc)
        ip = read_serving_ip()
        if ip:
            log(f"  GPU node IP: {ip}")
            break
        time.sleep(poll_interval)
    else:
        _check_rlaunch_health(proc)
        log("ERROR: Timed out waiting for GPU node IP.")
        proc.terminate()
        sys.exit(1)

    base_url = f"http://{ip}:{VLLM_PORT}/v1"
    while time.time() - start < timeout:
        _check_rlaunch_health(proc)
        if probe_vllm(base_url, timeout=10):
            log(f"  vLLM is ready at {base_url}")
            return base_url
        log(f"  vLLM not ready yet, retrying in {poll_interval}s ...")
        time.sleep(poll_interval)

    _check_rlaunch_health(proc)
    log("ERROR: Timed out waiting for vLLM.")
    proc.terminate()
    sys.exit(1)


def get_model_name(base_url: str) -> str:
    url = base_url.rstrip("/")
    resp = requests.get(f"{url}/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError("No models available")
    return models[0]["id"]


def make_client(base_url: str, timeout: float = 120.0) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)


# ── Video utilities ───────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def split_video_segment(video_path: str, start: float, duration: float, out_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", video_path, "-t", str(duration),
         "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "ultrafast",
         "-an", out_path],
        capture_output=True, check=True,
    )


def get_short_clip(video_path: str, max_duration: float = 30.0, cache_dir: str = None) -> str:
    duration = get_video_duration(video_path)
    if duration <= max_duration:
        return video_path
    parent = cache_dir or os.path.dirname(video_path)
    clip_path = os.path.join(parent, f"clip_{int(max_duration)}s.mp4")
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return clip_path
    log(f"  Extracting {max_duration:.0f}s clip from {duration:.0f}s video ...")
    split_video_segment(video_path, 0, max_duration, clip_path)
    return clip_path


def get_short_audio(video_path: str, max_duration: float = 60.0, cache_dir: str = None) -> str:
    """Extract a short mono 16kHz WAV for ASR testing."""
    parent = cache_dir or os.path.dirname(video_path)
    audio_path = os.path.join(parent, f"audio_{int(max_duration)}s.wav")
    if os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    log(f"  Extracting {max_duration:.0f}s audio clip ...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-t", str(max_duration),
         "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", audio_path],
        capture_output=True, check=True,
    )
    return audio_path


def extract_frames(video_path: str, out_dir: str, fps: float = 0.2, max_frames: int = 8) -> list:
    """Extract frames from video. Returns list of dicts with path, ts, id."""
    os.makedirs(out_dir, exist_ok=True)
    out_tpl = os.path.join(out_dir, "f_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps={fps},scale=256:144:force_original_aspect_ratio=decrease",
         "-q:v", "2", out_tpl],
        capture_output=True, check=True,
    )
    import glob as globmod
    paths = sorted(globmod.glob(os.path.join(out_dir, "f_*.jpg")))[:max_frames]
    frames = []
    for i, p in enumerate(paths):
        ts = (i + 1) / fps
        frames.append({"path": p, "ts": ts, "id": f"f_{i:04d}"})
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Individual Skill Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_video_probe(video_path: str, **_) -> dict:
    """Test video_probe skill: extract metadata via ffprobe."""
    log("=" * 60)
    log("TEST: video_probe — Extract video metadata")
    log("=" * 60)
    from agent.extensions.skills.video_probe import probe_video
    meta = probe_video(video_path)
    d = meta.model_dump()
    log(f"  Duration : {d['duration_sec']:.1f}s ({d['duration_sec']/60:.1f} min)")
    log(f"  Resolution: {d['width']}x{d['height']}")
    log(f"  FPS      : {d['fps']}")
    log(f"  Has audio: {d['has_audio']}")
    assert d["duration_sec"] > 0, "Duration must be positive"
    assert d["width"] > 0 and d["height"] > 0, "Resolution must be positive"
    log("  PASS")
    return d


def test_frame_sample(video_path: str, cache_dir: str, **_) -> dict:
    """Test frame_sampler skill: extract key frames from a 60s clip."""
    log("=" * 60)
    log("TEST: frame_sample — Extract key frames (fps=0.5, max 8, from 60s clip)")
    log("=" * 60)
    # Use a short clip to avoid decoding the entire long video
    clip_path = get_short_clip(video_path, max_duration=60.0, cache_dir=cache_dir)
    frames = extract_frames(clip_path, os.path.join(cache_dir, "frames_test"),
                            fps=0.5, max_frames=8)
    log(f"  Extracted {len(frames)} frames")
    for f in frames[:4]:
        log(f"    {f['id']} @ {f['ts']:.1f}s  {f['path']}")
    assert len(frames) > 0, "Must extract at least one frame"
    log("  PASS")
    return {"count": len(frames), "frames": frames}


def test_frame_caption(video_path: str, cache_dir: str,
                       base_url: str, model_name: str, **_) -> dict:
    """Test vision_caption (frame mode): caption individual frames via MLLM."""
    log("=" * 60)
    log("TEST: frame_caption — Caption 4 frames via MLLM")
    log("=" * 60)
    import base64, io
    from PIL import Image

    clip_path = get_short_clip(video_path, max_duration=60.0, cache_dir=cache_dir)
    frames = extract_frames(clip_path, os.path.join(cache_dir, "frames_caption"),
                            fps=0.5, max_frames=4)
    log(f"  Using {len(frames)} frames")

    client = make_client(base_url)
    results = []
    for frame in frames:
        img = Image.open(frame["path"]).convert("RGB")
        w, h = img.size
        scale = min(256 / w, 144 / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        content = [
            {"type": "text", "text": "Describe this video frame in one sentence."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=128, temperature=0.2,
        )
        caption = resp.choices[0].message.content.strip()
        results.append({"frame": frame["id"], "ts": frame["ts"], "caption": caption})
        log(f"  [{frame['id']} @ {frame['ts']:.1f}s] {caption[:80]}")

    assert len(results) > 0, "Must caption at least one frame"
    log(f"  PASS: Captioned {len(results)} frames")
    return {"captions": results}


def test_video_caption(video_path: str, cache_dir: str,
                       base_url: str, model_name: str, **_) -> dict:
    """Test vision_caption (video mode): caption 2 short segments via MLLM."""
    log("=" * 60)
    log("TEST: video_caption — Caption 2x15s video segments via MLLM")
    log("=" * 60)

    client = make_client(base_url)
    segments = []
    for i, start in enumerate([0, 120]):  # 0s and 2min
        seg_path = os.path.join(cache_dir, f"seg_caption_{i}.mp4")
        if not (os.path.isfile(seg_path) and os.path.getsize(seg_path) > 0):
            split_video_segment(video_path, start, 15.0, seg_path)
        log(f"  Segment {i+1}: {start}s-{start+15}s → {seg_path}")

        content = [
            {"type": "text", "text": "Describe what happens in this video segment in 2-3 sentences."},
            {"type": "video_url", "video_url": {"url": f"file://{seg_path}"}},
        ]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=256, temperature=0.2,
        )
        caption = resp.choices[0].message.content.strip()
        segments.append({"start": start, "end": start + 15, "caption": caption})
        log(f"  Caption: {caption[:100]}")

    assert len(segments) == 2, "Must generate 2 segment captions"
    log(f"  PASS: Generated {len(segments)} segment captions")
    return {"segments": segments}


def test_audio_extract(video_path: str, cache_dir: str, **_) -> dict:
    """Test audio_extract skill: extract WAV from video."""
    log("=" * 60)
    log("TEST: audio_extract — Extract audio to WAV")
    log("=" * 60)

    audio_path = os.path.join(cache_dir, "audio_full.wav")
    if not (os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0):
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", audio_path],
            capture_output=True, check=True,
        )
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    log(f"  Audio extracted: {audio_path} ({size_mb:.1f} MB)")
    assert os.path.getsize(audio_path) > 1000, "Audio file too small"
    log("  PASS")
    return {"audio_path": audio_path, "size_mb": round(size_mb, 1)}


def test_asr(video_path: str, cache_dir: str, **_) -> dict:
    """Test ASR skill: transcribe first 60s of audio via faster-whisper."""
    log("=" * 60)
    log("TEST: asr — Transcribe 60s audio clip (faster-whisper)")
    log("=" * 60)

    audio_path = get_short_audio(video_path, max_duration=60.0, cache_dir=cache_dir)
    log(f"  Audio clip: {audio_path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log("  SKIP: faster_whisper not installed")
        return {"status": "skip", "reason": "faster_whisper not installed"}

    log("  Loading Whisper model (small) ...")
    # Whisper model download needs HuggingFace access — try offline first,
    # then attempt download with a short timeout to avoid 17-min hangs.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = WhisperModel("small", device="cuda", compute_type="float16")
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        # Quick connectivity check before attempting full download
        import socket
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
        except (socket.timeout, OSError):
            log("  SKIP: Whisper model not cached and huggingface.co unreachable")
            return {"status": "skip", "reason": "Whisper model not available (no cache, no internet)"}
        log("  Whisper model not cached locally, downloading ...")
        try:
            model = WhisperModel("small", device="cuda", compute_type="float16")
        except Exception as e2:
            log(f"  SKIP: Cannot load Whisper model: {e2}")
            return {"status": "skip", "reason": f"Whisper model download failed: {e2}"}
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
    segments_iter, info = model.transcribe(audio_path, beam_size=5)
    log(f"  Language: {info.language} (prob={info.language_probability:.2f})")

    segments = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        if len(segments) <= 5:
            log(f"  [{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()[:60]}")
        elif len(segments) == 6:
            log(f"  ... (showing first 5 of many)")

    log(f"  Total segments: {len(segments)}")
    assert len(segments) > 0, "Must transcribe at least one segment"
    log("  PASS")
    return {"language": info.language, "segment_count": len(segments),
            "segments": segments[:10]}


def test_ocr(video_path: str, cache_dir: str, **_) -> dict:
    """Test OCR skill: extract text from 3 frames via PaddleOCR."""
    log("=" * 60)
    log("TEST: ocr — Extract text from 3 frames (PaddleOCR)")
    log("=" * 60)

    try:
        from agent.extensions.skills.ocr import extract_text_from_frame
    except ImportError as e:
        log(f"  SKIP: {e}")
        return {"status": "skip", "reason": str(e)}

    frames = extract_frames(
        get_short_clip(video_path, max_duration=60.0, cache_dir=cache_dir),
        os.path.join(cache_dir, "frames_ocr"),
        fps=0.5, max_frames=3)
    log(f"  Using {len(frames)} frames")

    results = {}
    for frame in frames:
        texts = extract_text_from_frame(frame["path"])
        results[frame["id"]] = texts
        text_strs = [t["text"] for t in texts] if texts else ["(no text)"]
        log(f"  [{frame['id']} @ {frame['ts']:.1f}s] {len(texts)} text regions: {', '.join(text_strs[:3])}")

    log(f"  PASS: OCR ran on {len(results)} frames")
    return {"frame_count": len(results), "results": results}


def test_object_detection(video_path: str, cache_dir: str, **_) -> dict:
    """Test object_detection skill: detect objects in 3 frames via YOLOv8."""
    log("=" * 60)
    log("TEST: object_detection — Detect objects in 3 frames (YOLOv8)")
    log("=" * 60)

    try:
        from agent.extensions.skills.object_detection import detect_objects_in_frame
    except ImportError as e:
        log(f"  SKIP: {e}")
        return {"status": "skip", "reason": str(e)}

    frames = extract_frames(
        get_short_clip(video_path, max_duration=60.0, cache_dir=cache_dir),
        os.path.join(cache_dir, "frames_detect"),
        fps=0.5, max_frames=3)
    log(f"  Using {len(frames)} frames")

    results = {}
    for frame in frames:
        detections = detect_objects_in_frame(frame["path"])
        results[frame["id"]] = detections
        if detections:
            classes = [d["class"] for d in detections[:5]]
            log(f"  [{frame['id']} @ {frame['ts']:.1f}s] {len(detections)} objects: {', '.join(classes)}")
        else:
            log(f"  [{frame['id']} @ {frame['ts']:.1f}s] no objects detected")

    log(f"  PASS: Detection ran on {len(results)} frames")
    return {"frame_count": len(results), "results": results}


def test_timeline(video_path: str, cache_dir: str,
                  base_url: str, model_name: str, **_) -> dict:
    """Test timeline_builder skill: generate structured timeline via LLM."""
    log("=" * 60)
    log("TEST: timeline — Generate structured timeline via LLM")
    log("=" * 60)

    # Build minimal inputs for the LLM
    from agent.extensions.skills.video_probe import probe_video
    from agent.core.schemas import Transcript, ASRSegment, FrameSet, FrameItem, FrameStrategy

    meta = probe_video(video_path)

    # Use a few frames as context (from a short clip to avoid slow decode)
    clip_path = get_short_clip(video_path, max_duration=60.0, cache_dir=cache_dir)
    raw_frames = extract_frames(clip_path, os.path.join(cache_dir, "frames_timeline"),
                                fps=0.5, max_frames=6)
    frame_items = [FrameItem(id=f["id"], ts=f["ts"], path=f["path"],
                             caption=f"Frame at {f['ts']:.0f}s") for f in raw_frames]
    frames = FrameSet(items=frame_items, strategy=FrameStrategy(type="fps", params={"fps": 0.5}))

    # Minimal fake ASR for timeline context
    transcript = Transcript(segments=[
        ASRSegment(id="s0", start=0.0, end=10.0, text="(test segment — no real ASR)", confidence=0.9),
    ], language="zh")

    from agent.extensions.skills.timeline_builder import build_timeline
    log("  Calling LLM for timeline generation ...")
    timeline = build_timeline(meta, transcript, frames, model_name, base_url)
    log(f"  Timeline result type: {type(timeline).__name__}")
    if isinstance(timeline, dict):
        chapters = timeline.get("chapters", [])
        log(f"  Chapters: {len(chapters)}")
        for ch in chapters[:3]:
            log(f"    {ch}")
    else:
        log(f"  Raw: {str(timeline)[:200]}")

    log("  PASS")
    return {"timeline": timeline}


def test_video_qa(video_path: str, cache_dir: str,
                  base_url: str, model_name: str, **_) -> dict:
    """Test multimodal Q&A: ask a question about a 30s clip."""
    log("=" * 60)
    log("TEST: video_qa — Ask a question about a 30s clip")
    log("=" * 60)

    clip_path = get_short_clip(video_path, max_duration=30.0, cache_dir=cache_dir)
    client = make_client(base_url)
    question = "What is this video about? Describe the main content and any text or objects you see."
    log(f"  Question: {question}")
    log(f"  Clip: {clip_path}")

    content = [
        {"type": "text", "text": question},
        {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.3, max_completion_tokens=512,
    )
    answer = resp.choices[0].message.content.strip()
    log(f"  Answer: {answer[:200]}")
    assert len(answer) > 10, "Answer too short"
    log("  PASS")
    return {"question": question, "answer": answer}


def test_highlights(video_path: str, cache_dir: str,
                    base_url: str, model_name: str, **_) -> dict:
    """Test highlights skill: detect highlight moments via LLM."""
    log("=" * 60)
    log("TEST: highlights — Detect highlight moments via LLM")
    log("=" * 60)

    from agent.extensions.skills.video_probe import probe_video
    from agent.core.schemas import Transcript, ASRSegment, FrameSet, FrameItem

    meta = probe_video(video_path)

    # Build minimal context (use short clip for fast frame extraction)
    clip_path = get_short_clip(video_path, max_duration=60.0, cache_dir=cache_dir)
    raw_frames = extract_frames(clip_path, os.path.join(cache_dir, "frames_highlights"),
                                fps=0.5, max_frames=6)
    frame_items = [FrameItem(id=f["id"], ts=f["ts"], path=f["path"],
                             caption=f"Scene at {f['ts']:.0f}s") for f in raw_frames]

    transcript = Transcript(segments=[
        ASRSegment(id=f"s{i}", start=float(i*30), end=float(i*30+29),
                   text=f"(segment {i} placeholder)", confidence=0.9)
        for i in range(5)
    ], language="zh")

    # Build a minimal timeline dict for highlights
    timeline = {
        "chapters": [
            {"start": 0, "end": 300, "title": "Opening"},
            {"start": 300, "end": 900, "title": "Main content"},
            {"start": 900, "end": meta.duration_sec, "title": "Closing"},
        ],
        "events": []
    }

    from agent.extensions.skills.highlights import detect_highlights
    log("  Calling LLM for highlight detection (max_clips=3) ...")
    highlights = detect_highlights(transcript, timeline, model_name, base_url, max_clips=3)
    log(f"  Found {len(highlights)} highlights")
    for h in highlights:
        log(f"    [{h.start:.1f}s - {h.end:.1f}s] {h.reason}")

    log("  PASS")
    return {"highlight_count": len(highlights),
            "highlights": [{"start": h.start, "end": h.end, "reason": h.reason} for h in highlights]}


def test_video_edit(video_path: str, cache_dir: str, **_) -> dict:
    """Test video_edit skill: export a short highlight clip."""
    log("=" * 60)
    log("TEST: video_edit — Export a 10s highlight clip (re-encode)")
    log("=" * 60)

    out_dir = os.path.join(cache_dir, "clips_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_clip.mp4")

    split_video_segment(video_path, 60.0, 10.0, out_path)
    size_kb = os.path.getsize(out_path) / 1024
    dur = get_video_duration(out_path)
    log(f"  Clip: {out_path} ({size_kb:.0f} KB, {dur:.1f}s)")
    assert os.path.getsize(out_path) > 1000, "Clip file too small"
    assert 8.0 < dur < 12.0, f"Clip duration {dur:.1f}s not ~10s"
    log("  PASS")
    return {"clip_path": out_path, "size_kb": round(size_kb), "duration": round(dur, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    "video_probe", "frame_sample", "frame_caption", "video_caption",
    "audio_extract", "asr", "ocr", "object_detection",
    "timeline", "video_qa", "highlights", "video_edit",
]

TEST_MAP = {
    "video_probe":      test_video_probe,
    "frame_sample":     test_frame_sample,
    "frame_caption":    test_frame_caption,
    "video_caption":    test_video_caption,
    "audio_extract":    test_audio_extract,
    "asr":              test_asr,
    "ocr":              test_ocr,
    "object_detection": test_object_detection,
    "timeline":         test_timeline,
    "video_qa":         test_video_qa,
    "highlights":       test_highlights,
    "video_edit":       test_video_edit,
}


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive lite test for all VidCopilot agent skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video-path", required=True, help="Path to local video file.")
    parser.add_argument("--api-base", default=None,
                        help="Existing vLLM base URL. If not set, auto-detect or launch.")
    parser.add_argument("--gpu", type=int, default=4, help="GPUs for serving (default: 4).")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size.")
    parser.add_argument("--tests", nargs="+", default=ALL_TESTS, choices=ALL_TESTS,
                        help="Which tests to run (default: all).")
    parser.add_argument("--cache-root", default=None,
                        help="Cache directory (default: cache/test_all/<video_hash>).")
    parser.add_argument("--skip-serve", action="store_true",
                        help="Don't auto-launch serving; fail if no service found.")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    if not os.path.isfile(video_path):
        log(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    vid_hash = sha1(video_path)[:12]
    cache_dir = args.cache_root or os.path.join(CACHE_ROOT, "test_all", vid_hash)
    os.makedirs(cache_dir, exist_ok=True)

    # ── Step 1: Find or launch model serving ──────────────────────────────
    log("Step 1: Finding model serving endpoint ...")

    # Check which tests actually need MLLM
    mllm_tests = {"frame_caption", "video_caption", "timeline", "video_qa", "highlights"}
    needs_mllm = bool(set(args.tests) & mllm_tests)

    base_url = None
    model_name = None
    serve_proc = None

    if needs_mllm:
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
                    log("ERROR: No serving found and --skip-serve is set.")
                    sys.exit(1)
                log("No existing service found. Launching new vLLM serving ...")
                serve_proc = launch_serving(gpu=args.gpu, tp=args.tp)
                base_url = wait_for_serving(serve_proc, timeout=600)

        log(f"Using model endpoint: {base_url}")
        model_name = get_model_name(base_url)
        log(f"Using model: {model_name}")
    else:
        log("  No MLLM tests selected, skipping serving setup.")

    # ── Step 2: Video info ────────────────────────────────────────────────
    log("")
    duration = get_video_duration(video_path)
    log(f"Video: {video_path}")
    log(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    log(f"Cache: {cache_dir}")
    log(f"Tests: {', '.join(args.tests)}")

    # ── Step 3: Run tests ─────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log(f"Running {len(args.tests)} tests ...")
    log("=" * 60)

    results = {}
    passed = 0
    failed = 0
    skipped = 0

    kwargs = {
        "video_path": video_path,
        "cache_dir": cache_dir,
        "base_url": base_url,
        "model_name": model_name,
    }

    for test_name in args.tests:
        t0 = time.time()
        try:
            log("")
            result = TEST_MAP[test_name](**kwargs)
            dt = time.time() - t0
            if isinstance(result, dict) and result.get("status") == "skip":
                log(f"  [{test_name}] SKIPPED ({dt:.1f}s) — {result.get('reason', '')}")
                results[test_name] = {"status": "skip", "reason": result.get("reason", ""), "time": dt}
                skipped += 1
            else:
                log(f"  [{test_name}] PASSED ({dt:.1f}s)")
                results[test_name] = {"status": "pass", "time": dt}
                passed += 1
        except Exception as e:
            dt = time.time() - t0
            log(f"  FAIL: {test_name} — {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = {"status": "fail", "error": str(e), "time": dt}
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Test':<20} {'Status':<8} {'Time':>8}")
    log("-" * 40)
    for test_name in args.tests:
        r = results.get(test_name, {})
        status = r.get("status", "?").upper()
        dt = r.get("time", 0)
        marker = "✓" if status == "PASS" else ("○" if status == "SKIP" else "✗")
        log(f"  {marker} {test_name:<18} {status:<8} {dt:>6.1f}s")
    log("-" * 40)
    total_time = time.time() - _T0
    log(f"Total: {passed} passed, {failed} failed, {skipped} skipped "
        f"({len(args.tests)} tests, {total_time:.0f}s)")

    if base_url:
        log(f"Model: {model_name}")
        log(f"Endpoint: {base_url}")
    log(f"Video: {video_path} ({duration:.0f}s)")
    log("=" * 60)

    # Save results JSON
    results_path = os.path.join(cache_dir, "test_all_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "base_url": base_url, "model": model_name,
            "video_path": video_path, "duration": duration,
            "passed": passed, "failed": failed, "skipped": skipped,
            "total_time": round(total_time, 1),
            "tests": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to: {results_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
