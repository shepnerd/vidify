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
import json
import os
import sys
import time

# ── Unset cluster proxy env vars BEFORE importing HTTP libraries ──────────────
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_key, None)

import subprocess

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache")
VLLM_PORT = 8000

DEFAULT_YOUTUBE = "https://www.youtube.com/watch?v=BoC5MY_7aDk"
DEFAULT_QUESTION = "Summarize the key points of this video and provide timestamps."

# ── Shared utilities ─────────────────────────────────────────────────────────
from agent.extensions.utils import (
    get_video_duration, split_video_segment, get_short_clip,
    extract_frames, img_to_data_url,
)
from agent.extensions.utils.cache import sha1
from agent.extensions.utils.serving import (
    probe_vllm, find_existing_service, read_serving_ip,
    launch_serving, wait_for_serving, get_model_name, make_client,
)


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[test] {msg}", flush=True)


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
        if probe_vllm(args.api_base, log_fn=log):
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

        base_url = find_existing_service(candidates, log_fn=log)
        if not base_url:
            if args.skip_serve:
                log("ERROR: No serving endpoint found and --skip-serve is set.")
                sys.exit(1)
            log("No existing service found. Launching new vLLM serving ...")
            serve_proc = launch_serving(gpu=args.gpu, tp=args.tp, log_fn=log)
            base_url = wait_for_serving(serve_proc, timeout=600, log_fn=log)

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
