#!/usr/bin/env python3
"""Compare qwen3.5 vs qwen3-mla: raw serving metrics + full pipeline comparison.

Usage:
    python scripts/compare_models.py \
        --video media/taste_in_china_s1e1.mp4 \
        --api-base-a http://host:8000/v1 \
        --api-base-b http://host:8001/v1

    # With model name overrides:
    python scripts/compare_models.py \
        --api-base-a http://host:8000/v1 --model-a qwen3.5-9b \
        --api-base-b http://host:8001/v1 --model-b qwen3-mla \
        --video media/taste_in_china_s1e1.mp4
"""
import argparse
import json
import os
import sys
import time

from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.extensions.utils import make_video_content, img_to_data_url
from agent.extensions.models.thinking import strip_thinking


def make_client(base_url: str) -> OpenAI:
    # Strip proxy env vars that corrupt multimodal payloads
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=300)


def discover_model_name(client: OpenAI) -> str:
    """Get the served model name from the vLLM /models endpoint."""
    models = client.models.list()
    if models.data:
        return models.data[0].id
    raise RuntimeError("No models found on the endpoint")


# ── Level 1: Raw serving metrics ────────────────────────────────────────────

def bench_text_prompt(client: OpenAI, model: str, prompt: str,
                      max_tokens: int = 256, runs: int = 3) -> dict:
    """Benchmark a text-only prompt. Returns avg latency and tokens/sec."""
    latencies = []
    token_counts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=0.2,
        )
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        text = strip_thinking(text)
        n_tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
        latencies.append(elapsed)
        token_counts.append(n_tokens)

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(token_counts) / len(token_counts)
    return {
        "avg_latency_s": round(avg_lat, 3),
        "avg_tokens": round(avg_tok, 1),
        "avg_tok_per_s": round(avg_tok / avg_lat, 1) if avg_lat > 0 else 0,
        "runs": runs,
    }


def bench_video_prompt(client: OpenAI, model: str, video_path: str,
                       prompt: str, max_tokens: int = 512, runs: int = 2) -> dict:
    """Benchmark a video+text prompt."""
    latencies = []
    token_counts = []
    outputs = []
    for _ in range(runs):
        content = [
            {"type": "text", "text": prompt},
            make_video_content(video_path),
        ]
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_tokens,
            temperature=0.2,
        )
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        text = strip_thinking(text)
        n_tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
        latencies.append(elapsed)
        token_counts.append(n_tokens)
        outputs.append(text)

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(token_counts) / len(token_counts)
    return {
        "avg_latency_s": round(avg_lat, 3),
        "avg_tokens": round(avg_tok, 1),
        "avg_tok_per_s": round(avg_tok / avg_lat, 1) if avg_lat > 0 else 0,
        "runs": runs,
        "sample_output": outputs[0][:500],
    }


def bench_image_prompt(client: OpenAI, model: str, image_path: str,
                       prompt: str, max_tokens: int = 256, runs: int = 3) -> dict:
    """Benchmark an image+text prompt."""
    data_url = img_to_data_url(image_path)
    latencies = []
    token_counts = []
    for _ in range(runs):
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_tokens,
            temperature=0.2,
        )
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        text = strip_thinking(text)
        n_tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
        latencies.append(elapsed)
        token_counts.append(n_tokens)

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(token_counts) / len(token_counts)
    return {
        "avg_latency_s": round(avg_lat, 3),
        "avg_tokens": round(avg_tok, 1),
        "avg_tok_per_s": round(avg_tok / avg_lat, 1) if avg_lat > 0 else 0,
        "runs": runs,
    }


# ── Level 2: Full pipeline comparison ───────────────────────────────────────

def run_pipeline(video_path: str, api_base: str, model_name: str) -> dict:
    """Run VidCopilot brief analysis and measure wall time."""
    from agent.core.orchestrator import run
    from agent.core.video_loader import load_video

    t0 = time.perf_counter()
    asset = load_video("local", video_path)
    result = run(
        asset,
        mode="brief",
        llm_base_url=api_base,
        llm_model=model_name,
    )
    elapsed = time.perf_counter() - t0

    # Extract key metrics from result
    summary = ""
    if hasattr(result, "summary"):
        summary = (result.summary or "")[:500]
    elif isinstance(result, dict):
        summary = str(result.get("summary", ""))[:500]

    return {
        "wall_time_s": round(elapsed, 2),
        "summary_preview": summary,
    }


# ── Display ─────────────────────────────────────────────────────────────────

def print_comparison(label: str, result_a: dict, result_b: dict,
                     name_a: str, name_b: str):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    keys = sorted(set(list(result_a.keys()) + list(result_b.keys())))
    # Separate numeric keys from text keys
    for k in keys:
        va = result_a.get(k, "N/A")
        vb = result_b.get(k, "N/A")
        if k in ("sample_output", "summary_preview"):
            continue
        speedup = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if "latency" in k or "time" in k:
                if vb > 0:
                    speedup = f"  ({va/vb:.2f}x)" if va > vb else f"  ({vb/va:.2f}x faster)"
            elif "tok_per_s" in k:
                if va > 0:
                    speedup = f"  ({vb/va:.2f}x)" if vb > va else f"  ({va/vb:.2f}x faster)"
        print(f"  {k:25s}  {name_a}: {va:>12}  {name_b}: {vb:>12}{speedup}")

    # Print sample outputs if present
    for k in ("sample_output", "summary_preview"):
        if k in result_a or k in result_b:
            print(f"\n  --- {name_a} {k} ---")
            print(f"  {result_a.get(k, 'N/A')[:300]}")
            print(f"\n  --- {name_b} {k} ---")
            print(f"  {result_b.get(k, 'N/A')[:300]}")


def main():
    parser = argparse.ArgumentParser(description="Compare two model endpoints")
    parser.add_argument("--video", default=os.path.join(PROJECT_ROOT, "media/taste_in_china_s1e1.mp4"))
    parser.add_argument("--api-base-a", required=True, help="Endpoint A (e.g. qwen3.5)")
    parser.add_argument("--api-base-b", required=True, help="Endpoint B (e.g. qwen3-mla)")
    parser.add_argument("--model-a", default=None, help="Model name for A (auto-detected if omitted)")
    parser.add_argument("--model-b", default=None, help="Model name for B (auto-detected if omitted)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per benchmark")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip full pipeline comparison")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw serving benchmarks")
    parser.add_argument("--skip-video", action="store_true", help="Skip video benchmark (useful for single-GPU servers that may OOM)")
    args = parser.parse_args()

    client_a = make_client(args.api_base_a)
    client_b = make_client(args.api_base_b)

    model_a = args.model_a or discover_model_name(client_a)
    model_b = args.model_b or discover_model_name(client_b)

    name_a = f"A ({model_a})"
    name_b = f"B ({model_b})"

    print(f"Model A: {model_a} @ {args.api_base_a}")
    print(f"Model B: {model_b} @ {args.api_base_b}")
    print(f"Video:   {args.video}")

    if not args.skip_raw:
        # --- Text-only benchmark ---
        print("\n[1/3] Text-only prompt benchmark...")
        text_prompt = "请列举中国四大名著及其作者，每本书用一句话概括主要内容。"
        ra = bench_text_prompt(client_a, model_a, text_prompt, runs=args.runs)
        rb = bench_text_prompt(client_b, model_b, text_prompt, runs=args.runs)
        print_comparison("Text-only Prompt", ra, rb, name_a, name_b)

        # --- Video benchmark ---
        if not args.skip_video:
            print("\n[2/3] Video prompt benchmark...")
            video_prompt = "请描述这段视频的主要内容，包括场景、人物和关键动作。"
            ra = bench_video_prompt(client_a, model_a, args.video, video_prompt, runs=min(args.runs, 2))
            rb = bench_video_prompt(client_b, model_b, args.video, video_prompt, runs=min(args.runs, 2))
            print_comparison("Video Prompt", ra, rb, name_a, name_b)
        else:
            print("\n[2/3] Video prompt benchmark... SKIPPED (--skip-video)")

        # --- Extract a frame for image benchmark ---
        print("\n[3/3] Image prompt benchmark...")
        import subprocess
        frame_path = "/tmp/_compare_frame.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-i", args.video, "-ss", "10", "-frames:v", "1",
            "-q:v", "2", frame_path
        ], capture_output=True)
        if os.path.exists(frame_path):
            image_prompt = "请描述这张图片中的内容。"
            ra = bench_image_prompt(client_a, model_a, frame_path, image_prompt, runs=args.runs)
            rb = bench_image_prompt(client_b, model_b, frame_path, image_prompt, runs=args.runs)
            print_comparison("Image Prompt", ra, rb, name_a, name_b)
        else:
            print("  [SKIP] Could not extract frame from video")

    if not args.skip_pipeline:
        print("\n[Pipeline] Full VidCopilot brief analysis comparison...")
        print(f"  Running pipeline with {name_a}...")
        try:
            pa = run_pipeline(args.video, args.api_base_a, model_a)
        except Exception as e:
            pa = {"wall_time_s": -1, "summary_preview": f"ERROR: {e}"}
            print(f"  [ERROR] Pipeline A failed: {e}")

        print(f"  Running pipeline with {name_b}...")
        try:
            pb = run_pipeline(args.video, args.api_base_b, model_b)
        except Exception as e:
            pb = {"wall_time_s": -1, "summary_preview": f"ERROR: {e}"}
            print(f"  [ERROR] Pipeline B failed: {e}")

        print_comparison("Full Pipeline (brief mode)", pa, pb, name_a, name_b)

    print("\n" + "="*70)
    print("  Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
