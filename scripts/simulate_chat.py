#!/usr/bin/env python3
"""Simulate interactive chat sessions for testing on D-cluster / GPU.

Runs a series of preset questions through VideoChat and saves the transcript.
Useful for testing the chat pipeline end-to-end without a real terminal.

Usage:
    # With a running vLLM server
    python scripts/simulate_chat.py --video cache/downloads/SSya123u9Yk.mp4 \
        --api-base http://localhost:8000/v1

    # Auto-detect model from vLLM
    python scripts/simulate_chat.py --video cache/downloads/SSya123u9Yk.mp4

    # Custom questions from file (one per line)
    python scripts/simulate_chat.py --video cache/downloads/SSya123u9Yk.mp4 \
        --questions questions.txt
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.extensions.skills.video_io import load_video
from agent.extensions.skills.persist import load_analysis, save_analysis
from agent.core.orchestrator import run
from agent.config import get_default_config, load_config
from agent.chat import VideoChat, _format_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("simulate_chat")

# ── Default questions for podcast-style videos ──────────────────────────

DEFAULT_QUESTIONS = [
    # Factual — should answer from ASR/timeline alone
    "What are the main topics discussed in this video?",
    "Who are the speakers and what are their backgrounds?",
    "What is the key argument or thesis presented?",
    "Summarize the discussion about AI breakthroughs mentioned in the video.",
    "What predictions or opinions about the future are shared?",

    # Visual — should trigger on-demand frame analysis
    "/visual What does the recording studio look like? Describe the setting.",
    "What text or graphics appear on screen during the discussion?",

    # Emotion / mood — should trigger targeted analysis
    "Does the guest seem enthusiastic or skeptical about the future of AI?",
    "What is the overall mood of the conversation?",

    # Follow-up / drill-down
    "Can you give me more detail on the first topic discussed?",
    "What timestamps should I jump to for the most interesting moments?",
]


def wait_for_vllm(base_url: str, timeout: int = 600) -> str:
    """Wait for vLLM to be ready, return the served model name."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    model_name = models[0]["id"]
                    logger.info("vLLM ready, model: %s", model_name)
                    return model_name
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"vLLM not ready after {timeout}s")


def run_analysis_if_needed(video_path: str, cache_root: str, cfg: dict) -> tuple:
    """Load cached analysis or run a new one. Returns (asset, analysis)."""
    asset = load_video("local", video_path, cache_root)

    try:
        analysis = load_analysis(asset.cache_dir)
        # Check if analysis has meaningful data (not the broken force_visual run)
        asr_segs = analysis.get("asr", {}).get("segments", [])
        timeline_chapters = (analysis.get("timeline", {}).get("chapters", [])
                             if isinstance(analysis.get("timeline"), dict) else [])
        if asr_segs or timeline_chapters:
            logger.info("Loaded cached analysis: %d ASR segments, %d chapters",
                        len(asr_segs), len(timeline_chapters))
            return asset, analysis
        logger.info("Cached analysis has no ASR/timeline data, re-running...")
    except Exception:
        logger.info("No cached analysis found, running new analysis...")

    # Run detailed analysis (no force_visual)
    result = run(asset, "detailed", cfg)
    analysis = load_analysis(asset.cache_dir)
    return asset, analysis


def simulate_chat(asset, analysis, client, model, questions, output_path=None):
    """Run a series of questions through VideoChat and print/save results."""
    console = Console()

    session = VideoChat(
        asset=asset,
        analysis=analysis,
        chat_client=client,
        chat_model=model,
    )

    # Show summary
    console.print()
    title = session.content_meta.get("title", "Video")
    duration = _format_timestamp(session.video_meta.get("duration_sec", 0))
    console.print(Panel(
        f"[bold]{title}[/bold]\n"
        f"Duration: {duration}  |  "
        f"ASR: {session.sufficiency.get('transcript_word_count', '?')} words  |  "
        f"Chapters: {len(session.chapters)}",
        title="[bold cyan]VidCopilot Chat Simulation[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    summary = session.get_summary()
    console.print(Markdown(summary))
    console.print()

    # Run questions
    transcript = {
        "video": title,
        "duration": duration,
        "summary": summary,
        "questions": [],
    }

    for i, q in enumerate(questions, 1):
        console.print(f"[bold]({i}/{len(questions)})[/bold] [cyan]> {q}[/cyan]")

        force_visual = False
        actual_q = q
        if q.startswith("/visual "):
            force_visual = True
            actual_q = q[len("/visual "):]

        start_time = time.time()
        try:
            answer = session.answer(actual_q, force_visual=force_visual)
        except Exception as e:
            answer = f"[ERROR] {e}"
            logger.error("Error answering question %d: %s", i, e)
        elapsed = time.time() - start_time

        console.print()
        console.print(Markdown(answer))
        console.print(f"[dim]({elapsed:.1f}s)[/dim]")
        console.print("---")
        console.print()

        transcript["questions"].append({
            "question": q,
            "answer": answer,
            "elapsed_sec": round(elapsed, 1),
            "force_visual": force_visual,
        })

    # Save transcript
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        console.print(f"\n[bold green]Transcript saved to {output_path}[/bold green]")

    return transcript


def main():
    parser = argparse.ArgumentParser(description="Simulate VidCopilot interactive chat")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if not given)")
    parser.add_argument("--cache-root", default="./cache", help="Cache root directory")
    parser.add_argument("--questions", default=None, help="Path to questions file (one per line)")
    parser.add_argument("--output", default=None, help="Output path for chat transcript JSON")
    parser.add_argument("--wait-vllm", action="store_true", help="Wait for vLLM to be ready")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Wait for vLLM if requested
    if args.wait_vllm:
        model = wait_for_vllm(args.api_base)
    elif args.model:
        model = args.model
    else:
        # Try to auto-detect
        try:
            model = wait_for_vllm(args.api_base, timeout=10)
        except TimeoutError:
            model = "qwen3.5-9b"
            logger.warning("Could not detect model, defaulting to %s", model)

    if args.model:
        model = args.model

    # Load config
    cfg = {**get_default_config(), **load_config(args.config)}
    cfg["llm_base_url"] = args.api_base
    cfg["llm_model"] = model

    # Load or run analysis
    asset, analysis = run_analysis_if_needed(args.video, args.cache_root, cfg)

    # Load questions
    if args.questions and os.path.exists(args.questions):
        with open(args.questions) as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        questions = DEFAULT_QUESTIONS

    # Default output path
    output = args.output
    if not output:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output = os.path.join(args.cache_root, f"chat_simulation_{video_name}.json")

    # Create client and run
    client = OpenAI(base_url=args.api_base, api_key="EMPTY", timeout=120.0)
    simulate_chat(asset, analysis, client, model, questions, output_path=output)


if __name__ == "__main__":
    main()
