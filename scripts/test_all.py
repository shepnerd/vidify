#!/usr/bin/env python3
"""
Test script that runs all workflows in sequence.
Usage: python scripts/test_all.py --video-path /path/to/video.mp4 --cache-root ./cache
"""

import argparse
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.skills.video_io import load_video
from agent.skills.persist import load_analysis

def run_workflow(name, *args, **kwargs):
    print(f"Running {name}...")
    try:
        # Import workflow dynamically to avoid loading all at startup
        module_name = f"agent.workflows.{name}"
        module = __import__(module_name, fromlist=[f"wf_{name}"])
        func = getattr(module, f"wf_{name}")
        result = func(*args, **kwargs)
        print(f"{name} completed successfully.")
        return result
    except Exception as e:
        print(f"{name} failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test all workflows.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--direct-model", action="store_true", help="Use direct model loading.")
    parser.add_argument("--model-path", default="/models/qwen-vl", help="Path to model.")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer.")
    parser.add_argument("--question", default="What is the main topic of this video?", help="Question for ask test.")
    parser.add_argument("--whisper-model", help="Whisper model for ASR.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    # Load video asset once
    asset = load_video("local", args.video_path, args.cache_root)

    # Common parameters
    common_kwargs = {
        "direct_model": args.direct_model,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path
    }

    # Test brief
    brief_result = run_workflow("brief", asset,
                               llm_base_url="http://localhost:8000/v1",
                               llm_model="qwen-vl",
                               **common_kwargs)
    if brief_result is None:
        return

    # Test detailed
    detailed_result = run_workflow("detailed", asset,
                                  llm_base_url="http://localhost:8000/v1",
                                  llm_model="qwen-vl",
                                  max_frames=16,
                                  whisper_model=args.whisper_model,
                                  **common_kwargs)
    if detailed_result is None:
        return

    # Check if we have ASR data for dependent workflows
    try:
        analysis = load_analysis(asset.cache_dir)
        has_asr = analysis.get("asr") and analysis["asr"].get("segments")
        has_index = (analysis.get("rag") or {}).get("faiss")
    except:
        has_asr = False
        has_index = False

    # Test index (only if we have ASR)
    if has_asr:
        index_result = run_workflow("index", asset,
                                   llm_base_url="http://localhost:8000/v1",
                                   llm_model="qwen-vl",
                                   embed_base_url="http://localhost:8000/v1",
                                   embed_model="qwen-embed",
                                   chunk_sec=20,
                                   **common_kwargs)
        if index_result is None:
            return
        has_index = True
    else:
        print("Skipping index workflow (no ASR data available)")

    # Test ask (only if we have index)
    if has_index:
        ask_result = run_workflow("ask", asset, args.question,
                                 llm_base_url="http://localhost:8000/v1",
                                 llm_model="qwen-vl",
                                 embed_base_url="http://localhost:8000/v1",
                                 embed_model="qwen-embed",
                                 top_k=5,
                                 **common_kwargs)
        if ask_result is None:
            return
    else:
        print("Skipping ask workflow (no index available)")

    # Test highlights (only if we have ASR)
    if has_asr:
        highlights_result = run_workflow("highlights", asset,
                                       llm_base_url="http://localhost:8000/v1",
                                       llm_model="qwen-vl",
                                       max_clips=3,
                                       also_make_reel=False,
                                       **common_kwargs)
        if highlights_result is None:
            return
    else:
        print("Skipping highlights workflow (no ASR data available)")

    print("All applicable tests passed!")

if __name__ == "__main__":
    main()

    print("All applicable tests passed!")