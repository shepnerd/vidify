#!/usr/bin/env python3
"""
Test script for highlights workflow.
Usage: python scripts/test_highlights.py --video-path /path/to/video.mp4 --cache-root ./cache
Assumes detailed analysis is done.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skills.video_io import load_video
from agent.workflows.highlights import wf_highlights

from agent.skills.persist import load_analysis

def main():
    parser = argparse.ArgumentParser(description="Test highlights workflow.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--direct-model", action="store_true", help="Use direct model loading.")
    parser.add_argument("--model-path", default="/models/qwen-vl", help="Path to model.")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    asset = load_video("local", args.video_path, args.cache_root)

    # Check if there's ASR data
    try:
        analysis = load_analysis(asset.cache_dir)
        if not analysis.get("asr") or not analysis["asr"].get("segments"):
            print("No ASR segments found. Skipping highlights test.")
            return
    except Exception as e:
        print(f"Could not load analysis: {e}. Skipping highlights test.")
        return

    result = wf_highlights(
        asset,
        llm_base_url="http://localhost:8000/v1",
        llm_model="qwen-vl",
        max_clips=3,
        also_make_reel=False,
        direct_model=args.direct_model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    print("Highlights completed.")
    print(f"Highlights: {len(result['highlights'])}")

if __name__ == "__main__":
    main()