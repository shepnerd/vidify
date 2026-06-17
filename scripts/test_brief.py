#!/usr/bin/env python3
"""
Test script for brief workflow.
Usage: python scripts/test_brief.py --video-path /path/to/video.mp4 --cache-root ./cache
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.extensions.skills.video_io import load_video
from agent.extensions.workflows.brief import wf_brief

def main():
    parser = argparse.ArgumentParser(description="Test brief workflow.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible LLM endpoint.")
    parser.add_argument("--llm-model", default="qwen3.5-9b", help="Model name served by the LLM endpoint.")
    parser.add_argument("--direct-model", action="store_true", help="Use direct model loading.")
    parser.add_argument("--model-path", default=None, help="Path to model. Required with --direct-model.")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer.")

    args = parser.parse_args()
    if args.direct_model and not args.model_path:
        parser.error("--model-path is required with --direct-model")

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    asset = load_video("local", args.video_path, args.cache_root)

    result = wf_brief(
        asset,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        max_frames=16,
        direct_model=args.direct_model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    print("Brief analysis completed.")
    print(f"Frames: {len(result['frames']['items'])}")
    print(f"Timeline chapters: {len(result['timeline'].get('chapters', []))}")

if __name__ == "__main__":
    main()
