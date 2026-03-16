#!/usr/bin/env python3
"""
Test script for detailed workflow.
Usage: python scripts/test_detailed.py --video-path /path/to/video.mp4 --cache-root ./cache
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skills.video_io import load_video
from agent.workflows.detailed import wf_detailed

def main():
    parser = argparse.ArgumentParser(description="Test detailed workflow.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--whisper-model", default=None, help="Whisper model.")
    parser.add_argument("--direct-model", action="store_true", help="Use direct model loading.")
    parser.add_argument("--model-path", default="/models/qwen-vl", help="Path to model.")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    asset = load_video("local", args.video_path, args.cache_root)

    result = wf_detailed(
        asset,
        llm_base_url="http://localhost:8000/v1",
        llm_model="qwen-vl",
        max_frames=16,
        whisper_model=args.whisper_model,
        direct_model=args.direct_model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    print("Detailed analysis completed.")
    print(f"Frames: {len(result['frames']['items'])}")
    print(f"ASR segments: {len(result['asr']['segments'])}")
    print(f"Timeline chapters: {len(result['timeline'].get('chapters', []))}")

if __name__ == "__main__":
    main()