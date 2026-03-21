#!/usr/bin/env python3
"""
Test script for index workflow.
Usage: python scripts/test_index.py --video-path /path/to/video.mp4 --cache-root ./cache
Assumes detailed analysis is already done.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.extensions.skills.video_io import load_video
from agent.workflows.index import wf_index

from agent.extensions.skills.persist import load_analysis
from agent.extensions.skills.deserialize import load_transcript

def main():
    parser = argparse.ArgumentParser(description="Test index workflow.")
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

    # Check if there's transcript data
    try:
        analysis = load_analysis(asset.cache_dir)
        if not analysis.get("asr") or not analysis["asr"].get("segments"):
            print("No ASR segments found. Skipping index creation.")
            return
        transcript = load_transcript(analysis["asr"])
        if not transcript.segments:
            print("No transcript segments found. Skipping index creation.")
            return
    except Exception as e:
        print(f"Could not load analysis: {e}. Skipping index creation.")
        return

    result = wf_index(
        asset,
        llm_base_url="http://localhost:8000/v1",
        llm_model="qwen-vl",
        embed_base_url="http://localhost:8000/v1",
        embed_model="qwen-embed",
        chunk_sec=20,
        direct_model=args.direct_model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    print("Index creation completed.")
    print(f"RAG: {result}")

if __name__ == "__main__":
    main()