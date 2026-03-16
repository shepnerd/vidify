#!/usr/bin/env python3
"""
Test script for ask workflow.
Usage: python scripts/test_ask.py --video-path /path/to/video.mp4 --cache-root ./cache --question "What is the video about?"
Assumes index is already built.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skills.video_io import load_video
from agent.workflows.ask import wf_ask

from agent.skills.persist import load_analysis

def main():
    parser = argparse.ArgumentParser(description="Test ask workflow.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--direct-model", action="store_true", help="Use direct model loading.")
    parser.add_argument("--model-path", default="/models/qwen-vl", help="Path to model.")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    asset = load_video("local", args.video_path, args.cache_root)

    # Check if there's an index
    try:
        analysis = load_analysis(asset.cache_dir)
        rag = (analysis.get("rag") or {}).get("faiss")
        if not rag:
            print("No FAISS index found. Skipping ask test.")
            return
    except Exception as e:
        print(f"Could not load analysis: {e}. Skipping ask test.")
        return

    result = wf_ask(
        asset, args.question,
        llm_base_url="http://localhost:8000/v1",
        llm_model="qwen-vl",
        embed_base_url="http://localhost:8000/v1",
        embed_model="qwen-embed",
        top_k=5,
        direct_model=args.direct_model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    print("Ask completed.")
    print(f"Answer: {result['result']['answer']}")

if __name__ == "__main__":
    main()