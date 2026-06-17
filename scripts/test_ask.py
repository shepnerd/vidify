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

from agent.extensions.skills.video_io import load_video
from agent.extensions.workflows.ask import wf_ask

from agent.extensions.skills.persist import load_analysis

def main():
    parser = argparse.ArgumentParser(description="Test ask workflow.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible LLM endpoint.")
    parser.add_argument("--llm-model", default="qwen3.5-9b", help="Model name served by the LLM endpoint.")
    parser.add_argument("--embed-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible embedding endpoint.")
    parser.add_argument("--embed-model", default="qwen-embed", help="Embedding model name.")
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
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        embed_base_url=args.embed_base_url,
        embed_model=args.embed_model,
        top_k=5,
        direct_model=args.direct_model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    print("Ask completed.")
    print(f"Answer: {result['result']['answer']}")

if __name__ == "__main__":
    main()
