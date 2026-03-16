#!/usr/bin/env python3
"""
Script to analyze a local video file and generate summary/description.
Usage: python scripts/local_video_summary.py --video-path /path/to/video.mp4 --cache-root ./cache --mode detailed
"""

import argparse
import os
import sys

# Add the parent directory to the Python path to import agent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skills.video_io import load_video
from agent.workflows.detailed import wf_detailed
from agent.workflows.brief import wf_brief

def main():
    parser = argparse.ArgumentParser(description="Generate summary/description for a local video file.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--cache-root", default="./cache", help="Cache directory root.")
    parser.add_argument("--mode", choices=["brief", "detailed"], default="detailed",
                        help="Analysis mode: brief (quick summary) or detailed (full analysis with ASR).")
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum frames to sample.")
    parser.add_argument("--whisper-model", default=None, help="Whisper model size or local path. If None, skip ASR for offline processing.")
    parser.add_argument("--direct-model", action="store_true", help="Use direct model loading.")
    parser.add_argument("--model-path", default="/models/qwen-vl", help="Path to model for direct loading.")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer (optional).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3, help="GPU memory utilization for vLLM (0.0-1.0).")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    # Load video asset
    asset = load_video("local", args.video_path, args.cache_root)

    # Run analysis
    if args.mode == "detailed":
        result = wf_detailed(
            asset,
            llm_base_url="http://localhost:8000/v1",  # Default, but will be overridden if direct_model
            llm_model="qwen-vl",
            max_frames=args.max_frames,
            whisper_model=args.whisper_model,
            direct_model=args.direct_model,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path
        )
    else:
        result = wf_brief(
            asset,
            llm_base_url="http://localhost:8000/v1",
            llm_model="qwen-vl",
            max_frames=args.max_frames,
            direct_model=args.direct_model,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path
        )

    # Print summary
    print("Video Summary/Description:")
    print(f"Video ID: {result['video']['id']}")
    print(f"Duration: {result['video']['duration_sec']:.2f} seconds")
    print(f"Resolution: {result['video']['width']}x{result['video']['height']}")
    print(f"Frames analyzed: {len(result['frames']['items'])}")
    if result['timeline']:
        print("Timeline:")
        for chapter in result['timeline'].get('chapters', []):
            print(f"  {chapter['start']:.1f}s - {chapter['end']:.1f}s: {chapter['title']} - {chapter['summary']}")
        for event in result['timeline'].get('events', []):
            print(f"  {event['start']:.1f}s - {event['end']:.1f}s: {event['text']}")
    if result['asr']:
        print(f"ASR segments: {len(result['asr']['segments'])}")

    # Save to file
    output_file = os.path.join(asset.cache_dir, "summary.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Video Summary/Description:\n")
        f.write(f"Video ID: {result['video']['id']}\n")
        f.write(f"Duration: {result['video']['duration_sec']:.2f} seconds\n")
        f.write(f"Resolution: {result['video']['width']}x{result['video']['height']}\n")
        f.write(f"Frames analyzed: {len(result['frames']['items'])}\n")
        if result['timeline']:
            f.write("Timeline:\n")
            for chapter in result['timeline'].get('chapters', []):
                f.write(f"  {chapter['start']:.1f}s - {chapter['end']:.1f}s: {chapter['title']} - {chapter['summary']}\n")
            for event in result['timeline'].get('events', []):
                f.write(f"  {event['start']:.1f}s - {event['end']:.1f}s: {event['text']}\n")
        if result['asr']:
            f.write(f"ASR segments: {len(result['asr']['segments'])}\n")

    print(f"Full results saved to: {asset.cache_dir}/analysis.json")
    print(f"Summary saved to: {output_file}")

if __name__ == "__main__":
    main()