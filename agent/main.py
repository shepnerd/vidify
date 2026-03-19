# agent/main.py
import argparse
from agent.skills.video_io import load_video
from agent.orchestrator import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-type", required=True, choices=["youtube", "url", "local"])
    ap.add_argument("--uri", required=True)
    ap.add_argument("--mode", required=True, choices=["quick", "detailed", "highlights", "index", "ask", "report"])
    ap.add_argument("--cache-root", default="./cache")

    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="qwen-vl")
    ap.add_argument("--embed-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--embed-model", default="qwen-embed")

    ap.add_argument("--direct-model", action="store_true", help="Use direct model loading instead of API")
    ap.add_argument("--model-path", default="/models/qwen-vl", help="Path to the model for direct loading")
    ap.add_argument("--tokenizer-path", default=None, help="Path to tokenizer, defaults to model-path")

    ap.add_argument("--question", default=None)
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--chunk-sec", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--include-web-search", action="store_true", help="Include web search enhancement in analysis")
    ap.add_argument("--analysis-type", default="brief", choices=["brief", "detailed"], help="Analysis type for report generation")
    ap.add_argument("--google-api-key", help="Google Custom Search API key for web search")
    ap.add_argument("--google-search-engine-id", help="Google Custom Search Engine ID for web search")
    args = ap.parse_args()

    asset = load_video(args.source_type, args.uri, args.cache_root)
    cfg = vars(args)
    out = run(asset, args.mode, cfg)
    print(out)

if __name__ == "__main__":
    main()