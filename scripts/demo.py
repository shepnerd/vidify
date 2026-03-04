import argparse
import json
import time
import requests

def post(url: str, payload: dict, timeout=3600):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:9000")
    ap.add_argument("--youtube", required=True, help="YouTube URL")
    ap.add_argument("--cache-root", default="./cache")

    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="qwen-vl")

    ap.add_argument("--embed-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--embed-model", default="qwen-embed")

    ap.add_argument("--question", default="总结视频的关键结论，并给出对应时间段证据。")
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--chunk-sec", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-clips", type=int, default=5)
    args = ap.parse_args()

    src = {"source_type": "youtube", "uri": args.youtube, "cache_root": args.cache_root}

    print("1) analyze(detailed)...")
    analyze = post(f"{args.server}/analyze", {
        **src,
        "mode": "detailed",
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "max_frames": args.max_frames,
        "whisper_model": "small",
    })
    print("analyze ok. keys:", list(analyze.keys()))

    print("2) index(faiss)...")
    idx = post(f"{args.server}/index", {
        **src,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "embed_base_url": args.embed_base_url,
        "embed_model": args.embed_model,
        "chunk_sec": args.chunk_sec,
    })
    print("index ok:", idx.get("rag", {}).get("faiss", {}).get("items_count"))

    print("3) ask...")
    ans = post(f"{args.server}/ask", {
        **src,
        "question": args.question,
        "top_k": args.top_k,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "embed_base_url": args.embed_base_url,
        "embed_model": args.embed_model,
    })
    print("answer:")
    print(json.dumps(ans.get("result"), ensure_ascii=False, indent=2))

    print("4) highlights...")
    hl = post(f"{args.server}/highlights", {
        **src,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "max_clips": args.max_clips,
        "also_make_reel": True
    })
    print("highlights ok. clips:", len(hl.get("highlights", [])))
    if hl.get("artifacts", {}).get("reel"):
        print("reel:", hl["artifacts"]["reel"]["reel_path"])

    print("Done.")

if __name__ == "__main__":
    main()