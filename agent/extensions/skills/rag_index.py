# agent/skills/rag_index.py
import os, math, json
from agent.extensions.utils.cache import ensure_dir, write_json

def build_index(video, transcript, frames, index_dir: str, chunk_sec: int = 20) -> dict:
    ensure_dir(index_dir)
    items = []

    # 简单按时间分桶
    duration = video.metadata.duration_sec if video.metadata else 0
    n = int(math.ceil(duration / chunk_sec)) if duration > 0 else 0

    # 预组织 frame/caption
    frame_by_bucket = {}
    for f in frames.items:
        b = int(f.ts // chunk_sec)
        frame_by_bucket.setdefault(b, []).append(f)

    seg_by_bucket = {}
    for s in transcript.segments:
        b = int(s.start // chunk_sec)
        seg_by_bucket.setdefault(b, []).append(s)

    for b in range(n):
        start = b * chunk_sec
        end = min((b + 1) * chunk_sec, duration)
        fs = frame_by_bucket.get(b, [])
        ss = seg_by_bucket.get(b, [])
        text = []
        text.append(f"[META] source={video.source.type} uri={video.source.uri}")
        for f in fs:
            if f.caption:
                text.append(f"[FRAME {f.id} @{f.ts:.1f}s] {f.caption}")
        for s in ss:
            text.append(f"[ASR {s.id} {s.start:.1f}-{s.end:.1f}] {s.text}")

        items.append({
            "chunk_id": f"chunk_{b:06d}",
            "start": start, "end": end,
            "text": "\n".join(text),
            "frame_ids": [f.id for f in fs],
            "asr_segment_ids": [s.id for s in ss],
        })

    jsonl_path = os.path.join(index_dir, "chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    info = {"index_dir": index_dir, "chunks_path": jsonl_path, "items_count": len(items)}
    write_json(os.path.join(index_dir, "index_meta.json"), info)
    return info