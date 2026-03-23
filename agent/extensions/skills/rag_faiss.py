# agent/skills/rag_faiss.py
import os, json, math
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from openai import OpenAI
from agent.extensions.utils.cache import ensure_dir, write_json

def _chunk_items(video, transcript, frames, chunk_sec: int = 20) -> List[Dict[str, Any]]:
    duration = video.metadata.duration_sec if video.metadata else 0.0
    n = int(math.ceil(duration / chunk_sec)) if duration > 0 else 0

    frame_by_bucket: Dict[int, list] = {}
    for f in frames.items:
        b = int(f.ts // chunk_sec)
        frame_by_bucket.setdefault(b, []).append(f)

    seg_by_bucket: Dict[int, list] = {}
    for s in transcript.segments:
        b = int(s.start // chunk_sec)
        seg_by_bucket.setdefault(b, []).append(s)

    items = []
    for b in range(n):
        start = b * chunk_sec
        end = min((b + 1) * chunk_sec, duration)
        fs = frame_by_bucket.get(b, [])
        ss = seg_by_bucket.get(b, [])

        parts = [f"[META] source={video.source.type} uri={video.source.uri}"]
        for f in fs:
            if f.caption:
                parts.append(f"[FRAME {f.id} @{f.ts:.1f}s] {f.caption}")
        for s in ss:
            parts.append(f"[ASR {s.id} {s.start:.1f}-{s.end:.1f}] {s.text}")

        items.append({
            "chunk_id": f"chunk_{b:06d}",
            "start": float(start),
            "end": float(end),
            "text": "\n".join(parts),
            "frame_ids": [f.id for f in fs],
            "asr_segment_ids": [s.id for s in ss],
        })
    return items

def _embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)  # cosine via inner product [1]
    return vecs

def build_faiss_index(
    video, transcript, frames,
    index_dir: str,
    embed_base_url: str,
    embed_model: str,
    chunk_sec: int = 20,
    batch_size: int = 64,
) -> Dict[str, Any]:
    ensure_dir(index_dir)
    items = _chunk_items(video, transcript, frames, chunk_sec=chunk_sec)
    texts = [it["text"] for it in items]
    if not texts:
        raise RuntimeError("No chunks to index.")

    client = OpenAI(base_url=embed_base_url, api_key="EMPTY")
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        all_vecs.append(_embed_texts(client, embed_model, texts[i:i+batch_size]))
    X = np.vstack(all_vecs)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine(sim) with normalized vecs [1]
    index.add(X)

    index_path = os.path.join(index_dir, "faiss.index")
    chunks_path = os.path.join(index_dir, "chunks.json")
    faiss.write_index(index, index_path)
    write_json(chunks_path, items)

    meta = {
        "index_dir": index_dir,
        "faiss_index_path": index_path,
        "chunks_path": chunks_path,
        "items_count": len(items),
        "dim": dim,
        "chunk_sec": chunk_sec,
        "embed_model": embed_model,
        "embed_base_url": embed_base_url
    }
    write_json(os.path.join(index_dir, "index_meta.json"), meta)
    return meta

def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    with open(os.path.join(index_dir, "index_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, items, meta

def search_faiss(
    index_dir: str,
    query: str,
    embed_base_url: str,
    embed_model: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    index, items, _ = load_faiss_index(index_dir)
    client = OpenAI(base_url=embed_base_url, api_key="EMPTY")

    qv = _embed_texts(client, embed_model, [query])  # (1, dim)
    D, I = index.search(qv, top_k)

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        it = items[idx]
        results.append({**it, "score": float(score)})
    return results