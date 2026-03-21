# agent/workflows/index.py
import os
from agent.skills.persist import load_analysis, save_analysis
from agent.skills.deserialize import load_frames, load_transcript
from agent.skills.rag_faiss import build_faiss_index
from agent.extensions.workflows.detailed import wf_detailed

def wf_index(asset,
             llm_base_url: str, llm_model: str,
             embed_base_url: str, embed_model: str,
             chunk_sec: int = 20,
             direct_model: bool = False,
             model_path: str = None,
             tokenizer_path: str = None) -> dict:
    try:
        analysis = load_analysis(asset.cache_dir)
    except Exception:
        analysis = wf_detailed(asset, llm_base_url, llm_model, direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    if not analysis.get("frames") or not analysis.get("asr"):
        analysis = wf_detailed(asset, llm_base_url, llm_model, direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    frames = load_frames(analysis["frames"])
    transcript = load_transcript(analysis["asr"])

    index_dir = os.path.join(asset.cache_dir, "index_faiss")
    faiss_meta = build_faiss_index(
        asset, transcript, frames,
        index_dir=index_dir,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        chunk_sec=chunk_sec
    )

    analysis.setdefault("rag", {})
    analysis["rag"]["faiss"] = faiss_meta
    save_analysis(asset.cache_dir, analysis)
    return analysis["rag"]