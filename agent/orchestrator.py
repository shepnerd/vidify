# agent/orchestrator.py
from agent.workflows.analyze import wf_analyze
from agent.workflows.index import wf_index
from agent.workflows.ask import wf_ask
from agent.workflows.highlights import wf_highlights

def run(asset, mode: str, cfg: dict) -> dict:
    if mode in ("brief", "detailed"):
        return wf_analyze(asset, mode,
                          llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                          max_frames=cfg.get("max_frames", 128),
                          whisper_model=cfg.get("whisper_model", "small"))
    if mode == "index":
        return wf_index(asset,
                        llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                        embed_base_url=cfg["embed_base_url"], embed_model=cfg["embed_model"],
                        chunk_sec=cfg.get("chunk_sec", 20))
    if mode == "ask":
        return wf_ask(asset, cfg["question"],
                      llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                      embed_base_url=cfg["embed_base_url"], embed_model=cfg["embed_model"],
                      top_k=cfg.get("top_k", 5))
    if mode == "highlights":
        return wf_highlights(asset,
                             llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                             max_clips=cfg.get("max_clips", 5),
                             also_make_reel=cfg.get("also_make_reel", True))
    raise ValueError(f"Unknown mode: {mode}")