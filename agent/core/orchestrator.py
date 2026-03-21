# agent/core/orchestrator.py
import logging
from agent.extensions.workflows.analyze import wf_analyze
from agent.extensions.workflows.index import wf_index
from agent.extensions.workflows.ask import wf_ask
from agent.extensions.workflows.highlights import wf_highlights
from agent.extensions.workflows.report import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(asset, mode: str, cfg: dict) -> dict:
    try:
        logger.info(f"Starting mode: {mode}")
        if mode in ("brief", "detailed"):
            return wf_analyze(asset, mode,
                              llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                              max_frames=cfg.get("max_frames", 128),
                              whisper_model=cfg.get("whisper_model", "small"),
                              direct_model=cfg.get("direct_model", False),
                              model_path=cfg.get("model_path"),
                              tokenizer_path=cfg.get("tokenizer_path"),
                              include_web_search=cfg.get("include_web_search", False),
                              google_api_key=cfg.get("google_api_key"),
                              google_search_engine_id=cfg.get("google_search_engine_id"))
        elif mode == "index":
            return wf_index(asset,
                            llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                            embed_base_url=cfg["embed_base_url"], embed_model=cfg["embed_model"],
                            chunk_sec=cfg.get("chunk_sec", 20),
                            direct_model=cfg.get("direct_model", False),
                            model_path=cfg.get("model_path"),
                            tokenizer_path=cfg.get("tokenizer_path"))
        elif mode == "ask":
            return wf_ask(asset, cfg["question"],
                          llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                          embed_base_url=cfg["embed_base_url"], embed_model=cfg["embed_model"],
                          top_k=cfg.get("top_k", 5),
                          direct_model=cfg.get("direct_model", False),
                          model_path=cfg.get("model_path"),
                          tokenizer_path=cfg.get("tokenizer_path"))
        elif mode == "highlights":
            return wf_highlights(asset,
                                 llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                                 max_clips=cfg.get("max_clips", 5),
                                 also_make_reel=cfg.get("also_make_reel", True),
                                 direct_model=cfg.get("direct_model", False),
                                 model_path=cfg.get("model_path"),
                                 tokenizer_path=cfg.get("tokenizer_path"))
        elif mode == "report":
            return generate_report(asset,
                                   analysis_type=cfg.get("analysis_type", "brief"),
                                   include_web_search=cfg.get("include_web_search", True),
                                   llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                                   direct_model=cfg.get("direct_model", False),
                                   model_path=cfg.get("model_path"),
                                   tokenizer_path=cfg.get("tokenizer_path"),
                                   google_api_key=cfg.get("google_api_key"),
                                   google_search_engine_id=cfg.get("google_search_engine_id"))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        logger.error(f"Error in mode {mode}: {str(e)}")
        raise