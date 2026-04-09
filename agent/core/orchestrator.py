# agent/core/orchestrator.py
import logging
from agent.extensions.workflows.analyze import wf_analyze
from agent.extensions.workflows.index import wf_index
from agent.extensions.workflows.ask import wf_ask
from agent.extensions.workflows.highlights import wf_highlights
from agent.extensions.workflows.report import generate_report
from agent.extensions.workflows.live import wf_live
from agent.core.hooks import get_hook_manager
from agent.core.events import event_bus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(asset, mode: str, cfg: dict) -> dict:
    hooks = get_hook_manager()
    hook_env = {
        "VIDEO_URI": getattr(asset, "id", str(getattr(asset.source, "uri", ""))),
        "CACHE_DIR": getattr(asset, "cache_dir", ""),
        "MODE": mode,
    }

    try:
        logger.info(f"Starting mode: {mode}")
        event_bus.emit_progress(f"Starting {mode} workflow", 0)
        hooks.trigger("pre_analysis", hook_env)

        if mode in ("brief", "detailed"):
            result = wf_analyze(asset, mode,
                              llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                              max_frames=cfg.get("max_frames", 128),
                              whisper_model=cfg.get("whisper_model", "small"),
                              direct_model=cfg.get("direct_model", False),
                              model_path=cfg.get("model_path"),
                              tokenizer_path=cfg.get("tokenizer_path"),
                              include_web_search=cfg.get("include_web_search", False),
                              google_api_key=cfg.get("google_api_key"),
                              google_search_engine_id=cfg.get("google_search_engine_id"),
                              frame_strategy=cfg.get("frame_strategy"),
                              frame_fps=cfg.get("frame_fps"),
                              force_visual=cfg.get("force_visual"))
        elif mode == "index":
            result = wf_index(asset,
                            llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                            embed_base_url=cfg["embed_base_url"], embed_model=cfg["embed_model"],
                            chunk_sec=cfg.get("chunk_sec", 20),
                            direct_model=cfg.get("direct_model", False),
                            model_path=cfg.get("model_path"),
                            tokenizer_path=cfg.get("tokenizer_path"))
        elif mode == "ask":
            result = wf_ask(asset, cfg["question"],
                          llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                          embed_base_url=cfg["embed_base_url"], embed_model=cfg["embed_model"],
                          top_k=cfg.get("top_k", 5),
                          direct_model=cfg.get("direct_model", False),
                          model_path=cfg.get("model_path"),
                          tokenizer_path=cfg.get("tokenizer_path"))
        elif mode == "highlights":
            result = wf_highlights(asset,
                                 llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                                 max_clips=cfg.get("max_clips", 5),
                                 also_make_reel=cfg.get("also_make_reel", True),
                                 direct_model=cfg.get("direct_model", False),
                                 model_path=cfg.get("model_path"),
                                 tokenizer_path=cfg.get("tokenizer_path"))
        elif mode == "report":
            result = generate_report(asset,
                                   analysis_type=cfg.get("analysis_type", "brief"),
                                   include_web_search=cfg.get("include_web_search", True),
                                   llm_base_url=cfg["llm_base_url"], llm_model=cfg["llm_model"],
                                   direct_model=cfg.get("direct_model", False),
                                   model_path=cfg.get("model_path"),
                                   tokenizer_path=cfg.get("tokenizer_path"),
                                   google_api_key=cfg.get("google_api_key"),
                                   google_search_engine_id=cfg.get("google_search_engine_id"))
        elif mode == "live":
            result = wf_live(
                source=cfg.get("stream_source", "webcam"),
                stream_url=cfg.get("stream_url"),
                cfg=cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Post-analysis hooks
        import os
        result_path = os.path.join(getattr(asset, "cache_dir", ""), "analysis.json")
        hooks.trigger("post_analysis", {**hook_env, "RESULT_PATH": result_path})
        if mode == "highlights":
            hooks.trigger("post_highlight", hook_env)
        elif mode == "index":
            hooks.trigger("post_index", hook_env)

        event_bus.emit_progress(f"{mode} workflow complete", 100)
        return result

    except Exception as e:
        logger.error(f"Error in mode {mode}: {str(e)}")
        hooks.trigger("on_error", {**hook_env, "ERROR_MSG": str(e)})
        raise