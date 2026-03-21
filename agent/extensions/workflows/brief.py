# agent/workflows/brief.py
import os
from agent.skills.video_probe import probe_video
from agent.skills.frame_sampler import sample_frames
from agent.skills.vision_caption import caption_frames, supports_video, caption_video_as_frameset
from agent.skills.timeline_builder import build_timeline
from agent.skills.persist import save_analysis
from agent.skills.web_search import deep_search_enhance
from agent.schemas import FrameStrategy

def wf_brief(asset, llm_base_url: str, llm_model: str, max_frames: int = 128,
             direct_model: bool = False, model_path: str = None, tokenizer_path: str = None,
             include_web_search: bool = False, google_api_key: str = None, google_search_engine_id: str = None) -> dict:
    meta = probe_video(asset.local_path)
    asset.metadata = meta

    if supports_video(llm_model):
        frames = caption_video_as_frameset(asset.local_path, llm_model, llm_base_url,
                                           direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)
    else:
        frames = sample_frames(
            asset, os.path.join(asset.cache_dir, "frames"),
            FrameStrategy(type="scene", params={"scene_threshold": 0.3, "max_frames": max_frames})
        )
        frames = caption_frames(frames, llm_model, llm_base_url, batch_size=8,
                                direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    # 无 ASR：传空 transcript
    class _T: segments = []
    timeline = build_timeline(meta, _T(), frames, llm_model, llm_base_url,
                              direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    # Optional web search enhancement
    web_search_results = {}
    if include_web_search:
        search_query = f"video analysis {timeline[:200]}"
        try:
            web_search_results = deep_search_enhance(search_query, timeline[:500], 
                                                   api_key=google_api_key, 
                                                   search_engine_id=google_search_engine_id)
        except Exception as e:
            web_search_results = {"error": str(e)}

    out = {
        "video": {"id": asset.id, "source": asset.source.model_dump(), "local_path": asset.local_path, **meta.model_dump()},
        "frames": frames.model_dump(),
        "asr": None,
        "timeline": timeline,
        "highlights": [],
        "rag": {},
        "web_search": web_search_results
    }
    save_analysis(asset.cache_dir, out)
    return out