# agent/workflows/brief.py
import os
from agent.skills.video_probe import probe_video
from agent.skills.frame_sampler import sample_frames
from agent.skills.vision_caption import caption_frames
from agent.skills.timeline_builder import build_timeline
from agent.skills.persist import save_analysis
from agent.schemas import FrameStrategy

def wf_brief(asset, llm_base_url: str, llm_model: str, max_frames: int = 128,
             direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> dict:
    meta = probe_video(asset.local_path)
    asset.metadata = meta

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

    out = {
        "video": {"id": asset.id, "source": asset.source.model_dump(), "local_path": asset.local_path, **meta.model_dump()},
        "frames": frames.model_dump(),
        "asr": None,
        "timeline": timeline,
        "highlights": [],
        "rag": {}
    }
    save_analysis(asset.cache_dir, out)
    return out