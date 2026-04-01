# agent/extensions/workflows/brief.py
import os
import logging
from agent.extensions.skills.video_probe import probe_video
from agent.extensions.skills.frame_sampler import sample_frames
from agent.extensions.skills.vision_caption import caption_frames, supports_video, caption_video_as_frameset
from agent.extensions.skills.audio_extract import extract_audio
from agent.extensions.skills.asr import transcribe
from agent.extensions.skills.subtitle_parser import load_best_subtitle
from agent.extensions.skills.content_sufficiency import assess_sufficiency
from agent.extensions.skills.timeline_builder import build_timeline
from agent.extensions.skills.persist import save_analysis
from agent.extensions.skills.web_search import deep_search_enhance
from agent.core.schemas import FrameStrategy, FrameSet, Transcript
from agent.config import load_models_config, load_workflows_config
from agent.core.skill_guard import skill_guard
from agent.core.events import event_bus, EventType

logger = logging.getLogger(__name__)

def wf_brief(asset, llm_base_url: str = None, llm_model: str = None, max_frames: int = None,
             direct_model: bool = None, model_path: str = None, tokenizer_path: str = None,
             include_web_search: bool = None, google_api_key: str = None, google_search_engine_id: str = None,
             whisper_model: str = None, force_visual: bool = None) -> dict:
    # Load configurations
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    wf_cfg = workflows_config.get('brief', {})

    # Use config defaults if not provided
    if llm_base_url is None:
        llm_base_url = models_config.get('mllm', {}).get('heavy', {}).get('base_url', 'http://localhost:8000/v1')
    if llm_model is None:
        llm_model = models_config.get('mllm', {}).get('heavy', {}).get('model_name', 'qwen-vl-7b')
    if max_frames is None:
        max_frames = wf_cfg.get('max_frames', 64)
    if whisper_model is None:
        whisper_model = models_config.get('asr', {}).get('size', 'small')
    if direct_model is None:
        direct_model = False
    if include_web_search is None:
        include_web_search = wf_cfg.get('include_web_search', False)
    if force_visual is None:
        force_visual = wf_cfg.get('force_visual', False)

    # --- Step 1: Probe video technical metadata ---
    event_bus.emit_skill_start("Video Probe", progress_pct=5)
    meta = probe_video(asset.local_path)
    if asset.content_metadata:
        meta.content = asset.content_metadata
    asset.metadata = meta
    event_bus.emit_skill_complete("Video Probe", progress_pct=10)

    # --- Step 2: Get transcript (subtitles first, then ASR fallback) ---
    event_bus.emit_skill_start("Transcript", progress_pct=10)
    transcript = None

    # 2a: Try subtitles (free, often higher quality than ASR)
    if asset.subtitle_tracks:
        logger.info("Subtitle tracks found (%d), parsing best one...", len(asset.subtitle_tracks))
        transcript = load_best_subtitle(asset.subtitle_tracks)
        if transcript:
            logger.info("Loaded subtitle transcript: %d segments, language=%s",
                        len(transcript.segments), transcript.language)

    # 2b: ASR fallback if no subtitles
    if transcript is None and meta.has_audio and whisper_model:
        logger.info("No subtitles available, running ASR with Whisper (%s)...", whisper_model)
        audio = extract_audio(asset, os.path.join(asset.cache_dir, "audio.wav"))
        transcript = transcribe(audio, os.path.join(asset.cache_dir, "asr.json"), model_size=whisper_model)

    if transcript is None:
        transcript = Transcript(segments=[], language=None)
    event_bus.emit_skill_complete("Transcript", progress_pct=30)

    # --- Step 3: Sufficiency check ---
    min_coverage = wf_cfg.get('min_coverage_ratio', 0.3)
    min_words = wf_cfg.get('min_word_count', 50)
    sufficiency = assess_sufficiency(transcript, meta,
                                     min_coverage_ratio=min_coverage,
                                     min_word_count=min_words,
                                     force_visual=force_visual)
    logger.info("Content sufficiency: %s — %s", sufficiency.is_sufficient, sufficiency.reason)

    # --- Step 4: Conditional visual processing ---
    if not sufficiency.is_sufficient:
        event_bus.emit_skill_start("Visual Captioning", progress_pct=35)
        logger.info("Transcript insufficient, running visual processing...")
        if supports_video(llm_model):
            frames = caption_video_as_frameset(asset.local_path, llm_model, llm_base_url,
                                               direct_model=direct_model, model_path=model_path,
                                               tokenizer_path=tokenizer_path)
        else:
            frames = sample_frames(
                asset, os.path.join(asset.cache_dir, "frames"),
                FrameStrategy(type="scene", params={"scene_threshold": 0.3, "max_frames": max_frames})
            )
            frames = caption_frames(frames, llm_model, llm_base_url, batch_size=8,
                                    direct_model=direct_model, model_path=model_path,
                                    tokenizer_path=tokenizer_path)
    else:
        logger.info("Transcript sufficient, skipping MLLM visual processing.")
        frames = FrameSet(items=[], strategy=FrameStrategy(type="skipped", params={}))
        event_bus.emit_skill_skipped("Visual Captioning", reason="transcript sufficient", progress_pct=60)

    event_bus.emit_skill_complete("Visual Captioning", progress_pct=60)

    # --- Step 5: Build timeline ---
    event_bus.emit_skill_start("Timeline Builder", progress_pct=60)
    content_meta = meta.content if meta.content else asset.content_metadata
    timeline = build_timeline(meta, transcript, frames, llm_model, llm_base_url,
                              content_metadata=content_meta,
                              direct_model=direct_model, model_path=model_path,
                              tokenizer_path=tokenizer_path)
    event_bus.emit_skill_complete("Timeline Builder", progress_pct=80)

    # --- Step 6: Optional web search enhancement ---
    web_search_results = {}
    if include_web_search:
        _safe_search = skill_guard("Web Search", optional=True, default={})(
            deep_search_enhance
        )
        search_query = f"video analysis {timeline[:200]}" if isinstance(timeline, str) else "video analysis"
        web_search_results = _safe_search(search_query, str(timeline)[:500],
                                          api_key=google_api_key,
                                          search_engine_id=google_search_engine_id)

    # --- Step 7: Output ---
    out = {
        "video": {"id": asset.id, "source": asset.source.model_dump(), "local_path": asset.local_path, **meta.model_dump()},
        "content_metadata": content_meta.model_dump() if content_meta else None,
        "frames": frames.model_dump(),
        "asr": transcript.model_dump(),
        "sufficiency": sufficiency.model_dump(),
        "visual_processing_skipped": sufficiency.is_sufficient,
        "timeline": timeline,
        "highlights": [],
        "rag": {},
        "web_search": web_search_results
    }
    save_analysis(asset.cache_dir, out)
    return out
