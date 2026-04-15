# agent/extensions/workflows/brief.py
import os
import logging
from agent.extensions.skills.video_probe import probe_video
from agent.extensions.skills.frame_sampler import sample_frames
from agent.extensions.skills.vision_caption import caption_frames, supports_video, caption_video_as_frameset
from agent.extensions.skills.audio_extract import extract_audio
from agent.extensions.skills.asr import transcribe, has_local_whisper_model
from agent.extensions.skills.subtitle_parser import load_best_subtitle
from agent.extensions.skills.content_sufficiency import assess_sufficiency
from agent.extensions.skills.timeline_builder import build_timeline
from agent.extensions.skills.persist import save_analysis
from agent.extensions.skills.web_search import deep_search_enhance
from agent.core.schemas import FrameStrategy, FrameSet, Transcript
from agent.config import load_models_config, load_workflows_config
from agent.core.skill_guard import skill_guard
from agent.core.events import event_bus, EventType
from agent.core.parallel import run_segments_parallel
from agent.core.segment import split_video_into_segments, merge_framesets
from agent.extensions.utils import split_video_segment
from agent.extensions.models.vllm_openai_client import resolve_model_name

_UNSET = object()  # sentinel to distinguish "not provided" from explicit None

logger = logging.getLogger(__name__)


def _is_frameset_dump(value) -> bool:
    return isinstance(value, dict) and "items" in value and "strategy" in value


def _normalize_base_urls(llm_base_url) -> list[str]:
    if llm_base_url is None:
        return []
    if isinstance(llm_base_url, str):
        return [u.strip() for u in llm_base_url.split(",") if u.strip()]
    return [str(u).strip() for u in llm_base_url if str(u).strip()]


def _primary_base_url(llm_base_url: str) -> str:
    urls = _normalize_base_urls(llm_base_url)
    return urls[0] if urls else llm_base_url


def _pick_base_url(llm_base_url, segment_index: int) -> str:
    urls = _normalize_base_urls(llm_base_url)
    if not urls:
        return llm_base_url
    return urls[segment_index % len(urls)]


def _parallel_seg_cfg(wf_cfg: dict) -> dict:
    seg_cfg = dict(wf_cfg.get('parallel_segments', {}))
    env_enabled = os.environ.get("VIDIFY_PARALLEL_SEGMENTS")
    if env_enabled is not None:
        seg_cfg["enabled"] = env_enabled.lower() in ("1", "true", "yes", "on")
    if os.environ.get("VIDIFY_SEGMENTOR"):
        seg_cfg["segmentor_name"] = os.environ["VIDIFY_SEGMENTOR"]
    if os.environ.get("VIDIFY_SEGMENT_DURATION"):
        seg_cfg["segment_duration"] = float(os.environ["VIDIFY_SEGMENT_DURATION"])
    if os.environ.get("VIDIFY_SCENE_THRESHOLD"):
        seg_cfg["scene_threshold"] = float(os.environ["VIDIFY_SCENE_THRESHOLD"])
    if os.environ.get("VIDIFY_PARALLEL_WORKERS"):
        seg_cfg["max_workers"] = int(os.environ["VIDIFY_PARALLEL_WORKERS"])
    if os.environ.get("VIDIFY_MIN_VIDEO_DURATION"):
        seg_cfg["min_video_duration"] = float(os.environ["VIDIFY_MIN_VIDEO_DURATION"])
    if os.environ.get("VIDIFY_MIN_SEGMENT_DURATION"):
        seg_cfg["min_segment_duration"] = float(os.environ["VIDIFY_MIN_SEGMENT_DURATION"])
    return seg_cfg


def _parallel_asr_cfg(wf_cfg: dict) -> dict:
    asr_cfg = dict(wf_cfg.get('parallel_asr', {}))
    env_enabled = os.environ.get("VIDIFY_PARALLEL_ASR")
    if env_enabled is not None:
        asr_cfg["enabled"] = env_enabled.lower() in ("1", "true", "yes", "on")
    if os.environ.get("VIDIFY_ASR_WORKERS"):
        asr_cfg["max_workers"] = int(os.environ["VIDIFY_ASR_WORKERS"])
    if os.environ.get("VIDIFY_ASR_SEGMENT_DURATION"):
        asr_cfg["segment_duration"] = float(os.environ["VIDIFY_ASR_SEGMENT_DURATION"])
    if os.environ.get("VIDIFY_ASR_MIN_AUDIO_DURATION"):
        asr_cfg["min_audio_duration"] = float(os.environ["VIDIFY_ASR_MIN_AUDIO_DURATION"])
    if os.environ.get("VIDIFY_ASR_MIN_SEGMENT_DURATION"):
        asr_cfg["min_segment_duration"] = float(os.environ["VIDIFY_ASR_MIN_SEGMENT_DURATION"])
    if os.environ.get("VIDIFY_ASR_DEVICES"):
        asr_cfg["devices"] = [
            item.strip() for item in os.environ["VIDIFY_ASR_DEVICES"].split(",") if item.strip()
        ]
    return asr_cfg


def _frame_strategy_params(wf_cfg: dict, max_frames: int, **extra) -> dict:
    params = {
        "max_frames": max_frames,
        "adaptive_by_duration": wf_cfg.get('adaptive_frame_sampling', True),
        "min_frames": wf_cfg.get('min_frames', 16),
    }
    params.update(extra)
    return params


def _brief_segment_worker(segment, asset, strategy, llm_model, llm_base_url,
                          direct_model, model_path, tokenizer_path):
    """Process a single segment for brief workflow: sample frames + caption."""
    from agent.core.segment import VideoSegment
    seg_label = f"seg_{segment.index:03d}"
    logger.info("[brief_segment] Processing %s", seg_label)

    seg_base_url = _pick_base_url(llm_base_url, segment.index)
    if supports_video(llm_model):
        seg_path = os.path.join(segment.cache_dir, "segment.mp4")
        if not (os.path.isfile(seg_path) and os.path.getsize(seg_path) > 0):
            split_video_segment(
                asset.local_path,
                segment.start_sec,
                segment.end_sec - segment.start_sec,
                seg_path,
            )
        frames = caption_video_as_frameset(
            seg_path, llm_model, seg_base_url,
            direct_model=direct_model, model_path=model_path,
            tokenizer_path=tokenizer_path,
            source_duration_sec=asset.metadata.duration_sec,
        )
        for item in frames.items:
            item.ts += segment.start_sec
    else:
        frames_dir = os.path.join(segment.cache_dir, "frames")
        frames = sample_frames(
            asset, frames_dir, strategy,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
        )
        frames = caption_frames(
            frames, llm_model, seg_base_url, batch_size=8,
            direct_model=direct_model, model_path=model_path,
            tokenizer_path=tokenizer_path,
            video_duration_sec=asset.metadata.duration_sec,
        )
    logger.info("[brief_segment] %s: done (%d frames)", seg_label, len(frames.items))
    return {"frames": frames.model_dump()}


def wf_brief(asset, llm_base_url: str = None, llm_model: str = None, max_frames: int = None,
             direct_model: bool = None, model_path: str = None, tokenizer_path: str = None,
             include_web_search: bool = None, google_api_key: str = None, google_search_engine_id: str = None,
             whisper_model: str = _UNSET, force_visual: bool = None,
             frame_strategy: str = None, frame_fps: float = None) -> dict:
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
    if whisper_model is _UNSET:
        whisper_model = models_config.get('asr', {}).get('size', 'small')
    if direct_model is None:
        direct_model = False
    if include_web_search is None:
        include_web_search = wf_cfg.get('include_web_search', False)
    if force_visual is None:
        force_visual = wf_cfg.get('force_visual', False)
    llm_base_url = _normalize_base_urls(llm_base_url)
    if not direct_model:
        llm_model = resolve_model_name(llm_model, _primary_base_url(llm_base_url))

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
        asr_cfg = _parallel_asr_cfg(wf_cfg)
        try:
            transcript = transcribe(
                audio,
                os.path.join(asset.cache_dir, "asr.json"),
                model_size=whisper_model,
                parallel=asr_cfg.get("enabled", False),
                max_workers=asr_cfg.get("max_workers"),
                devices=asr_cfg.get("devices"),
                segment_duration_sec=asr_cfg.get("segment_duration", 300),
                min_audio_duration_sec=asr_cfg.get("min_audio_duration", 300),
                min_segment_duration_sec=asr_cfg.get("min_segment_duration", 30),
            )
        except Exception as e:
            local_only = has_local_whisper_model(whisper_model)
            if local_only:
                logger.warning("ASR failed with local Whisper model %s: %s", whisper_model, e)
            else:
                logger.warning(
                    "ASR unavailable for Whisper %s; continuing transcript-first without ASR. "
                    "Place models/whisper-%s locally to enable offline ASR. Error: %s",
                    whisper_model, whisper_model, e,
                )
            transcript = None

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
    # Build frame strategy from CLI params or defaults
    def _make_strategy(mf=max_frames):
        if frame_strategy == "fps" and frame_fps is not None:
            return FrameStrategy(type="fps", params={"fps": frame_fps, "max_frames": mf})
        return FrameStrategy(type="scene", params=_frame_strategy_params(wf_cfg, mf, scene_threshold=0.3))

    if not sufficiency.is_sufficient:
        event_bus.emit_skill_start("Visual Captioning", progress_pct=35)
        logger.info("Transcript insufficient, running visual processing...")

        # Check if parallel segments should be used
        seg_cfg = _parallel_seg_cfg(wf_cfg)
        use_parallel = (
            seg_cfg.get('enabled', False)
            and meta.duration_sec >= seg_cfg.get('min_video_duration', 300)
        )

        if use_parallel:
            segments = split_video_into_segments(
                duration_sec=meta.duration_sec,
                base_cache_dir=asset.cache_dir,
                video_path=asset.local_path,
                segment_duration=seg_cfg.get('segment_duration', 180),
                min_segment_duration=seg_cfg.get('min_segment_duration', 30),
                segmentor_name=seg_cfg.get('segmentor_name', 'duration'),
                scene_threshold=seg_cfg.get('scene_threshold', 0.25),
            )
            if len(segments) > 1:
                per_seg_max = max(8, max_frames // len(segments))
                strategy = _make_strategy(per_seg_max)
                logger.info(
                    "Brief parallel: %d segments, %d workers, segmentor=%s, endpoints=%d",
                    len(segments), seg_cfg.get('max_workers', 4),
                    seg_cfg.get('segmentor_name', 'duration'),
                    len(llm_base_url) or 1,
                )

                seg_outputs = run_segments_parallel(
                    segments=segments,
                    worker_fn=_brief_segment_worker,
                    worker_kwargs={
                        "asset": asset, "strategy": strategy,
                        "llm_model": llm_model, "llm_base_url": llm_base_url,
                        "direct_model": direct_model, "model_path": model_path,
                        "tokenizer_path": tokenizer_path,
                    },
                    max_workers=seg_cfg.get('max_workers', 4),
                )
                seg_frames = [
                    FrameSet(**out["frames"]) if _is_frameset_dump(out.get("frames"))
                    else FrameSet(items=[], strategy=strategy)
                    for out in seg_outputs
                ]
                frames = merge_framesets(seg_frames, segments, strategy)
            else:
                use_parallel = False

        if use_parallel:
            pass
        elif supports_video(llm_model):
            frames = caption_video_as_frameset(asset.local_path, llm_model, _primary_base_url(llm_base_url),
                                               direct_model=direct_model, model_path=model_path,
                                               tokenizer_path=tokenizer_path,
                                               source_duration_sec=asset.metadata.duration_sec)
        else:
            # Original sequential path
            frames = sample_frames(
                asset, os.path.join(asset.cache_dir, "frames"),
                _make_strategy()
            )
            frames = caption_frames(frames, llm_model, _primary_base_url(llm_base_url), batch_size=8,
                                    direct_model=direct_model, model_path=model_path,
                                    tokenizer_path=tokenizer_path,
                                    video_duration_sec=asset.metadata.duration_sec)
    else:
        logger.info("Transcript sufficient, skipping MLLM visual processing.")
        frames = FrameSet(items=[], strategy=FrameStrategy(type="skipped", params={}))
        event_bus.emit_skill_skipped("Visual Captioning", reason="transcript sufficient", progress_pct=60)

    event_bus.emit_skill_complete("Visual Captioning", progress_pct=60)

    # --- Step 5: Build timeline ---
    event_bus.emit_skill_start("Timeline Builder", progress_pct=60)
    content_meta = meta.content if meta.content else asset.content_metadata
    timeline = build_timeline(meta, transcript, frames, llm_model, _primary_base_url(llm_base_url),
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
