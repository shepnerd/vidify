# agent/extensions/workflows/detailed.py
"""Detailed analysis workflow with maximized parallelism.

Parallelism opportunities exploited:
1. Audio extraction + ASR  ∥  Frame sampling   (after probe, independent)
2. MLLM captioning  ∥  OCR + detection + emotion  (after frames, all need paths only)
3. Translation  ∥  Timeline builder  (independent: translation needs transcript, timeline needs transcript+frames)
"""
import os
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from agent.extensions.skills.video_probe import probe_video
from agent.extensions.skills.frame_sampler import sample_frames
from agent.extensions.skills.vision_caption import caption_frames, supports_video, caption_video_as_frameset
from agent.extensions.skills.audio_extract import extract_audio
from agent.extensions.skills.subtitle_parser import load_best_subtitle
from agent.extensions.skills.content_sufficiency import assess_sufficiency
from agent.extensions.skills.persist import save_analysis
from agent.extensions.skills.web_search import deep_search_enhance
from agent.extensions.skills.ocr import extract_text_from_video_frames
try:
    from agent.extensions.skills.timeline_builder import build_timeline
except ImportError:
    def build_timeline(*args, **kwargs):
        raise ImportError("timeline builder dependencies are not installed")

try:
    from agent.extensions.skills.asr import transcribe, has_local_whisper_model
except ImportError:
    def transcribe(*args, **kwargs):
        raise ImportError("ASR dependencies are not installed")
    def has_local_whisper_model(*args, **kwargs):
        return False

try:
    from agent.extensions.skills.emotion_analysis import analyze_emotions
except ImportError:
    def analyze_emotions(*args, **kwargs):
        raise ImportError("emotion analysis dependencies are not installed")
try:
    from agent.extensions.skills.object_detection import detect_objects_in_video_frames
except ImportError:
    detect_objects_in_video_frames = None
from agent.extensions.skills.translation import translate_asr_results
from agent.core.schemas import FrameStrategy, FrameSet, Transcript
from agent.config import load_models_config, load_workflows_config
from agent.core.skill_guard import skill_guard
from agent.core.events import event_bus, EventType
from agent.core.parallel import run_skills_parallel, run_segments_parallel
from agent.core.segment import split_video_into_segments, merge_segment_results
from agent.core.segment_worker import process_segment
from agent.extensions.models.vllm_openai_client import resolve_model_name

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel to distinguish "not provided" from explicit None


def _is_frameset_dump(value) -> bool:
    return isinstance(value, dict) and "items" in value and "strategy" in value


def _normalize_base_urls(llm_base_url) -> list[str]:
    if llm_base_url is None:
        return []
    if isinstance(llm_base_url, str):
        return [u.strip() for u in llm_base_url.split(",") if u.strip()]
    return [str(u).strip() for u in llm_base_url if str(u).strip()]


def _primary_base_url(llm_base_url):
    urls = _normalize_base_urls(llm_base_url)
    return urls[0] if urls else llm_base_url


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

def wf_detailed(asset, llm_base_url: str = None, llm_model: str = None,
                max_frames: int = None,
                whisper_model: str = _UNSET,
                direct_model: bool = None,
                model_path: str = None,
                tokenizer_path: str = None,
                include_web_search: bool = None,
                google_api_key: str = None, google_search_engine_id: str = None,
                force_visual: bool = None,
                frame_strategy: str = None, frame_fps: float = None) -> dict:
    # Load configurations
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    wf_cfg = workflows_config.get('detailed', {})

    # Use config defaults if not provided
    if llm_base_url is None:
        llm_base_url = models_config.get('mllm', {}).get('heavy', {}).get('base_url', 'http://localhost:8000/v1')
    if llm_model is None:
        llm_model = models_config.get('mllm', {}).get('heavy', {}).get('model_name', 'qwen3.5-9b')
    if max_frames is None:
        max_frames = wf_cfg.get('max_frames', 128)
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
    event_bus.emit_skill_complete("Video Probe", progress_pct=8)

    # --- Step 2: Audio/ASR ∥ Frame sampling (parallel after probe) ---
    # Audio extraction + ASR and frame sampling are independent — both only
    # need asset.local_path from probe. Run them concurrently.
    event_bus.emit_skill_start("Transcript + Frames", progress_pct=8)

    executor = ThreadPoolExecutor(max_workers=2)

    # Submit audio/transcript path
    asset._parallel_asr_cfg = _parallel_asr_cfg(wf_cfg)
    transcript_future = executor.submit(
        _extract_transcript, asset, meta, whisper_model
    )

    # Build frame strategy from CLI params or defaults
    frames_dir = os.path.join(asset.cache_dir, "frames")
    if frame_strategy == "fps" and frame_fps is not None:
        _frame_strat = FrameStrategy(type="fps", params={"fps": frame_fps, "max_frames": max_frames})
    else:
        _frame_strat = FrameStrategy(
            type="scene",
            params=_frame_strategy_params(wf_cfg, max_frames, scene_threshold=0.25),
        )

    # Submit frame sampling (always needed for OCR/detection in detailed mode)
    frames_future = executor.submit(sample_frames, asset, frames_dir, _frame_strat)

    # Collect results
    transcript, audio_path = transcript_future.result()
    frames = frames_future.result()
    executor.shutdown(wait=False)

    event_bus.emit_skill_complete("Transcript + Frames", progress_pct=28)

    # --- Step 3: Sufficiency check ---
    min_coverage = wf_cfg.get('min_coverage_ratio', 0.3)
    min_words = wf_cfg.get('min_word_count', 50)
    sufficiency = assess_sufficiency(transcript, meta,
                                     min_coverage_ratio=min_coverage,
                                     min_word_count=min_words,
                                     force_visual=force_visual)
    logger.info("Content sufficiency: %s — %s", sufficiency.is_sufficient, sufficiency.reason)

    # --- Step 4: Parallel segment path vs. sequential path ---
    seg_cfg = _parallel_seg_cfg(wf_cfg)
    use_parallel = (
        seg_cfg.get('enabled', False)
        and meta.duration_sec >= seg_cfg.get('min_video_duration', 300)
    )

    if use_parallel:
        frames, ocr_results, object_results, emotion_results = _run_parallel_segments(
            asset=asset, meta=meta, sufficiency=sufficiency,
            llm_model=llm_model, llm_base_url=llm_base_url,
            max_frames=max_frames, direct_model=direct_model,
            model_path=model_path, tokenizer_path=tokenizer_path,
            audio_path=audio_path, seg_cfg=seg_cfg, wf_cfg=wf_cfg,
        )
    else:
        frames, ocr_results, object_results, emotion_results = _run_sequential(
            asset=asset, frames=frames, sufficiency=sufficiency,
            llm_model=llm_model, llm_base_url=llm_base_url,
            direct_model=direct_model,
            model_path=model_path, tokenizer_path=tokenizer_path,
            audio_path=audio_path, wf_cfg=wf_cfg,
        )

    # --- Step 5: Translation ∥ Timeline builder (parallel — independent) ---
    # Translation needs only transcript.segments; timeline needs transcript+frames+meta.
    # Neither depends on the other's output.
    event_bus.emit_skill_start("Timeline + Translation", progress_pct=60)

    _safe_translate = skill_guard("Translation", optional=True, default=[])(
        translate_asr_results
    )
    target_lang = models_config.get('translation', {}).get('target_lang', 'zh')
    content_meta = meta.content if meta.content else asset.content_metadata

    executor2 = ThreadPoolExecutor(max_workers=2)

    translate_future = executor2.submit(
        lambda: _safe_translate(transcript.segments, target_lang=target_lang) if transcript.segments else []
    )
    timeline_future = executor2.submit(
        build_timeline, meta, transcript, frames, llm_model, _primary_base_url(llm_base_url),
        content_metadata=content_meta, direct_model=direct_model,
        model_path=model_path, tokenizer_path=tokenizer_path,
    )

    translated_asr = translate_future.result()
    timeline = timeline_future.result()
    executor2.shutdown(wait=False)

    event_bus.emit_skill_complete("Timeline + Translation", progress_pct=82)

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
        "translated_asr": translated_asr,
        "sufficiency": sufficiency.model_dump(),
        "visual_processing_skipped": sufficiency.is_sufficient,
        "ocr": ocr_results,
        "emotions": emotion_results,
        "objects": object_results,
        "timeline": timeline,
        "highlights": [],
        "rag": {},
        "web_search": web_search_results
    }
    save_analysis(asset.cache_dir, out)
    return out


def _extract_transcript(asset, meta, whisper_model):
    """Extract transcript via subtitles or ASR. Also extracts audio for emotion analysis.

    Returns (Transcript, audio_path_or_None).
    """
    transcript = None
    audio_path = None

    # Try subtitles first
    if asset.subtitle_tracks:
        logger.info("Subtitle tracks found (%d), parsing best one...", len(asset.subtitle_tracks))
        transcript = load_best_subtitle(asset.subtitle_tracks)
        if transcript:
            logger.info("Loaded subtitle transcript: %d segments, language=%s",
                        len(transcript.segments), transcript.language)

    # ASR fallback — also extract audio (needed for emotion analysis)
    if meta.has_audio:
        asr_cfg = getattr(asset, "_parallel_asr_cfg", None) or {}
        audio_path = extract_audio(asset, os.path.join(asset.cache_dir, "audio.wav"))
        if transcript is None and whisper_model:
            logger.info("No subtitles, running ASR with Whisper (%s)...", whisper_model)
            try:
                transcript = transcribe(
                    audio_path,
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

    return transcript, audio_path


def _run_sequential(asset, frames, sufficiency, llm_model, llm_base_url,
                    direct_model, model_path, tokenizer_path, audio_path, wf_cfg):
    """Sequential path with captioning ∥ advanced analysis parallelism.

    After frame sampling (already done), captioning and OCR/detection/emotion
    run concurrently — they all only need frame paths, not each other's output.
    """
    frame_paths = [f.path for f in frames.items]

    # --- Build parallel skill list: captioning + OCR + detection + emotion ---
    event_bus.emit_skill_start("Visual Processing", progress_pct=30)

    _safe_ocr = skill_guard("OCR", optional=True, default={})(
        extract_text_from_video_frames
    )
    _safe_emotion = skill_guard("Emotion Analysis", optional=True, default={})(
        analyze_emotions
    )

    parallel_skills = []

    # Captioning (if needed) runs alongside analysis
    need_captioning = not sufficiency.is_sufficient
    if need_captioning and frame_paths:
        if supports_video(llm_model):
            # Video-native model: pass through video file directly
            _safe_caption_video = skill_guard("Video Captioning", optional=True, default=None)(
                lambda: caption_video_as_frameset(asset.local_path, llm_model, _primary_base_url(llm_base_url),
                                                   direct_model=direct_model, model_path=model_path,
                                                   tokenizer_path=tokenizer_path,
                                                   source_duration_sec=asset.metadata.duration_sec)
            )
            parallel_skills.append(("captioning", _safe_caption_video, (), {}))
        else:
            _safe_caption = skill_guard("Frame Captioning", optional=True, default=None)(
                lambda: caption_frames(frames, llm_model, _primary_base_url(llm_base_url), batch_size=8,
                                       direct_model=direct_model, model_path=model_path,
                                       tokenizer_path=tokenizer_path,
                                       video_duration_sec=asset.metadata.duration_sec)
            )
            parallel_skills.append(("captioning", _safe_caption, (), {}))
    elif need_captioning:
        event_bus.emit_skill_skipped("Visual Captioning", reason="no frames", progress_pct=45)
    else:
        logger.info("Transcript sufficient, skipping MLLM captioning.")
        event_bus.emit_skill_skipped("Visual Captioning", reason="transcript sufficient", progress_pct=45)

    # Analysis skills — all only need frame paths
    if frame_paths:
        parallel_skills.append(("ocr", _safe_ocr, (frame_paths,), {}))
    if detect_objects_in_video_frames and frame_paths:
        _safe_detect = skill_guard("Object Detection", optional=True, default={})(
            detect_objects_in_video_frames
        )
        parallel_skills.append(("objects", _safe_detect, (frame_paths,), {}))
    if audio_path:
        parallel_skills.append(("emotions", _safe_emotion, (audio_path, frame_paths), {}))

    max_workers = wf_cfg.get('max_parallel_skills', 3) + (1 if need_captioning else 0)
    parallel_results = run_skills_parallel(parallel_skills, max_workers=max_workers)

    # Use captioned frames if available, else original
    captioned = parallel_results.get("captioning")
    if captioned is not None:
        if isinstance(captioned, FrameSet):
            frames = captioned
        elif _is_frameset_dump(captioned):
            frames = FrameSet(**captioned)

    ocr_results = parallel_results.get("ocr", {})
    object_results = parallel_results.get("objects", {})
    emotion_results = parallel_results.get("emotions", {})
    event_bus.emit_skill_complete("Visual Processing", progress_pct=60)

    return frames, ocr_results, object_results, emotion_results


def _run_parallel_segments(asset, meta, sufficiency, llm_model, llm_base_url,
                           max_frames, direct_model, model_path, tokenizer_path,
                           audio_path, seg_cfg, wf_cfg):
    """Parallel segment path: split video → process segments concurrently → merge.

    Each segment independently runs: frame sampling → [captioning ∥ OCR ∥ detection ∥ emotion].
    Results are merged with timestamp adjustment before timeline generation.
    """
    segment_duration = seg_cfg.get('segment_duration', 300)
    max_workers = seg_cfg.get('max_workers', 4)
    min_segment = seg_cfg.get('min_segment_duration', 30)

    segments = split_video_into_segments(
        duration_sec=meta.duration_sec,
        base_cache_dir=asset.cache_dir,
        video_path=asset.local_path,
        segment_duration=segment_duration,
        min_segment_duration=min_segment,
        segmentor_name=seg_cfg.get('segmentor_name', 'duration'),
        scene_threshold=seg_cfg.get('scene_threshold', 0.25),
    )

    if len(segments) <= 1:
        logger.info("Video too short for segment parallelism, falling back to sequential.")
        # Need to sample frames first for sequential path
        frames_dir = os.path.join(asset.cache_dir, "frames")
        frames = sample_frames(
            asset, frames_dir,
            FrameStrategy(type="scene", params=_frame_strategy_params(wf_cfg, max_frames, scene_threshold=0.25))
        )
        return _run_sequential(
            asset=asset, frames=frames, sufficiency=sufficiency,
            llm_model=llm_model, llm_base_url=_primary_base_url(llm_base_url),
            direct_model=direct_model,
            model_path=model_path, tokenizer_path=tokenizer_path,
            audio_path=audio_path, wf_cfg=wf_cfg,
        )

    event_bus.emit_skill_start(
        "Parallel Segments",
        message=f"Processing {len(segments)} segments with {max_workers} workers...",
        progress_pct=28,
    )

    # Distribute max_frames across segments proportionally
    per_seg_max_frames = max(8, max_frames // len(segments))
    strategy = FrameStrategy(type="scene", params={
        **_frame_strategy_params(wf_cfg, per_seg_max_frames),
        "scene_threshold": 0.25,
    })

    need_captioning = not sufficiency.is_sufficient

    worker_kwargs = {
        "asset": asset,
        "strategy": strategy,
        "need_captioning": need_captioning,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "direct_model": direct_model,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "audio_path": audio_path,
        "max_parallel_skills": wf_cfg.get('max_parallel_skills', 3),
    }

    segment_outputs = run_segments_parallel(
        segments=segments,
        worker_fn=process_segment,
        worker_kwargs=worker_kwargs,
        max_workers=max_workers,
    )

    # Merge all segment results
    merged = merge_segment_results(segment_outputs, segments)

    frames = merged["frames"]
    ocr_results = merged["ocr"]
    object_results = merged["objects"]
    emotion_results = merged["emotions"]

    logger.info(
        "Parallel segments done: %d total frames, %d OCR entries, %d object entries",
        len(frames.items), len(ocr_results), len(object_results),
    )
    event_bus.emit_skill_complete("Parallel Segments", progress_pct=60)

    return frames, ocr_results, object_results, emotion_results
