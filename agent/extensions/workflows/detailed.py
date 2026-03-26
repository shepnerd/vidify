# agent/extensions/workflows/detailed.py
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
from agent.extensions.skills.ocr import extract_text_from_video_frames
from agent.extensions.skills.emotion_analysis import analyze_emotions
try:
    from agent.extensions.skills.object_detection import detect_objects_in_video_frames
except ImportError:
    detect_objects_in_video_frames = None
from agent.extensions.skills.translation import translate_asr_results
from agent.core.schemas import FrameStrategy, FrameSet, Transcript
from agent.config import load_models_config, load_workflows_config

logger = logging.getLogger(__name__)

def wf_detailed(asset, llm_base_url: str = None, llm_model: str = None,
                max_frames: int = None,
                whisper_model: str = None,
                direct_model: bool = None,
                model_path: str = None,
                tokenizer_path: str = None,
                include_web_search: bool = None,
                google_api_key: str = None, google_search_engine_id: str = None,
                force_visual: bool = None) -> dict:
    # Load configurations
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    wf_cfg = workflows_config.get('detailed', {})

    # Use config defaults if not provided
    if llm_base_url is None:
        llm_base_url = models_config.get('mllm', {}).get('heavy', {}).get('base_url', 'http://localhost:8000/v1')
    if llm_model is None:
        llm_model = models_config.get('mllm', {}).get('heavy', {}).get('model_name', 'qwen-vl-7b')
    if max_frames is None:
        max_frames = wf_cfg.get('max_frames', 128)
    if whisper_model is None:
        whisper_model = models_config.get('asr', {}).get('size', 'small')
    if direct_model is None:
        direct_model = False
    if include_web_search is None:
        include_web_search = wf_cfg.get('include_web_search', False)
    if force_visual is None:
        force_visual = wf_cfg.get('force_visual', False)

    # --- Step 1: Probe video technical metadata ---
    meta = probe_video(asset.local_path)
    if asset.content_metadata:
        meta.content = asset.content_metadata
    asset.metadata = meta

    # --- Step 2: Get transcript (subtitles first, then ASR fallback) ---
    transcript = None
    audio_path = None

    # 2a: Try subtitles
    if asset.subtitle_tracks:
        logger.info("Subtitle tracks found (%d), parsing best one...", len(asset.subtitle_tracks))
        transcript = load_best_subtitle(asset.subtitle_tracks)
        if transcript:
            logger.info("Loaded subtitle transcript: %d segments, language=%s",
                        len(transcript.segments), transcript.language)

    # 2b: ASR fallback — also extract audio (needed for emotion analysis)
    if meta.has_audio:
        audio_path = extract_audio(asset, os.path.join(asset.cache_dir, "audio.wav"))
        if transcript is None and whisper_model:
            logger.info("No subtitles, running ASR with Whisper (%s)...", whisper_model)
            transcript = transcribe(audio_path, os.path.join(asset.cache_dir, "asr.json"),
                                    model_size=whisper_model)

    if transcript is None:
        transcript = Transcript(segments=[], language=None)

    # --- Step 3: Sufficiency check ---
    min_coverage = wf_cfg.get('min_coverage_ratio', 0.3)
    min_words = wf_cfg.get('min_word_count', 50)
    sufficiency = assess_sufficiency(transcript, meta,
                                     min_coverage_ratio=min_coverage,
                                     min_word_count=min_words,
                                     force_visual=force_visual)
    logger.info("Content sufficiency: %s — %s", sufficiency.is_sufficient, sufficiency.reason)

    # --- Step 4: Frame sampling (always, for OCR/detection) + conditional MLLM captioning ---
    # In detailed mode, always sample frames for OCR and object detection (lightweight local models).
    # Only skip the expensive MLLM captioning when transcript is sufficient.
    frames_dir = os.path.join(asset.cache_dir, "frames")
    frames = sample_frames(
        asset, frames_dir,
        FrameStrategy(type="scene", params={"scene_threshold": 0.25, "max_frames": max_frames})
    )

    if not sufficiency.is_sufficient:
        logger.info("Transcript insufficient, running MLLM frame captioning...")
        if supports_video(llm_model):
            frames = caption_video_as_frameset(asset.local_path, llm_model, llm_base_url,
                                               direct_model=direct_model, model_path=model_path,
                                               tokenizer_path=tokenizer_path)
        else:
            frames = caption_frames(frames, llm_model, llm_base_url, batch_size=8,
                                    direct_model=direct_model, model_path=model_path,
                                    tokenizer_path=tokenizer_path)
    else:
        logger.info("Transcript sufficient, skipping MLLM captioning. Frames sampled for OCR/detection only.")

    # --- Step 5: Advanced analysis (OCR, object detection, emotion) ---
    # BUG FIX: was `frames.frames` — correct attribute is `frames.items`
    frame_paths = [f.path for f in frames.items]

    ocr_results = extract_text_from_video_frames(frame_paths) if frame_paths else {}
    object_results = detect_objects_in_video_frames(frame_paths) if (detect_objects_in_video_frames and frame_paths) else {}
    emotion_results = analyze_emotions(audio_path, frame_paths) if audio_path else {}

    # --- Step 6: Translation ---
    # BUG FIX: was using `transcript` before it was defined; now it's defined above
    target_lang = models_config.get('translation', {}).get('target_lang', 'zh')
    translated_asr = translate_asr_results(transcript.segments, target_lang=target_lang) if transcript.segments else []

    # --- Step 7: Build timeline ---
    content_meta = meta.content if meta.content else asset.content_metadata
    timeline = build_timeline(meta, transcript, frames, llm_model, llm_base_url,
                              content_metadata=content_meta,
                              direct_model=direct_model, model_path=model_path,
                              tokenizer_path=tokenizer_path)

    # --- Step 8: Optional web search enhancement ---
    web_search_results = {}
    if include_web_search:
        search_query = f"video analysis {timeline[:200]}" if isinstance(timeline, str) else "video analysis"
        try:
            web_search_results = deep_search_enhance(search_query, str(timeline)[:500],
                                                     api_key=google_api_key,
                                                     search_engine_id=google_search_engine_id)
        except Exception as e:
            web_search_results = {"error": str(e)}

    # --- Step 9: Output ---
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
