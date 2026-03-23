# agent/extensions/workflows/detailed.py
import os
from agent.extensions.skills.video_probe import probe_video
from agent.extensions.skills.frame_sampler import sample_frames
from agent.extensions.skills.vision_caption import caption_frames, supports_video, caption_video_as_frameset
from agent.extensions.skills.audio_extract import extract_audio
from agent.extensions.skills.asr import transcribe
from agent.extensions.skills.timeline_builder import build_timeline
from agent.extensions.skills.persist import save_analysis
from agent.extensions.skills.web_search import deep_search_enhance
from agent.extensions.skills.ocr import extract_text_from_video_frames
from agent.extensions.skills.emotion_analysis import analyze_emotions
from agent.extensions.skills.object_detection import detect_objects_in_video_frames
from agent.extensions.skills.translation import translate_asr_results
from agent.core.schemas import FrameStrategy, Transcript
from agent.config import load_models_config, load_workflows_config

def wf_detailed(asset, llm_base_url: str = None, llm_model: str = None,
                max_frames: int = None,
                whisper_model: str = None,
                direct_model: bool = None,
                model_path: str = None,
                tokenizer_path: str = None,
                include_web_search: bool = None,
                google_api_key: str = None, google_search_engine_id: str = None) -> dict:
    # Load configurations
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    
    # Use config defaults if not provided
    if llm_base_url is None:
        llm_base_url = models_config.get('mllm', {}).get('heavy', {}).get('base_url', 'http://localhost:8000/v1')
    if llm_model is None:
        llm_model = models_config.get('mllm', {}).get('heavy', {}).get('model_name', 'qwen-vl-7b')
    if max_frames is None:
        max_frames = workflows_config.get('detailed', {}).get('max_frames', 128)
    if whisper_model is None:
        whisper_model = models_config.get('asr', {}).get('size', 'small')
    if direct_model is None:
        direct_model = False  # Keep as parameter or add to config
    if include_web_search is None:
        include_web_search = workflows_config.get('detailed', {}).get('include_web_search', False)
    meta = probe_video(asset.local_path)
    asset.metadata = meta

    if supports_video(llm_model):
        frames = caption_video_as_frameset(asset.local_path, llm_model, llm_base_url,
                                           direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)
    else:
        frames = sample_frames(
            asset, os.path.join(asset.cache_dir, "frames"),
            FrameStrategy(type="scene", params={"scene_threshold": 0.25, "max_frames": max_frames})
        )
        frames = caption_frames(frames, llm_model, llm_base_url, batch_size=8,
                                direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    # Advanced analysis: OCR, emotion, object detection
    frame_paths = [f.path for f in frames.frames]
    ocr_results = extract_text_from_video_frames(frame_paths)
    emotion_results = analyze_emotions(audio, frame_paths)
    object_results = detect_objects_in_video_frames(frame_paths)

    # Translate ASR if needed
    translated_asr = translate_asr_results(transcript.segments, target_lang='zh') if transcript.segments else []

    audio = extract_audio(asset, os.path.join(asset.cache_dir, "audio.wav"))
    if whisper_model:
        transcript = transcribe(audio, os.path.join(asset.cache_dir, "asr.json"), model_size=whisper_model)
    else:
        # Skip ASR
        transcript = Transcript(segments=[], language=None)

    timeline = build_timeline(meta, transcript, frames, llm_model, llm_base_url,
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
        "asr": transcript.model_dump(),
        "translated_asr": translated_asr,
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