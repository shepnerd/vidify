# agent/core/segment_worker.py
"""Per-segment video processing worker.

Processes a single VideoSegment through the visual pipeline:
frame sampling → (optional) MLLM captioning → OCR / detection / emotion.

Designed to be called from run_segments_parallel() for concurrent execution.
"""
import os
import logging
from typing import Dict, Any, Optional

from agent.core.schemas import FrameStrategy, FrameSet
from agent.core.segment import VideoSegment
from agent.core.parallel import run_skills_parallel
from agent.core.skill_guard import skill_guard
from agent.extensions.skills.frame_sampler import sample_frames
from agent.extensions.skills.vision_caption import caption_frames, supports_video
from agent.extensions.skills.ocr import extract_text_from_video_frames
from agent.extensions.skills.emotion_analysis import analyze_emotions

try:
    from agent.extensions.skills.object_detection import detect_objects_in_video_frames
except ImportError:
    detect_objects_in_video_frames = None

logger = logging.getLogger(__name__)


def process_segment(
    segment: VideoSegment,
    asset,
    strategy: FrameStrategy,
    need_captioning: bool = False,
    llm_model: str = None,
    llm_base_url: str = None,
    direct_model: bool = False,
    model_path: str = None,
    tokenizer_path: str = None,
    audio_path: Optional[str] = None,
    max_parallel_skills: int = 3,
) -> Dict[str, Any]:
    """Process a single video segment through the visual pipeline.

    Args:
        segment: VideoSegment with time range and cache directory.
        asset: VideoAsset (shared, read-only).
        strategy: FrameStrategy for frame sampling.
        need_captioning: Whether to run MLLM captioning on frames.
        llm_model: LLM model name for captioning.
        llm_base_url: vLLM server URL.
        direct_model: Use direct model loading instead of server.
        model_path: Path for direct model loading.
        tokenizer_path: Path for direct tokenizer loading.
        audio_path: Path to extracted audio (for emotion analysis).
        max_parallel_skills: Max concurrent skill threads.

    Returns:
        Dict with 'frames', 'ocr', 'objects', 'emotions' keys.
    """
    seg_label = f"seg_{segment.index:03d} [{segment.start_sec:.0f}s-{segment.end_sec:.0f}s]"
    logger.info("[segment_worker] Processing %s", seg_label)

    # --- Step 1: Frame sampling for this segment's time range ---
    frames_dir = os.path.join(segment.cache_dir, "frames")
    frames = sample_frames(
        asset, frames_dir, strategy,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
    )
    logger.info("[segment_worker] %s: sampled %d frames", seg_label, len(frames.items))

    # --- Step 2: Optional MLLM captioning ---
    if need_captioning and frames.items:
        logger.info("[segment_worker] %s: running MLLM captioning on %d frames", seg_label, len(frames.items))
        frames = caption_frames(
            frames, llm_model, llm_base_url, batch_size=8,
            direct_model=direct_model, model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

    # --- Step 3: Parallel advanced analysis (OCR, detection, emotion) ---
    frame_paths = [f.path for f in frames.items]

    _safe_ocr = skill_guard("OCR", optional=True, default={})(
        extract_text_from_video_frames
    )
    _safe_emotion = skill_guard("Emotion Analysis", optional=True, default={})(
        analyze_emotions
    )

    parallel_skills = []
    if frame_paths:
        parallel_skills.append(("ocr", _safe_ocr, (frame_paths,), {}))
    if detect_objects_in_video_frames and frame_paths:
        _safe_detect = skill_guard("Object Detection", optional=True, default={})(
            detect_objects_in_video_frames
        )
        parallel_skills.append(("objects", _safe_detect, (frame_paths,), {}))
    if audio_path and frame_paths:
        parallel_skills.append(("emotions", _safe_emotion, (audio_path, frame_paths), {}))

    parallel_results = run_skills_parallel(parallel_skills, max_workers=max_parallel_skills)

    result = {
        "frames": frames.model_dump(),
        "ocr": parallel_results.get("ocr", {}),
        "objects": parallel_results.get("objects", {}),
        "emotions": parallel_results.get("emotions", {}),
    }
    logger.info("[segment_worker] %s: done", seg_label)
    return result
