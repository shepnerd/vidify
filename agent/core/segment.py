# agent/core/segment.py
"""Video segmentation and result merging for parallel processing.

Splits long videos into temporal segments that can be processed independently
by parallel workers, then merges per-segment results back into a unified output.
Inspired by AgentScope's fanout_pipeline pattern.

## Segmentation Interface

The segmentation logic is abstracted behind `BaseSegmentor` so that different
strategies can be swapped in:

- `DurationSegmentor` (default) — fixed-duration splits using FFmpeg time ranges.
- Future: `SceneSegmentor` — deep-learning-based temporal segmentation
  (e.g., TransNetV2, PySceneDetect with content-aware thresholds).
- Future: `SemanticSegmentor` — LLM/CLIP-driven semantic boundary detection.

To implement a custom segmentor, subclass `BaseSegmentor` and override
`segment()`. Register it via `get_segmentor(name)`.
"""
import os
import logging
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from agent.core.schemas import FrameSet, FrameItem, FrameStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class VideoSegment(BaseModel):
    """A temporal slice of a video for parallel processing."""
    index: int
    start_sec: float
    end_sec: float
    cache_dir: str  # per-segment sub-cache directory


# ---------------------------------------------------------------------------
# Segmentor interface
# ---------------------------------------------------------------------------

class BaseSegmentor(ABC):
    """Abstract base class for video segmentation strategies.

    Subclass this to implement custom temporal segmentation (e.g., DL-based
    scene boundary detection). The only required method is `segment()`.

    Example usage::

        segmentor = get_segmentor("duration", segment_duration=300)
        segments = segmentor.segment(video_path, duration_sec, base_cache_dir)
    """

    def __init__(self, min_segment_duration: float = 30.0, **kwargs):
        self.min_segment_duration = min_segment_duration

    @abstractmethod
    def segment(
        self,
        video_path: str,
        duration_sec: float,
        base_cache_dir: str,
    ) -> List[VideoSegment]:
        """Split a video into temporal segments.

        Args:
            video_path: Path to the video file (available for DL-based
                segmentors that need to analyze visual content).
            duration_sec: Total video duration in seconds.
            base_cache_dir: Root cache dir; segments go under segments/.

        Returns:
            Ordered list of non-overlapping VideoSegment objects.
        """
        ...

    def _make_segment(self, index: int, start: float, end: float,
                      base_cache_dir: str) -> VideoSegment:
        """Helper to create a VideoSegment with its cache directory."""
        segments_dir = os.path.join(base_cache_dir, "segments")
        seg_cache = os.path.join(segments_dir, f"seg_{index:03d}")
        os.makedirs(seg_cache, exist_ok=True)
        return VideoSegment(
            index=index, start_sec=start, end_sec=end, cache_dir=seg_cache,
        )

    def _merge_tiny_tail(self, segments: List[VideoSegment],
                         end_sec: float) -> List[VideoSegment]:
        """Merge the last segment into the previous one if it's too short."""
        if (len(segments) >= 2
                and (segments[-1].end_sec - segments[-1].start_sec) < self.min_segment_duration):
            segments[-2] = VideoSegment(
                index=segments[-2].index,
                start_sec=segments[-2].start_sec,
                end_sec=end_sec,
                cache_dir=segments[-2].cache_dir,
            )
            segments.pop()
        return segments


class DurationSegmentor(BaseSegmentor):
    """Split video into fixed-duration segments using FFmpeg time ranges.

    This is the default strategy — fast, no model required. Each segment is
    a time range processed independently via FFmpeg ``-ss``/``-to`` flags.
    """

    def __init__(self, segment_duration: float = 300.0, **kwargs):
        super().__init__(**kwargs)
        self.segment_duration = segment_duration

    def segment(self, video_path: str, duration_sec: float,
                base_cache_dir: str) -> List[VideoSegment]:
        if duration_sec <= 0:
            return []

        n_segments = max(1, math.ceil(duration_sec / self.segment_duration))
        segments = []

        for i in range(n_segments):
            start = i * self.segment_duration
            end = min((i + 1) * self.segment_duration, duration_sec)
            segments.append(self._make_segment(i, start, end, base_cache_dir))

        segments = self._merge_tiny_tail(segments, duration_sec)

        logger.info(
            "DurationSegmentor: split %.1fs video into %d segments (%.0fs each)",
            duration_sec, len(segments), self.segment_duration,
        )
        return segments


# ---------------------------------------------------------------------------
# Segmentor registry
# ---------------------------------------------------------------------------

_SEGMENTOR_REGISTRY: Dict[str, type] = {
    "duration": DurationSegmentor,
}


def register_segmentor(name: str, cls: type) -> None:
    """Register a custom segmentor class.

    Args:
        name: Short name used in config (e.g., "scene", "semantic").
        cls: Subclass of BaseSegmentor.
    """
    if not issubclass(cls, BaseSegmentor):
        raise TypeError(f"{cls} must be a subclass of BaseSegmentor")
    _SEGMENTOR_REGISTRY[name] = cls
    logger.info("Registered segmentor: %s -> %s", name, cls.__name__)


def get_segmentor(name: str = "duration", **kwargs) -> BaseSegmentor:
    """Get a segmentor instance by name.

    Args:
        name: Segmentor name (default "duration"). Available: "duration".
            Future: "scene", "semantic".
        **kwargs: Passed to the segmentor constructor.

    Returns:
        Configured BaseSegmentor instance.

    Raises:
        ValueError: If the segmentor name is not registered.
    """
    cls = _SEGMENTOR_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_SEGMENTOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown segmentor '{name}'. Available: {available}. "
            f"Use register_segmentor() to add custom implementations."
        )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Convenience wrapper (backward-compatible)
# ---------------------------------------------------------------------------

def split_video_into_segments(
    duration_sec: float,
    base_cache_dir: str,
    segment_duration: float = 300.0,
    min_segment_duration: float = 30.0,
    video_path: str = None,
    segmentor_name: str = "duration",
    **segmentor_kwargs,
) -> List[VideoSegment]:
    """Split a video into temporal segments for parallel processing.

    This is a convenience wrapper around the segmentor interface. For advanced
    usage, instantiate a segmentor directly via ``get_segmentor()``.

    Args:
        duration_sec: Total video duration in seconds.
        base_cache_dir: Root cache dir for the video.
        segment_duration: Target duration per segment (seconds).
        min_segment_duration: Minimum duration for tail segment.
        video_path: Path to video file (needed by DL-based segmentors).
        segmentor_name: Which segmentor to use (default "duration").
        **segmentor_kwargs: Extra kwargs passed to the segmentor.

    Returns:
        List of VideoSegment objects. Returns a single segment for short videos.
    """
    segmentor = get_segmentor(
        segmentor_name,
        segment_duration=segment_duration,
        min_segment_duration=min_segment_duration,
        **segmentor_kwargs,
    )
    return segmentor.segment(
        video_path=video_path or "",
        duration_sec=duration_sec,
        base_cache_dir=base_cache_dir,
    )


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------

def merge_framesets(
    segment_frames: List[FrameSet],
    segments: List[VideoSegment],
    strategy: Optional[FrameStrategy] = None,
) -> FrameSet:
    """Merge per-segment FrameSets into a single FrameSet.

    Frame timestamps are already global (frame_sampler adds start_sec offset),
    so we just concatenate and re-index.
    """
    all_items: List[FrameItem] = []
    for seg, fs in zip(segments, segment_frames):
        all_items.extend(fs.items)

    # Sort by timestamp and re-index
    all_items.sort(key=lambda f: f.ts)
    for idx, item in enumerate(all_items):
        item.id = f"f_{idx:06d}"

    merged_strategy = strategy or (
        segment_frames[0].strategy if segment_frames else FrameStrategy(type="scene", params={})
    )
    return FrameSet(items=all_items, strategy=merged_strategy)


def merge_ocr_results(segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-segment OCR results.

    OCR results are typically {frame_path: [detected_texts]} or similar.
    We just union-merge the dicts.
    """
    merged: Dict[str, Any] = {}
    for result in segment_results:
        if isinstance(result, dict):
            merged.update(result)
    return merged


def merge_object_results(segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-segment object detection results."""
    merged: Dict[str, Any] = {}
    for result in segment_results:
        if isinstance(result, dict):
            merged.update(result)
    return merged


def merge_emotion_results(
    segment_results: List[Dict[str, Any]],
    segments: List[VideoSegment],
) -> Dict[str, Any]:
    """Merge per-segment emotion results.

    Emotion results may contain both audio-level and frame-level data.
    Audio emotions need timestamp adjustment; frame emotions key by path.
    """
    merged: Dict[str, Any] = {}

    for seg, result in zip(segments, segment_results):
        if not isinstance(result, dict):
            continue

        # Frame-level emotions (keyed by frame path) — just merge
        for key in ("frame_emotions", "visual_emotions"):
            if key in result:
                merged.setdefault(key, {}).update(result[key])

        # Audio-level emotions — adjust timestamps
        if "audio_emotions" in result:
            adjusted = []
            for entry in result["audio_emotions"]:
                if isinstance(entry, dict):
                    entry = dict(entry)
                    if "start" in entry:
                        entry["start"] += seg.start_sec
                    if "end" in entry:
                        entry["end"] += seg.start_sec
                    adjusted.append(entry)
            merged.setdefault("audio_emotions", []).extend(adjusted)

        # Top-level keys that aren't segment-specific
        for key, val in result.items():
            if key not in ("frame_emotions", "visual_emotions", "audio_emotions"):
                merged.setdefault(key, val)

    return merged


def merge_segment_results(
    segment_outputs: List[Dict[str, Any]],
    segments: List[VideoSegment],
) -> Dict[str, Any]:
    """Merge all per-segment outputs into unified result dicts.

    Args:
        segment_outputs: List of dicts from process_segment(), each containing
            'frames', 'ocr', 'objects', 'emotions' keys.
        segments: Corresponding VideoSegment list.

    Returns:
        Dict with merged 'frames', 'ocr', 'objects', 'emotions'.
    """
    seg_frames = []
    for out in segment_outputs:
        frames = out.get("frames")
        if isinstance(frames, dict) and "items" in frames and "strategy" in frames:
            seg_frames.append(FrameSet(**frames))
        elif isinstance(frames, FrameSet):
            seg_frames.append(frames)
        else:
            seg_frames.append(FrameSet(items=[], strategy=FrameStrategy(type="scene", params={})))
    seg_ocr = [out.get("ocr", {}) for out in segment_outputs]
    seg_objects = [out.get("objects", {}) for out in segment_outputs]
    seg_emotions = [out.get("emotions", {}) for out in segment_outputs]

    return {
        "frames": merge_framesets(seg_frames, segments),
        "ocr": merge_ocr_results(seg_ocr),
        "objects": merge_object_results(seg_objects),
        "emotions": merge_emotion_results(seg_emotions, segments),
    }
