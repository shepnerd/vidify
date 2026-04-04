"""Live stream processing with adaptive scene segmentation and two-level memory.

Processes webcam or RTMP/HTTP streams frame-by-frame with:
- CLIP-based scene-change detection for adaptive segmentation
- SlowFast strategy: heavy model every N-th frame, light model otherwise
- Two-level memory: local (per-segment) + global (cross-segment)
- Backup-on-query for consistent live Q&A
"""
import os
import cv2
import time
import asyncio
import logging
import tempfile
import threading
from typing import Callable, Any, Tuple, Dict, List, Optional

from agent.config import load_models_config, load_workflows_config
from agent.core.schemas import StreamSegment, StreamConfig
from agent.extensions.models.thinking import strip_thinking
from agent.extensions.skills.scene_similarity import (
    compute_frame_embedding, is_scene_change,
)
from agent.extensions.skills.stream_memory import StreamMemoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame analysis: heavy / light
# ---------------------------------------------------------------------------

def analyze_frame_heavy(frame_path: str, models_config: Dict[str, Any]) -> Dict[str, Any]:
    """Full analysis: MLLM caption + object detection + OCR."""
    from agent.extensions.skills.vision_caption import caption_frame
    heavy_cfg = models_config.get("mllm", {}).get("heavy", {})
    model_name = heavy_cfg.get("model_name", "qwen-vl-7b")
    base_url = heavy_cfg.get("base_url", "http://localhost:8000/v1")

    caption = caption_frame(frame_path, model_name=model_name, base_url=base_url)

    objects = []
    try:
        from agent.extensions.skills.object_detection import detect_objects_in_frame
        objects = detect_objects_in_frame(frame_path)
    except (ImportError, Exception) as e:
        logger.debug(f"Object detection skipped: {e}")

    ocr_text = ""
    try:
        from agent.extensions.skills.ocr import extract_text_from_frame
        ocr_text = extract_text_from_frame(frame_path)
    except (ImportError, Exception) as e:
        logger.debug(f"OCR skipped: {e}")

    return {
        "caption": caption,
        "objects": objects,
        "ocr": ocr_text,
        "model": "heavy",
    }


def analyze_frame_light(frame_path: str, models_config: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight analysis: small MLLM caption + OCR only."""
    from agent.extensions.skills.vision_caption import caption_frame
    light_cfg = models_config.get("mllm", {}).get("light", {})
    model_name = light_cfg.get("model_name", "qwen-vl-1b")
    base_url = light_cfg.get("base_url", "http://localhost:8000/v1")

    caption = caption_frame(frame_path, model_name=model_name, base_url=base_url)

    ocr_text = ""
    try:
        from agent.extensions.skills.ocr import extract_text_from_frame
        ocr_text = extract_text_from_frame(frame_path)
    except (ImportError, Exception) as e:
        logger.debug(f"OCR skipped: {e}")

    return {
        "caption": caption,
        "objects": [],
        "ocr": ocr_text,
        "model": "light",
    }


# ---------------------------------------------------------------------------
# Core streaming processor
# ---------------------------------------------------------------------------

class LiveStreamProcessor:
    """Processes a live video stream with adaptive segmentation and memory."""

    def __init__(self, config: StreamConfig, models_config: Optional[Dict] = None,
                 workflows_config: Optional[Dict] = None):
        self.config = config
        self.models_config = models_config or load_models_config()
        self.workflows_config = workflows_config or load_workflows_config()
        self.memory = StreamMemoryManager()

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_count = 0
        self._segment_count = 0
        self._current_segment_frames: List[str] = []
        self._current_segment_analyses: List[Dict[str, Any]] = []
        self._current_segment_start_ts: float = 0.0
        self._prev_embedding: Optional[Any] = None
        self._lock = threading.Lock()
        self._tmp_dir = tempfile.mkdtemp(prefix="vidcopilot_live_")

    def start(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Start processing the stream. Blocks until stop() is called or stream ends."""
        if self.config.source == "webcam":
            self._cap = cv2.VideoCapture(0)
        elif self.config.source == "stream" and self.config.stream_url:
            self._cap = cv2.VideoCapture(self.config.stream_url)
        else:
            raise ValueError(f"Invalid source: {self.config.source} "
                             f"(stream_url={self.config.stream_url})")

        if not self._cap.isOpened():
            raise RuntimeError("Cannot open video source")

        self._running = True
        self._frame_count = 0
        self._segment_count = 0
        self._current_segment_frames = []
        self._current_segment_analyses = []
        self._current_segment_start_ts = 0.0
        self._prev_embedding = None
        stream_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(stream_fps / self.config.fps))

        logger.info(f"Live stream started: source={self.config.source}, "
                     f"target_fps={self.config.fps}, frame_interval={frame_interval}")

        raw_count = 0
        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    break

                raw_count += 1
                if raw_count % frame_interval != 0:
                    continue

                self._frame_count += 1
                ts = self._frame_count / self.config.fps

                # Save frame
                frame_path = os.path.join(
                    self._tmp_dir, f"live_{self._frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame)

                # Compute embedding for scene detection
                curr_embedding = compute_frame_embedding(frame_path)

                # Check for segment boundary
                frames_in_seg = len(self._current_segment_frames)
                need_new_seg = False

                if frames_in_seg >= self.config.min_segment_frames:
                    if frames_in_seg >= self.config.max_segment_frames:
                        need_new_seg = True
                    elif (self._prev_embedding is not None and
                          is_scene_change(self._prev_embedding, curr_embedding,
                                          self.config.similarity_threshold)):
                        need_new_seg = True

                if need_new_seg and self._current_segment_frames:
                    self._finalize_segment(ts)

                # Add frame to current segment
                self._current_segment_frames.append(frame_path)
                if not self._current_segment_frames[1:]:
                    self._current_segment_start_ts = ts

                # SlowFast analysis
                if self._frame_count % self.config.heavy_interval == 0:
                    analysis = analyze_frame_heavy(frame_path, self.models_config)
                else:
                    analysis = analyze_frame_light(frame_path, self.models_config)

                analysis["frame_id"] = self._frame_count
                analysis["ts"] = ts
                analysis["segment_id"] = self._segment_count
                self._current_segment_analyses.append(analysis)

                if callback:
                    callback(analysis)

                self._prev_embedding = curr_embedding

        finally:
            # Finalize last segment
            if self._current_segment_frames:
                final_ts = self._frame_count / self.config.fps
                self._finalize_segment(final_ts)
            self._cleanup()

    def _finalize_segment(self, end_ts: float) -> None:
        """Close the current segment: compute caption, embedding, store in memory."""
        with self._lock:
            seg_id = f"seg_{self._segment_count:04d}"
            self._segment_count += 1

            # Aggregate captions from frame analyses
            captions = [
                a["caption"] for a in self._current_segment_analyses
                if a.get("caption")
            ]
            segment_caption = " ".join(captions) if captions else None

            # Use the embedding of the middle frame as segment embedding
            mid_idx = len(self._current_segment_frames) // 2
            mid_path = self._current_segment_frames[mid_idx]
            seg_embedding = compute_frame_embedding(mid_path)

            segment = StreamSegment(
                segment_id=seg_id,
                start_ts=self._current_segment_start_ts,
                end_ts=end_ts,
                frame_paths=list(self._current_segment_frames),
                caption=segment_caption,
                embedding=seg_embedding.tolist(),
            )
            self.memory.add_segment(segment)

            # Reset for next segment
            self._current_segment_frames = []
            self._current_segment_analyses = []
            self._current_segment_start_ts = end_ts

    def stop(self) -> StreamMemoryManager:
        """Signal the processing loop to stop. Returns the memory manager."""
        self._running = False
        return self.memory

    def _cleanup(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def query(self, question: str, llm_client, model_name: str,
              embed_fn: Callable) -> Dict[str, Any]:
        """Query the live stream memory (backup-on-query pattern).

        Takes a snapshot of memory, retrieves relevant segments,
        and generates an answer using the reasoning LLM.
        """
        with self._lock:
            snapshot = self.memory.backup()

        # Embed the question
        query_embedding = embed_fn(question)

        # Retrieve relevant segments
        relevant = snapshot.retrieve_relevant_segments(query_embedding, threshold=0.3)

        # Build context
        context = snapshot.get_context_for_query(relevant)

        # Generate answer
        prompt = (
            f"You are analyzing a live video stream. Use the video context below "
            f"to answer the question.\n\n"
            f"Video context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=500,
            )
            answer = strip_thinking(resp.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"Query failed: {e}")
            answer = f"Error generating answer: {e}"

        return {
            "answer": answer,
            "relevant_segments": [s.model_dump() for s in relevant],
            "global_summary": snapshot.global_summary,
            "segments_searched": snapshot.segment_count,
        }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def process_live_stream(
    source: str = None,
    stream_url: str = None,
    callback: Callable[[Dict[str, Any]], None] = None,
    resolution: Tuple[int, int] = None,
    fps: int = None,
    heavy_interval: int = None,
) -> StreamMemoryManager:
    """Process a live stream. Returns the populated memory manager.

    This is the backward-compatible entry point matching the old signature.
    """
    workflows_config = load_workflows_config()
    ls_cfg = workflows_config.get("live_stream", {})

    config = StreamConfig(
        source=source or ls_cfg.get("source", "webcam"),
        stream_url=stream_url,
        fps=fps or ls_cfg.get("fps", 1),
        heavy_interval=heavy_interval or ls_cfg.get("heavy_interval", 5),
        similarity_threshold=ls_cfg.get("similarity_threshold", 0.9),
        min_segment_frames=ls_cfg.get("min_segment_frames", 3),
        max_segment_frames=ls_cfg.get("max_segment_frames", 16),
    )

    processor = LiveStreamProcessor(config)
    processor.start(callback=callback)
    return processor.memory


async def process_live_stream_async(
    source: str = None,
    stream_url: str = None,
    callback: Callable[[Dict[str, Any]], None] = None,
    resolution: Tuple[int, int] = None,
    fps: int = None,
    heavy_interval: int = None,
) -> StreamMemoryManager:
    """Async wrapper for live stream processing."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, process_live_stream, source, stream_url, callback, resolution, fps, heavy_interval
    )
