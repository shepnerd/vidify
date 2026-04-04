"""Two-level streaming memory manager.

Inspired by OmniLive's memory architecture:
- Local memory: per-segment compressed representations (caption + embedding)
- Global memory: LLM-generated summary across all segments

Supports query-time retrieval via cosine similarity and backup-on-query
for consistency during live processing.
"""
import copy
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional

from agent.core.schemas import StreamSegment, StreamMemory
from agent.extensions.models.thinking import strip_thinking

logger = logging.getLogger(__name__)


class StreamMemoryManager:
    """Manages a two-level memory hierarchy for streaming video understanding."""

    def __init__(self):
        self._segments: List[StreamSegment] = []
        self._global_summary: str = ""
        self._global_embedding: Optional[np.ndarray] = None
        self._total_frames: int = 0
        self._total_duration: float = 0.0

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def global_summary(self) -> str:
        return self._global_summary

    def add_segment(self, segment: StreamSegment) -> None:
        """Append a new local memory segment."""
        self._segments.append(segment)
        self._total_frames += len(segment.frame_paths)
        if segment.end_ts > self._total_duration:
            self._total_duration = segment.end_ts
        logger.info(f"Added segment {segment.segment_id} "
                     f"({segment.start_ts:.1f}-{segment.end_ts:.1f}s, "
                     f"{len(segment.frame_paths)} frames)")

    def update_global_summary(self, llm_client, model_name: str) -> str:
        """Re-generate global summary from all local segment captions using LLM."""
        if not self._segments:
            self._global_summary = ""
            return ""

        captions = []
        for seg in self._segments:
            if seg.caption:
                captions.append(
                    f"[{seg.start_ts:.1f}-{seg.end_ts:.1f}s] {seg.caption}"
                )

        if not captions:
            self._global_summary = ""
            return ""

        prompt = (
            "Below are chronological descriptions of video segments from a live stream. "
            "Generate a concise global summary capturing the key content and progression. "
            "Output only the summary, no extra text.\n\n"
            + "\n".join(captions)
        )

        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=300,
            )
            self._global_summary = strip_thinking(resp.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"Failed to update global summary: {e}")
            # Fallback: concatenate last few captions
            self._global_summary = " ".join(
                seg.caption for seg in self._segments[-5:] if seg.caption
            )

        return self._global_summary

    def update_global_embedding(self, embed_fn) -> None:
        """Update global embedding by averaging all segment embeddings.

        Args:
            embed_fn: callable that takes a text string and returns np.ndarray
        """
        if self._global_summary:
            try:
                self._global_embedding = embed_fn(self._global_summary)
            except Exception as e:
                logger.error(f"Failed to compute global embedding: {e}")

    def retrieve_relevant_segments(
        self, query_embedding: np.ndarray, threshold: float = 0.3, top_k: int = 5
    ) -> List[StreamSegment]:
        """Retrieve segments most relevant to a query via cosine similarity.

        Returns segments with similarity above threshold, sorted by score descending.
        If none exceed threshold, returns the latest segment as fallback.
        """
        if not self._segments:
            return []

        scored = []
        for seg in self._segments:
            if seg.embedding is None:
                continue
            seg_emb = np.array(seg.embedding, dtype=np.float32)
            sim = float(np.dot(query_embedding, seg_emb))
            if sim > threshold:
                scored.append((sim, seg))

        if not scored:
            # Fallback: return most recent segment
            return [self._segments[-1]]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [seg for _, seg in scored[:top_k]]

    def backup(self) -> "StreamMemoryManager":
        """Create a deep-copy snapshot for query-time consistency.

        While the main manager continues processing new frames,
        the backup provides a frozen view for retrieval.
        """
        snapshot = StreamMemoryManager()
        snapshot._segments = copy.deepcopy(self._segments)
        snapshot._global_summary = self._global_summary
        if self._global_embedding is not None:
            snapshot._global_embedding = self._global_embedding.copy()
        snapshot._total_frames = self._total_frames
        snapshot._total_duration = self._total_duration
        return snapshot

    def to_stream_memory(self) -> StreamMemory:
        """Serialize to Pydantic model."""
        return StreamMemory(
            segments=copy.deepcopy(self._segments),
            global_summary=self._global_summary,
            global_embedding=(
                self._global_embedding.tolist()
                if self._global_embedding is not None
                else None
            ),
            total_frames_processed=self._total_frames,
            total_duration_sec=self._total_duration,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict."""
        return self.to_stream_memory().model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamMemoryManager":
        """Restore from a serialized dict."""
        mem = StreamMemory(**data)
        mgr = cls()
        mgr._segments = mem.segments
        mgr._global_summary = mem.global_summary
        if mem.global_embedding is not None:
            mgr._global_embedding = np.array(mem.global_embedding, dtype=np.float32)
        mgr._total_frames = mem.total_frames_processed
        mgr._total_duration = mem.total_duration_sec
        return mgr

    def get_context_for_query(self, relevant_segments: List[StreamSegment]) -> str:
        """Build a text context string from global summary + relevant segments.

        Suitable for passing to a reasoning LLM alongside a user question.
        """
        parts = []
        if self._global_summary:
            parts.append(f"[Video Overview] {self._global_summary}")

        for seg in relevant_segments:
            label = f"[{seg.start_ts:.1f}-{seg.end_ts:.1f}s]"
            if seg.caption:
                parts.append(f"{label} {seg.caption}")

        return "\n".join(parts) if parts else "(No video context available)"
