"""Live streaming workflow — orchestrates real-time video processing.

Composes: live_stream_processing + scene_similarity + stream_memory
into a managed pipeline with optional live Q&A.
"""
import logging
import threading
from typing import Dict, Any, Optional, Callable

from agent.config import load_models_config, load_workflows_config
from agent.core.schemas import StreamConfig
from agent.extensions.models.vllm_openai_client import make_client
from agent.extensions.skills.live_stream_processing import LiveStreamProcessor
from agent.extensions.skills.scene_similarity import compute_frame_embedding
from agent.extensions.skills.stream_memory import StreamMemoryManager

logger = logging.getLogger(__name__)


def _make_embed_fn(embed_client, embed_model: str):
    """Create an embedding function from an OpenAI-compatible client."""
    import numpy as np

    def embed_fn(text: str):
        resp = embed_client.embeddings.create(model=embed_model, input=[text])
        vec = np.array(resp.data[0].embedding, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    return embed_fn


class LiveSession:
    """A managed live streaming session with background processing and live Q&A."""

    def __init__(self, session_id: str, processor: LiveStreamProcessor,
                 llm_client, llm_model: str, embed_fn):
        self.session_id = session_id
        self.processor = processor
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embed_fn = embed_fn
        self._thread: Optional[threading.Thread] = None
        self._results: list = []

    def _on_frame(self, analysis: Dict[str, Any]) -> None:
        self._results.append(analysis)

    def start(self) -> None:
        """Start processing in a background thread."""
        self._thread = threading.Thread(
            target=self.processor.start,
            kwargs={"callback": self._on_frame},
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Live session {self.session_id} started")

    def ask(self, question: str) -> Dict[str, Any]:
        """Query the stream memory (thread-safe, uses backup-on-query)."""
        return self.processor.query(
            question=question,
            llm_client=self.llm_client,
            model_name=self.llm_model,
            embed_fn=self.embed_fn,
        )

    def stop(self) -> Dict[str, Any]:
        """Stop the stream and return final state."""
        memory = self.processor.stop()
        if self._thread:
            self._thread.join(timeout=10)

        # Update global summary one last time
        memory.update_global_summary(self.llm_client, self.llm_model)

        return {
            "session_id": self.session_id,
            "memory": memory.to_dict(),
            "total_frame_results": len(self._results),
        }

    def status(self) -> Dict[str, Any]:
        mem = self.processor.memory
        return {
            "session_id": self.session_id,
            "running": self.processor._running,
            "segments_processed": mem.segment_count,
            "total_frames": mem._total_frames,
            "total_duration_sec": mem._total_duration,
            "global_summary": mem.global_summary,
        }


def wf_live(source: str, stream_url: Optional[str], cfg: Dict[str, Any],
            on_result_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run the live streaming workflow synchronously.

    Blocks until stream ends or is interrupted. Returns final memory state.
    """
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    ls_cfg = workflows_config.get("live_stream", {})

    config = StreamConfig(
        source=source or ls_cfg.get("source", "webcam"),
        stream_url=stream_url,
        fps=ls_cfg.get("fps", 1),
        heavy_interval=ls_cfg.get("heavy_interval", 5),
        similarity_threshold=ls_cfg.get("similarity_threshold", 0.9),
        min_segment_frames=ls_cfg.get("min_segment_frames", 3),
        max_segment_frames=ls_cfg.get("max_segment_frames", 16),
    )

    processor = LiveStreamProcessor(config, models_config, workflows_config)
    results: list = []

    def _callback(analysis):
        results.append(analysis)
        if on_result_callback:
            on_result_callback(analysis)

    processor.start(callback=_callback)

    # After stream ends, generate global summary
    llm_client = make_client(cfg.get("llm_base_url", "http://localhost:8000/v1"))
    llm_model = cfg.get("llm_model", "qwen3.5-9b")
    processor.memory.update_global_summary(llm_client, llm_model)

    return {
        "memory": processor.memory.to_dict(),
        "total_frame_results": len(results),
    }


def create_live_session(session_id: str, cfg: Dict[str, Any]) -> LiveSession:
    """Create a managed live session for the API server.

    Returns a LiveSession that can be started, queried, and stopped.
    """
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    ls_cfg = workflows_config.get("live_stream", {})

    config = StreamConfig(
        source=cfg.get("source", ls_cfg.get("source", "webcam")),
        stream_url=cfg.get("stream_url"),
        fps=cfg.get("fps", ls_cfg.get("fps", 1)),
        heavy_interval=cfg.get("heavy_interval", ls_cfg.get("heavy_interval", 5)),
        similarity_threshold=ls_cfg.get("similarity_threshold", 0.9),
        min_segment_frames=ls_cfg.get("min_segment_frames", 3),
        max_segment_frames=ls_cfg.get("max_segment_frames", 16),
    )

    processor = LiveStreamProcessor(config, models_config, workflows_config)
    llm_client = make_client(cfg.get("llm_base_url", "http://localhost:8000/v1"))
    llm_model = cfg.get("llm_model", "qwen3.5-9b")

    embed_client = make_client(cfg.get("embed_base_url", "http://localhost:8000/v1"))
    embed_model = cfg.get("embed_model", "qwen-embed")
    embed_fn = _make_embed_fn(embed_client, embed_model)

    return LiveSession(
        session_id=session_id,
        processor=processor,
        llm_client=llm_client,
        llm_model=llm_model,
        embed_fn=embed_fn,
    )
