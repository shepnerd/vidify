# agent/core/parallel.py
"""Parallel skill and segment execution for independent processing steps.

Inspired by Claude Code's concurrent tool execution pattern and
AgentScope's fanout_pipeline:
- Concurrent-safe tools run in parallel via ThreadPoolExecutor
- Non-concurrent tools run serially
- Segment-level parallelism for long video processing
- Results collected with error isolation per skill/segment
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def run_skills_parallel(
    skills: List[Tuple[str, Callable, tuple, dict]],
    max_workers: int = 3,
) -> Dict[str, Any]:
    """Run multiple independent skills in parallel.

    Args:
        skills: List of (name, function, args, kwargs) tuples.
        max_workers: Maximum concurrent threads.

    Returns:
        Dict mapping skill name → result. Failed skills map to {}.
    """
    results: Dict[str, Any] = {}

    if not skills:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {}
        for name, fn, args, kwargs in skills:
            future = executor.submit(fn, *args, **kwargs)
            future_to_name[future] = name

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                logger.info("[parallel] %s completed", name)
            except Exception as e:
                logger.warning("[parallel] %s failed: %s: %s", name, type(e).__name__, e)
                results[name] = {}

    return results


def run_segments_parallel(
    segments: list,
    worker_fn: Callable,
    worker_kwargs: dict,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Run a worker function on multiple video segments in parallel.

    Uses ThreadPoolExecutor (safe because most heavy lifting is in
    subprocesses — FFmpeg, PaddleOCR — or GPU inference on a shared
    vLLM server which handles its own concurrency).

    Args:
        segments: List of VideoSegment objects.
        worker_fn: Function that takes (segment, **worker_kwargs) and returns a dict.
        worker_kwargs: Shared keyword arguments passed to every worker call.
        max_workers: Maximum concurrent segment workers.

    Returns:
        List of result dicts, ordered by segment index.
        Failed segments return {} and log a warning.
    """
    if not segments:
        return []

    results: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seg = {}
        for seg in segments:
            future = executor.submit(worker_fn, segment=seg, **worker_kwargs)
            future_to_seg[future] = seg

        for future in as_completed(future_to_seg):
            seg = future_to_seg[future]
            label = f"seg_{seg.index:03d}"
            try:
                results[seg.index] = future.result()
                logger.info("[parallel_segments] %s completed", label)
            except Exception as e:
                logger.warning(
                    "[parallel_segments] %s failed: %s: %s",
                    label, type(e).__name__, e,
                )
                results[seg.index] = {}

    # Return in segment order
    return [results.get(i, {}) for i in range(len(segments))]
