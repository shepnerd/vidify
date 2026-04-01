# agent/core/parallel.py
"""Parallel skill execution for independent processing steps.

Inspired by Claude Code's concurrent tool execution pattern:
- Concurrent-safe tools run in parallel via ThreadPoolExecutor
- Non-concurrent tools run serially
- Results collected with error isolation per skill
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
