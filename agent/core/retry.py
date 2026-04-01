# agent/core/retry.py
"""Retry with exponential backoff for transient failures.

Inspired by Claude Code's multi-layer error recovery pattern:
- Exponential backoff with jitter to avoid thundering herd
- Configurable retryable exceptions
- Logging at each retry attempt
"""
import time
import random
import logging
import functools
from typing import Tuple, Type

logger = logging.getLogger(__name__)

# Default exceptions considered transient / retryable
RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    OSError,  # covers network-level errors
)

# Try to include httpx/openai-specific errors if available
try:
    from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (
        APITimeoutError, APIConnectionError, InternalServerError, RateLimitError,
    )
except ImportError:
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: Tuple[Type[BaseException], ...] = None,
):
    """Decorator that retries a function on transient failures with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types to retry on.
            Defaults to RETRYABLE_EXCEPTIONS (network errors, timeouts, 5xx).
    """
    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_EXCEPTIONS

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            "[retry] %s failed after %d attempts: %s",
                            func.__name__, max_retries + 1, e,
                        )
                        raise
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.25)
                    sleep_time = delay + jitter
                    logger.warning(
                        "[retry] %s attempt %d/%d failed (%s: %s), retrying in %.1fs...",
                        func.__name__, attempt + 1, max_retries + 1,
                        type(e).__name__, e, sleep_time,
                    )
                    time.sleep(sleep_time)
            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator
