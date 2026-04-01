# agent/core/skill_guard.py
"""Graceful degradation for optional skills.

Inspired by Claude Code's handling of missing MCP servers and broken plugins:
- Log and skip optional skills instead of crashing the pipeline
- Return structured SkillResult with status and reason
- Required skills still raise on failure
"""
import logging
import functools
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result wrapper that tracks whether a skill succeeded, was skipped, or failed."""
    status: str  # "ok", "skipped", "error"
    data: Any = None
    reason: Optional[str] = None
    skill_name: Optional[str] = None


def skill_guard(skill_name: str = None, optional: bool = True, default=None):
    """Decorator that catches errors from optional skills and returns a default value.

    Args:
        skill_name: Human-readable name for logging. Defaults to function name.
        optional: If True, catch errors and return default. If False, re-raise.
        default: Value to return when the skill fails/is unavailable.
            Defaults to empty dict {}.
    """
    if default is None:
        default = {}

    def decorator(func):
        name = skill_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except ImportError as e:
                if not optional:
                    raise
                logger.warning(
                    "[skill_guard] %s skipped — dependency not installed: %s",
                    name, e,
                )
                return default
            except Exception as e:
                if not optional:
                    raise
                logger.warning(
                    "[skill_guard] %s failed, degrading gracefully: %s: %s",
                    name, type(e).__name__, e,
                )
                return default

        # Expose metadata for introspection
        wrapper._skill_name = name
        wrapper._optional = optional
        return wrapper

    return decorator
