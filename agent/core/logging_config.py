# agent/core/logging_config.py
"""Structured logging with skill execution tracking.

Inspired by Claude Code's OpenTelemetry spans and cost tracking:
- JSON structured logging with video_id, skill_name, duration_ms
- @log_skill_execution decorator for automatic timing
- Summary table of skill performance at workflow end
"""
import time
import json
import logging
import functools
from typing import Optional, Dict, List
from dataclasses import dataclass, field


# ── JSON Formatter ────────────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Outputs log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra fields attached by log_skill_execution
        for key in ("video_id", "skill_name", "duration_ms", "status"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(log_format: str = "text", level: int = logging.INFO):
    """Configure root logger.

    Args:
        log_format: "json" for structured JSON, "text" for human-readable.
        level: Logging level.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
    root.addHandler(handler)


# ── Skill Execution Tracker ──────────────────────────────────────────────

@dataclass
class SkillTiming:
    name: str
    duration_ms: float
    status: str  # "ok", "error", "skipped"


class WorkflowTracker:
    """Collects timing data for all skills in a workflow run."""

    def __init__(self, workflow_name: str = "", video_id: str = ""):
        self.workflow_name = workflow_name
        self.video_id = video_id
        self.timings: List[SkillTiming] = []
        self._start_time = time.time()

    def record(self, name: str, duration_ms: float, status: str = "ok"):
        self.timings.append(SkillTiming(name=name, duration_ms=duration_ms, status=status))

    def summary(self) -> str:
        """Format a summary table of skill durations."""
        total_ms = (time.time() - self._start_time) * 1000
        lines = [
            f"\n{'='*60}",
            f"  Workflow: {self.workflow_name}  |  Video: {self.video_id}",
            f"  Total: {total_ms:.0f}ms",
            f"{'='*60}",
            f"  {'Skill':<30} {'Duration':>10} {'Status':>8}",
            f"  {'-'*30} {'-'*10} {'-'*8}",
        ]
        for t in self.timings:
            dur = f"{t.duration_ms:.0f}ms"
            lines.append(f"  {t.name:<30} {dur:>10} {t.status:>8}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        total_ms = (time.time() - self._start_time) * 1000
        return {
            "workflow": self.workflow_name,
            "video_id": self.video_id,
            "total_ms": round(total_ms),
            "skills": [
                {"name": t.name, "duration_ms": round(t.duration_ms), "status": t.status}
                for t in self.timings
            ],
        }


def log_skill_execution(skill_name: str, tracker: Optional[WorkflowTracker] = None):
    """Decorator that logs skill execution timing.

    Automatically records:
    - skill start (INFO)
    - skill completion with duration_ms (INFO)
    - skill failure with duration_ms (ERROR)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                logger.info(
                    "%s completed in %.0fms",
                    skill_name, duration_ms,
                    extra={"skill_name": skill_name, "duration_ms": round(duration_ms), "status": "ok"},
                )
                if tracker:
                    tracker.record(skill_name, duration_ms, "ok")
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                logger.error(
                    "%s failed after %.0fms: %s",
                    skill_name, duration_ms, e,
                    extra={"skill_name": skill_name, "duration_ms": round(duration_ms), "status": "error"},
                )
                if tracker:
                    tracker.record(skill_name, duration_ms, "error")
                raise
        return wrapper
    return decorator
