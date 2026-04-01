# agent/core/events.py
"""Event bus for streaming progress and lifecycle notifications.

Inspired by Claude Code's StreamEvent generator and tool progress callbacks.
Provides real-time progress reporting for long-running video analysis pipelines.
"""
import time
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    SKILL_START = "skill_start"
    SKILL_COMPLETE = "skill_complete"
    SKILL_ERROR = "skill_error"
    SKILL_SKIPPED = "skill_skipped"
    PROGRESS = "progress"


@dataclass
class Event:
    type: EventType
    skill_name: str = ""
    message: str = ""
    progress_pct: float = 0.0
    detail: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = {
            "type": self.type.value,
            "skill_name": self.skill_name,
            "message": self.message,
            "progress_pct": round(self.progress_pct, 1),
            "timestamp": self.timestamp,
        }
        if self.detail:
            d["detail"] = self.detail
        return d

    def to_sse(self) -> str:
        """Format as a Server-Sent Event line."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class EventBus:
    """Simple publish-subscribe event bus for pipeline progress.

    Thread-safe: subscribers can be added/removed from any thread.
    Events are dispatched synchronously to subscribers (fire-and-forget).
    """

    def __init__(self):
        self._subscribers: Dict[Optional[EventType], List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._history: List[Event] = []

    def subscribe(self, event_type: Optional[EventType], callback: Callable[[Event], None]):
        """Subscribe to events. Pass event_type=None to subscribe to all events."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: Optional[EventType], callback: Callable[[Event], None]):
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb is not callback
                ]

    def emit(self, event: Event):
        """Emit an event to all matching subscribers."""
        self._history.append(event)
        with self._lock:
            # Notify type-specific subscribers
            for cb in self._subscribers.get(event.type, []):
                try:
                    cb(event)
                except Exception as e:
                    logger.debug("Event subscriber error: %s", e)
            # Notify wildcard subscribers (subscribed to None)
            for cb in self._subscribers.get(None, []):
                try:
                    cb(event)
                except Exception as e:
                    logger.debug("Event subscriber error: %s", e)

    def emit_skill_start(self, skill_name: str, message: str = "", progress_pct: float = 0.0):
        self.emit(Event(
            type=EventType.SKILL_START,
            skill_name=skill_name,
            message=message or f"Starting {skill_name}...",
            progress_pct=progress_pct,
        ))

    def emit_skill_complete(self, skill_name: str, message: str = "", progress_pct: float = 0.0,
                            detail: dict = None):
        self.emit(Event(
            type=EventType.SKILL_COMPLETE,
            skill_name=skill_name,
            message=message or f"{skill_name} complete",
            progress_pct=progress_pct,
            detail=detail,
        ))

    def emit_skill_error(self, skill_name: str, error: str, progress_pct: float = 0.0):
        self.emit(Event(
            type=EventType.SKILL_ERROR,
            skill_name=skill_name,
            message=f"{skill_name} failed: {error}",
            progress_pct=progress_pct,
        ))

    def emit_skill_skipped(self, skill_name: str, reason: str = "", progress_pct: float = 0.0):
        self.emit(Event(
            type=EventType.SKILL_SKIPPED,
            skill_name=skill_name,
            message=f"{skill_name} skipped: {reason}",
            progress_pct=progress_pct,
        ))

    def emit_progress(self, message: str, progress_pct: float):
        self.emit(Event(
            type=EventType.PROGRESS,
            message=message,
            progress_pct=progress_pct,
        ))

    @property
    def history(self) -> List[Event]:
        return list(self._history)

    def clear(self):
        self._history.clear()
        with self._lock:
            self._subscribers.clear()


# Global event bus instance — workflows emit to this, CLI/API subscribe
event_bus = EventBus()
