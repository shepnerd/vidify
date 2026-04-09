# agent/core/cli_progress.py
"""Rich Live progress display for CLI video analysis.

Subscribes to the global EventBus and renders a live-updating terminal display
with an overall progress bar, current-skill spinner, and completed-skill log.
Thread-safe — events may arrive from parallel worker threads.
"""
import time
import threading
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from agent.core.events import EventBus, Event, EventType


_STATUS_ICON = {
    "ok": "[green]✓[/green]",
    "error": "[red]✗[/red]",
    "skipped": "[dim]–[/dim]",
}


class CLIProgressDisplay:
    """Rich Live display driven by EventBus events.

    Usage::

        with CLIProgressDisplay(event_bus):
            result = run(asset, mode, cfg)
    """

    def __init__(self, bus: EventBus, console: Optional[Console] = None):
        self._bus = bus
        self._console = console or Console(stderr=True)
        self._lock = threading.Lock()

        # State
        self._current_skill: str = ""
        self._current_msg: str = ""
        self._pct: float = 0.0
        self._completed: list[tuple[str, str, str]] = []  # (icon, name, duration)
        self._skill_start_time: float = 0.0
        self._start_time: float = 0.0

        # Rich widgets
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._task_id = self._progress.add_task("Analyzing video", total=100)
        self._live: Optional[Live] = None

    # ── EventBus callback (called from any thread) ──────────────────────

    def _on_event(self, event: Event):
        with self._lock:
            if event.progress_pct:
                self._pct = event.progress_pct
                self._progress.update(self._task_id, completed=self._pct)

            if event.type == EventType.SKILL_START:
                self._current_skill = event.skill_name
                self._current_msg = event.message
                self._skill_start_time = time.time()

            elif event.type == EventType.SKILL_COMPLETE:
                dur = self._format_duration(time.time() - self._skill_start_time)
                self._completed.append((_STATUS_ICON["ok"], event.skill_name, dur))
                self._current_skill = ""

            elif event.type == EventType.SKILL_ERROR:
                dur = self._format_duration(time.time() - self._skill_start_time)
                self._completed.append((_STATUS_ICON["error"], event.skill_name, dur))
                self._current_skill = ""

            elif event.type == EventType.SKILL_SKIPPED:
                self._completed.append((_STATUS_ICON["skipped"], event.skill_name, "–"))
                self._current_skill = ""

            elif event.type == EventType.WORKFLOW_COMPLETE:
                self._pct = 100
                self._progress.update(self._task_id, completed=100)

        # Trigger a Live refresh (thread-safe in rich)
        if self._live:
            self._live.refresh()

    # ── Renderable ──────────────────────────────────────────────────────

    def __rich_console__(self, console, options):
        """Make CLIProgressDisplay itself a renderable for rich.live.Live."""
        yield from console.render(self._build_display(), options)

    def _build_display(self) -> Group:
        with self._lock:
            # Completed skills table
            table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
            table.add_column(width=2)
            table.add_column(ratio=1)
            table.add_column(justify="right", width=10)
            for icon, name, dur in self._completed:
                table.add_row(icon, name, f"[dim]{dur}[/dim]")

            # Current skill spinner
            if self._current_skill:
                spinner_text = Text.from_markup(
                    f"  [bold cyan]>[/bold cyan] {self._current_msg or self._current_skill}"
                )
            else:
                spinner_text = Text("")

        return Group(self._progress, Text(""), table, spinner_text)

    # ── Context manager ─────────────────────────────────────────────────

    def __enter__(self):
        self._start_time = time.time()
        self._bus.subscribe(None, self._on_event)
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._bus.unsubscribe(None, self._on_event)
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None

        # Print final summary
        elapsed = self._format_duration(time.time() - self._start_time)
        self._console.print()
        if exc_type is None:
            self._console.print(f"[bold green]Done[/bold green] in {elapsed}")
        else:
            self._console.print(f"[bold red]Failed[/bold red] after {elapsed}")
        return False

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
