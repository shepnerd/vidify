# tests/test_core_features.py
"""Tests for the 6 production features added to agent/core/.

These tests are self-contained and do not require vLLM or external services.
"""
import os
import time
import json
import threading
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Feature 1: Retry with Exponential Backoff
# ---------------------------------------------------------------------------

class TestRetry:
    def test_succeeds_without_retry(self):
        from agent.core.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_transient_error(self):
        from agent.core.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1,
                            retryable_exceptions=(ConnectionError,))
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        assert fail_twice() == "recovered"
        assert call_count == 3

    def test_gives_up_after_max_retries(self):
        from agent.core.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.1,
                            retryable_exceptions=(TimeoutError,))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("always")

        with pytest.raises(TimeoutError):
            always_fail()
        assert call_count == 3  # 1 initial + 2 retries

    def test_does_not_retry_non_retryable(self):
        from agent.core.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1,
                            retryable_exceptions=(ConnectionError,))
        def wrong_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            wrong_error()
        assert call_count == 1  # no retry

    def test_backoff_increases_delay(self):
        from agent.core.retry import retry_with_backoff

        timestamps = []

        @retry_with_backoff(max_retries=3, base_delay=0.05, max_delay=5.0,
                            retryable_exceptions=(OSError,))
        def track_timing():
            timestamps.append(time.time())
            if len(timestamps) < 4:
                raise OSError("fail")
            return "ok"

        track_timing()
        assert len(timestamps) == 4
        # Second gap should be larger than first (exponential backoff)
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        assert gap2 > gap1 * 1.5  # allow some slack for jitter


# ---------------------------------------------------------------------------
# Feature 2: Graceful Degradation (skill_guard)
# ---------------------------------------------------------------------------

class TestSkillGuard:
    def test_passes_through_on_success(self):
        from agent.core.skill_guard import skill_guard

        @skill_guard("test_skill", optional=True, default={})
        def good_skill(x):
            return {"result": x * 2}

        assert good_skill(5) == {"result": 10}

    def test_returns_default_on_error(self):
        from agent.core.skill_guard import skill_guard

        @skill_guard("bad_skill", optional=True, default={"fallback": True})
        def bad_skill():
            raise RuntimeError("boom")

        assert bad_skill() == {"fallback": True}

    def test_returns_default_on_import_error(self):
        from agent.core.skill_guard import skill_guard

        @skill_guard("missing_dep", optional=True, default=[])
        def needs_import():
            raise ImportError("no module named 'ultralytics'")

        assert needs_import() == []

    def test_raises_when_not_optional(self):
        from agent.core.skill_guard import skill_guard

        @skill_guard("required_skill", optional=False)
        def required():
            raise RuntimeError("critical failure")

        with pytest.raises(RuntimeError):
            required()

    def test_wrapping_preserves_function_name(self):
        from agent.core.skill_guard import skill_guard

        @skill_guard("my_skill")
        def original_name():
            pass

        assert original_name.__name__ == "original_name"
        assert original_name._skill_name == "my_skill"


# ---------------------------------------------------------------------------
# Feature 3: Event System
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_subscribe_and_emit(self):
        from agent.core.events import EventBus, EventType, Event

        bus = EventBus()
        received = []

        bus.subscribe(EventType.SKILL_START, lambda e: received.append(e))
        bus.emit(Event(type=EventType.SKILL_START, skill_name="OCR", message="Starting OCR"))

        assert len(received) == 1
        assert received[0].skill_name == "OCR"

    def test_wildcard_subscriber(self):
        from agent.core.events import EventBus, EventType, Event

        bus = EventBus()
        received = []

        bus.subscribe(None, lambda e: received.append(e))
        bus.emit(Event(type=EventType.SKILL_START, skill_name="A"))
        bus.emit(Event(type=EventType.SKILL_COMPLETE, skill_name="B"))

        assert len(received) == 2

    def test_emit_helpers(self):
        from agent.core.events import EventBus, EventType

        bus = EventBus()
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        bus.emit_skill_start("ASR", progress_pct=10)
        bus.emit_skill_complete("ASR", progress_pct=30)
        bus.emit_skill_error("OCR", "failed", progress_pct=40)
        bus.emit_skill_skipped("Detection", reason="not installed", progress_pct=50)
        bus.emit_progress("halfway", 50)

        assert len(received) == 5
        assert received[0].type == EventType.SKILL_START
        assert received[1].type == EventType.SKILL_COMPLETE
        assert received[2].type == EventType.SKILL_ERROR
        assert received[3].type == EventType.SKILL_SKIPPED
        assert received[4].type == EventType.PROGRESS

    def test_event_to_sse(self):
        from agent.core.events import Event, EventType

        event = Event(type=EventType.SKILL_START, skill_name="OCR", message="Starting",
                      progress_pct=25.5)
        sse = event.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["type"] == "skill_start"
        assert data["skill_name"] == "OCR"
        assert data["progress_pct"] == 25.5

    def test_history_tracking(self):
        from agent.core.events import EventBus, EventType, Event

        bus = EventBus()
        bus.emit(Event(type=EventType.PROGRESS, message="a"))
        bus.emit(Event(type=EventType.PROGRESS, message="b"))

        assert len(bus.history) == 2
        assert bus.history[0].message == "a"

    def test_unsubscribe(self):
        from agent.core.events import EventBus, EventType, Event

        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)

        bus.subscribe(EventType.PROGRESS, cb)
        bus.emit(Event(type=EventType.PROGRESS, message="before"))
        bus.unsubscribe(EventType.PROGRESS, cb)
        bus.emit(Event(type=EventType.PROGRESS, message="after"))

        assert len(received) == 1

    def test_thread_safety(self):
        from agent.core.events import EventBus, EventType, Event

        bus = EventBus()
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        def emit_many():
            for i in range(100):
                bus.emit(Event(type=EventType.PROGRESS, message=str(i)))

        threads = [threading.Thread(target=emit_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 500


# ---------------------------------------------------------------------------
# Feature 4: Parallel Skill Execution
# ---------------------------------------------------------------------------

class TestParallel:
    def test_runs_skills_and_collects_results(self):
        from agent.core.parallel import run_skills_parallel

        def skill_a(x):
            return {"a": x}

        def skill_b(x, y):
            return {"b": x + y}

        results = run_skills_parallel([
            ("a", skill_a, (10,), {}),
            ("b", skill_b, (3, 4), {}),
        ], max_workers=2)

        assert results["a"] == {"a": 10}
        assert results["b"] == {"b": 7}

    def test_isolates_failures(self):
        from agent.core.parallel import run_skills_parallel

        def good():
            return "ok"

        def bad():
            raise RuntimeError("boom")

        results = run_skills_parallel([
            ("good", good, (), {}),
            ("bad", bad, (), {}),
        ], max_workers=2)

        assert results["good"] == "ok"
        assert results["bad"] == {}  # default on failure

    def test_empty_skills_list(self):
        from agent.core.parallel import run_skills_parallel
        assert run_skills_parallel([]) == {}

    def test_actual_parallelism(self):
        from agent.core.parallel import run_skills_parallel

        def slow_skill(name):
            time.sleep(0.1)
            return name

        start = time.time()
        results = run_skills_parallel([
            ("a", slow_skill, ("a",), {}),
            ("b", slow_skill, ("b",), {}),
            ("c", slow_skill, ("c",), {}),
        ], max_workers=3)
        elapsed = time.time() - start

        assert len(results) == 3
        # Should take ~0.1s (parallel), not ~0.3s (sequential)
        assert elapsed < 0.25, f"Expected parallel execution, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Feature 5: Structured Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_structured_formatter(self):
        import logging
        from agent.core.logging_config import StructuredFormatter

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        record.skill_name = "OCR"
        record.duration_ms = 123
        record.status = "ok"

        output = formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "hello"
        assert data["skill_name"] == "OCR"
        assert data["duration_ms"] == 123
        assert data["status"] == "ok"

    def test_setup_logging_text(self):
        from agent.core.logging_config import setup_logging
        setup_logging(log_format="text")
        # Should not raise

    def test_setup_logging_json(self):
        from agent.core.logging_config import setup_logging
        setup_logging(log_format="json")
        # Should not raise

    def test_workflow_tracker(self):
        from agent.core.logging_config import WorkflowTracker

        tracker = WorkflowTracker(workflow_name="detailed", video_id="abc123")
        tracker.record("ASR", 1500, "ok")
        tracker.record("OCR", 800, "ok")
        tracker.record("Detection", 0, "skipped")

        summary = tracker.summary()
        assert "detailed" in summary
        assert "ASR" in summary
        assert "1500ms" in summary

        d = tracker.summary_dict()
        assert d["workflow"] == "detailed"
        assert len(d["skills"]) == 3
        assert d["skills"][0]["name"] == "ASR"

    def test_log_skill_execution_decorator(self):
        from agent.core.logging_config import log_skill_execution, WorkflowTracker

        tracker = WorkflowTracker("test")

        @log_skill_execution("TestSkill", tracker=tracker)
        def some_skill():
            time.sleep(0.05)
            return "done"

        result = some_skill()
        assert result == "done"
        assert len(tracker.timings) == 1
        assert tracker.timings[0].name == "TestSkill"
        assert tracker.timings[0].status == "ok"
        assert tracker.timings[0].duration_ms >= 40  # at least ~50ms

    def test_log_skill_execution_on_error(self):
        from agent.core.logging_config import log_skill_execution, WorkflowTracker

        tracker = WorkflowTracker("test")

        @log_skill_execution("FailSkill", tracker=tracker)
        def failing():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            failing()
        assert len(tracker.timings) == 1
        assert tracker.timings[0].status == "error"


# ---------------------------------------------------------------------------
# Feature 6: Hook System
# ---------------------------------------------------------------------------

class TestHooks:
    def test_empty_hooks(self):
        from agent.core.hooks import HookManager

        mgr = HookManager(hooks_path="/nonexistent/path.yaml")
        assert not mgr.has_hooks
        # Should not raise
        mgr.trigger("post_analysis")

    def test_load_hooks_from_yaml(self):
        from agent.core.hooks import HookManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
hooks:
  post_analysis:
    - command: "echo done"
      timeout: 5
  on_error:
    - command: "echo error"
""")
            f.flush()
            mgr = HookManager(hooks_path=f.name)

        try:
            assert mgr.has_hooks
        finally:
            os.unlink(f.name)

    def test_sync_hook_execution(self):
        from agent.core.hooks import HookManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yf:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as out:
                yf.write(f"""
hooks:
  post_analysis:
    - command: "echo $VIDEO_URI > {out.name}"
      timeout: 5
""")
                yf.flush()
                mgr = HookManager(hooks_path=yf.name)
                mgr.trigger("post_analysis", {"VIDEO_URI": "test_video_123"})

                with open(out.name) as f:
                    content = f.read().strip()

        try:
            assert content == "test_video_123"
        finally:
            os.unlink(yf.name)
            os.unlink(out.name)

    def test_async_hook_execution(self):
        from agent.core.hooks import HookManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yf:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as out:
                yf.write(f"""
hooks:
  on_error:
    - command: "echo $ERROR_MSG > {out.name}"
      async: true
      timeout: 5
""")
                yf.flush()
                mgr = HookManager(hooks_path=yf.name)
                mgr.trigger("on_error", {"ERROR_MSG": "something_broke"})

                # Async hook runs in a thread — give it a moment
                time.sleep(0.5)

                with open(out.name) as f:
                    content = f.read().strip()

        try:
            assert content == "something_broke"
        finally:
            os.unlink(yf.name)
            os.unlink(out.name)

    def test_hook_timeout(self):
        from agent.core.hooks import HookManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yf:
            yf.write("""
hooks:
  pre_analysis:
    - command: "sleep 10"
      timeout: 1
""")
            yf.flush()
            mgr = HookManager(hooks_path=yf.name)

            start = time.time()
            mgr.trigger("pre_analysis")  # should timeout after ~1s
            elapsed = time.time() - start

        try:
            assert elapsed < 3, f"Hook should have timed out in ~1s, took {elapsed:.1f}s"
        finally:
            os.unlink(yf.name)

    def test_no_trigger_for_unconfigured_hook(self):
        from agent.core.hooks import HookManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yf:
            yf.write("""
hooks:
  post_analysis:
    - command: "echo should_not_run"
""")
            yf.flush()
            mgr = HookManager(hooks_path=yf.name)
            # Triggering a different hook point should do nothing
            mgr.trigger("on_error")  # no hooks configured for on_error

        os.unlink(yf.name)
