# agent/core/hooks.py
"""Lifecycle hook system for analysis pipelines.

Inspired by Claude Code's 30+ lifecycle hooks with configurable shell commands.
Vidify uses a simpler version focused on analysis lifecycle events.

Hook points:
  pre_analysis   — before any workflow starts
  post_analysis  — after workflow completes successfully
  post_skill     — after each skill completes (env: $SKILL_NAME)
  on_error       — when a workflow or skill fails (env: $ERROR_MSG)
  post_highlight — after highlight clips are exported
  post_index     — after FAISS index is built

Hooks are configured in hooks.yaml:

    hooks:
      post_analysis:
        - command: "curl -X POST $WEBHOOK_URL -d @$RESULT_PATH"
          async: true
          timeout: 10
      on_error:
        - command: "echo 'Failed: $ERROR_MSG' >> errors.log"
"""
import os
import logging
import subprocess
import threading
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

HOOK_POINTS = (
    "pre_analysis", "post_analysis", "post_skill",
    "on_error", "post_highlight", "post_index",
)


class HookManager:
    """Loads and executes lifecycle hooks from hooks.yaml."""

    def __init__(self, hooks_path: str = "hooks.yaml"):
        self._hooks: Dict[str, List[dict]] = {}
        self._load(hooks_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            self._hooks = data.get("hooks", {})
            count = sum(len(v) for v in self._hooks.values())
            if count:
                logger.info("[hooks] Loaded %d hook(s) from %s", count, path)
        except Exception as e:
            logger.warning("[hooks] Failed to load %s: %s", path, e)

    def trigger(self, hook_point: str, env_extra: Optional[Dict[str, str]] = None):
        """Fire all hooks registered for the given hook point.

        Args:
            hook_point: One of HOOK_POINTS.
            env_extra: Additional environment variables passed to the hook command.
                Common keys: VIDEO_URI, CACHE_DIR, RESULT_PATH, SKILL_NAME, ERROR_MSG.
        """
        entries = self._hooks.get(hook_point, [])
        if not entries:
            return

        env = {**os.environ}
        if env_extra:
            env.update({k: str(v) for k, v in env_extra.items()})

        for entry in entries:
            cmd = entry.get("command", "")
            if not cmd:
                continue
            is_async = entry.get("async", False)
            timeout = entry.get("timeout", 30)

            if is_async:
                threading.Thread(
                    target=self._run_hook, args=(hook_point, cmd, env, timeout),
                    daemon=True,
                ).start()
            else:
                self._run_hook(hook_point, cmd, env, timeout)

    def _run_hook(self, hook_point: str, cmd: str, env: dict, timeout: int):
        try:
            result = subprocess.run(
                cmd, shell=True, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                logger.warning(
                    "[hooks] %s hook failed (rc=%d): %s",
                    hook_point, result.returncode, result.stderr.strip(),
                )
            else:
                logger.debug("[hooks] %s hook completed: %s", hook_point, cmd[:80])
        except subprocess.TimeoutExpired:
            logger.warning("[hooks] %s hook timed out after %ds: %s", hook_point, timeout, cmd[:80])
        except Exception as e:
            logger.warning("[hooks] %s hook error: %s", hook_point, e)

    @property
    def has_hooks(self) -> bool:
        return bool(self._hooks)


# Global hook manager — initialized lazily
_hook_manager: Optional[HookManager] = None


def get_hook_manager(hooks_path: str = "hooks.yaml") -> HookManager:
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager(hooks_path)
    return _hook_manager
