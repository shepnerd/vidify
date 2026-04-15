from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from agent.config import get_default_config
from agent.core.orchestrator import normalize_mode, run
from agent.extensions.skills.video_io import load_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HERMES_SKILLS_ROOT = Path.home() / ".hermes" / "skills"
HERMES_SKILL_SOURCE_DIR = PROJECT_ROOT / ".agents" / "skills" / "media" / "vidify"


def _build_config(config: Optional[Dict[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    merged = {**get_default_config(), **(config or {})}
    merged.update({key: value for key, value in overrides.items() if value is not None})
    if "mode" in merged:
        merged["mode"] = normalize_mode(merged["mode"])
    return merged


def analyze_video(
    source_type: str,
    uri: str,
    mode: str = "brief",
    config: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Stable Hermes-facing Python API for running Vidify workflows."""
    cfg = _build_config(config, source_type=source_type, uri=uri, mode=mode, **overrides)
    asset = load_video(source_type, uri, cfg["cache_root"])
    return run(asset, cfg["mode"], cfg)


def ask_video(
    source_type: str,
    uri: str,
    question: str,
    config: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    return analyze_video(
        source_type,
        uri,
        mode="ask",
        config=config,
        question=question,
        **overrides,
    )


def build_index(
    source_type: str,
    uri: str,
    config: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    return analyze_video(source_type, uri, mode="index", config=config, **overrides)


def generate_highlights(
    source_type: str,
    uri: str,
    config: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    return analyze_video(source_type, uri, mode="highlights", config=config, **overrides)


def get_skill_source_dir() -> Path:
    if not HERMES_SKILL_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Hermes skill assets not found at {HERMES_SKILL_SOURCE_DIR}"
        )
    return HERMES_SKILL_SOURCE_DIR


def install_skill(
    dest_root: str | os.PathLike[str] | None = None,
    strategy: str = "symlink",
    force: bool = False,
) -> Path:
    """Install the repo's Hermes skill into a user Hermes skill directory."""
    if strategy not in {"symlink", "copy"}:
        raise ValueError("strategy must be 'symlink' or 'copy'")

    source_dir = get_skill_source_dir()
    root = Path(dest_root).expanduser() if dest_root else DEFAULT_HERMES_SKILLS_ROOT
    destination = root / "media" / "vidify"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if not force:
            raise FileExistsError(
                f"{destination} already exists. Re-run with force=True to replace it."
            )
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)

    if strategy == "symlink":
        destination.symlink_to(source_dir, target_is_directory=True)
    else:
        shutil.copytree(source_dir, destination)

    return destination

