from pathlib import Path

import pytest

from agent.integrations import hermes


def test_analyze_video_uses_defaults_and_normalizes_mode(monkeypatch):
    seen = {}

    def fake_load_video(source_type, uri, cache_root):
        seen["load_video"] = (source_type, uri, cache_root)
        return {"asset": True}

    def fake_run(asset, mode, cfg):
        seen["run"] = (asset, mode, cfg)
        return {"mode": mode, "cache_root": cfg["cache_root"]}

    monkeypatch.setattr(hermes, "load_video", fake_load_video)
    monkeypatch.setattr(hermes, "run", fake_run)

    result = hermes.analyze_video("youtube", "https://example.com/v", mode="quick")

    assert seen["load_video"] == ("youtube", "https://example.com/v", "./cache")
    assert seen["run"][1] == "brief"
    assert result == {"mode": "brief", "cache_root": "./cache"}


def test_ask_video_passes_question(monkeypatch):
    seen = {}

    def fake_analyze(source_type, uri, mode="brief", config=None, **overrides):
        seen["call"] = (source_type, uri, mode, config, overrides)
        return {"ok": True}

    monkeypatch.setattr(hermes, "analyze_video", fake_analyze)

    result = hermes.ask_video("local", "/tmp/video.mp4", "what happened?")

    assert result == {"ok": True}
    assert seen["call"][2] == "ask"
    assert seen["call"][4]["question"] == "what happened?"


def test_install_skill_copy(tmp_path):
    destination = hermes.install_skill(dest_root=tmp_path, strategy="copy")
    assert destination == tmp_path / "media" / "vidify"
    assert destination.joinpath("SKILL.md").exists()
    assert destination.joinpath("scripts", "vidify-analyze.sh").exists()


def test_install_skill_rejects_existing_without_force(tmp_path):
    destination = tmp_path / "media" / "vidify"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        hermes.install_skill(dest_root=tmp_path, strategy="copy")


def test_get_skill_source_dir_points_at_repo_skill():
    source_dir = hermes.get_skill_source_dir()
    assert isinstance(source_dir, Path)
    assert source_dir.joinpath("SKILL.md").exists()
