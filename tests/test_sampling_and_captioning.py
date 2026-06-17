from types import SimpleNamespace

from agent.core.schemas import FrameItem, FrameSet, FrameStrategy
from agent.extensions.skills import frame_sampler, vision_caption


def test_resolve_frame_budget_adapts_to_duration():
    strategy = FrameStrategy(type="scene", params={
        "max_frames": 64,
        "min_frames": 16,
        "adaptive_by_duration": True,
    })

    assert frame_sampler._resolve_frame_budget(strategy, 5.0) == 16
    assert frame_sampler._resolve_frame_budget(strategy, 40.0) == 40
    assert frame_sampler._resolve_frame_budget(strategy, 500.0) == 64


def test_sample_frames_uses_adaptive_budget(monkeypatch, tmp_path):
    asset = SimpleNamespace(
        local_path="/fake/video.mp4",
        metadata=SimpleNamespace(duration_sec=10.0),
    )
    strategy = FrameStrategy(type="scene", params={
        "max_frames": 64,
        "min_frames": 16,
        "adaptive_by_duration": True,
        "scene_threshold": 0.25,
    })

    monkeypatch.setattr(frame_sampler, "_run", lambda cmd: None)
    monkeypatch.setattr(
        frame_sampler.glob,
        "glob",
        lambda pattern: [str(tmp_path / f"f_{i:06d}.jpg") for i in range(100)],
    )

    frames = frame_sampler.sample_frames(asset, str(tmp_path), strategy)

    assert len(frames.items) == 16


def test_caption_frames_includes_duration_and_timestamps(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            message = SimpleNamespace(content='[{"frame_id":"f_0001","caption":"first"},{"frame_id":"f_0002","caption":"second"}]')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(vision_caption, "make_client", lambda base_url: FakeClient())
    monkeypatch.setattr(vision_caption, "img_to_data_url", lambda path, max_w=256, max_h=144: f"data://{path}")

    frames = FrameSet(
        items=[
            FrameItem(id="f_0001", ts=12.0, path="/tmp/a.jpg"),
            FrameItem(id="f_0002", ts=24.5, path="/tmp/b.jpg"),
        ],
        strategy=FrameStrategy(type="scene", params={"scene_threshold": 0.25}),
    )

    result = vision_caption.caption_frames(
        frames,
        model_name="qwen3.5-9b",
        base_url="http://localhost:8000/v1",
        batch_size=2,
        video_duration_sec=120.0,
    )

    prompt = captured["content"][0]["text"]
    assert "源视频总时长约 120.0s" in prompt
    assert "f_0001=12.0s" in prompt
    assert "f_0002=24.5s" in prompt
    assert result.items[0].caption == "first"
    assert result.items[1].caption == "second"


def test_caption_frame_wraps_single_frame(monkeypatch):
    captured = {}

    def fake_caption_frames(frames, *args, **kwargs):
        captured["frames"] = frames
        captured["kwargs"] = kwargs
        frames.items[0].caption = "single caption"
        return frames

    monkeypatch.setattr(vision_caption, "caption_frames", fake_caption_frames)

    caption = vision_caption.caption_frame(
        "/tmp/frame.jpg",
        model_name="qwen3.5-9b",
        base_url="http://localhost:8000/v1",
        ts=3.5,
    )

    assert caption == "single caption"
    assert captured["frames"].items[0].path == "/tmp/frame.jpg"
    assert captured["frames"].items[0].ts == 3.5
    assert captured["kwargs"]["max_frames"] == 1
