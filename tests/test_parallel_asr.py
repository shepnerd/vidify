import json
from pathlib import Path

from agent.core.schemas import ASRSegment, Transcript
from agent.extensions.skills import asr


def test_plan_audio_ranges_merges_short_tail():
    ranges = asr._plan_audio_ranges(
        duration_sec=620.0,
        segment_duration_sec=300.0,
        min_segment_duration_sec=30.0,
    )

    assert ranges == [
        (0, 0.0, 300.0),
        (1, 300.0, 620.0),
    ]


def test_merge_transcripts_reindexes_and_sorts():
    t1 = Transcript(segments=[
        ASRSegment(id="seg_x", start=40.0, end=50.0, text="later"),
        ASRSegment(id="seg_y", start=0.0, end=10.0, text="first"),
    ], language="en")
    t2 = Transcript(segments=[
        ASRSegment(id="seg_z", start=20.0, end=30.0, text="middle"),
    ], language=None)

    merged = asr._merge_transcripts([t1, t2])

    assert [seg.text for seg in merged.segments] == ["first", "middle", "later"]
    assert [seg.id for seg in merged.segments] == ["seg_000000", "seg_000001", "seg_000002"]
    assert merged.language == "en"


def test_resolve_backend_prefers_local_faster_whisper(monkeypatch, tmp_path):
    fw_dir = tmp_path / "faster-whisper-small"
    fw_dir.mkdir()
    (fw_dir / "model.bin").write_bytes(b"x")

    monkeypatch.setattr(asr, "FasterWhisperModel", object())
    monkeypatch.setattr(asr, "torch", None)
    monkeypatch.setattr(asr, "WhisperProcessor", None)
    monkeypatch.setattr(asr, "WhisperForConditionalGeneration", None)

    def fake_get_model_path(name: str):
        if name == "faster-whisper-small":
            return str(fw_dir)
        return str(Path(tmp_path) / name)

    import agent.config as config
    monkeypatch.setattr(config, "get_model_path", fake_get_model_path)

    backend, model_id = asr._resolve_backend("small")

    assert backend == "faster-whisper"
    assert model_id == str(fw_dir)


def test_transcribe_uses_parallel_path_when_enabled(monkeypatch, tmp_path):
    out_json = tmp_path / "asr.json"
    observed = {}

    def fake_parallel(audio_path, model_size, worker_devices, ranges):
        observed["parallel"] = {
            "audio_path": audio_path,
            "model_size": model_size,
            "worker_devices": worker_devices,
            "ranges": ranges,
        }
        return Transcript(segments=[
            ASRSegment(id="seg_000000", start=0.0, end=30.0, text="hello")
        ], language="en")

    def fail_sequential(*args, **kwargs):
        raise AssertionError("sequential path should not be used")

    monkeypatch.setattr(asr, "_get_audio_duration", lambda _: 900.0)
    monkeypatch.setattr(asr, "_run_parallel_transcribe", fake_parallel)
    monkeypatch.setattr(asr, "_transcribe_sequential", fail_sequential)

    result = asr.transcribe(
        "fake.wav",
        str(out_json),
        model_size="small",
        parallel=True,
        max_workers=2,
        devices=["cpu", "cpu"],
        segment_duration_sec=300.0,
        min_audio_duration_sec=300.0,
        min_segment_duration_sec=30.0,
    )

    assert result.language == "en"
    assert observed["parallel"]["worker_devices"] == ["cpu", "cpu"]
    assert observed["parallel"]["ranges"] == [
        (0, 0.0, 300.0),
        (1, 300.0, 600.0),
        (2, 600.0, 900.0),
    ]
    assert json.loads(out_json.read_text(encoding="utf-8"))["language"] == "en"


def test_transcribe_falls_back_to_sequential_for_short_audio(monkeypatch, tmp_path):
    out_json = tmp_path / "asr.json"
    observed = {"sequential": 0}

    monkeypatch.setattr(asr, "_get_audio_duration", lambda _: 120.0)
    monkeypatch.setattr(
        asr,
        "_run_parallel_transcribe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parallel path should not run")),
    )

    def fake_sequential(audio_path, model_size="small", device=None):
        observed["sequential"] += 1
        return Transcript(segments=[
            ASRSegment(id="seg_000000", start=0.0, end=10.0, text="short clip")
        ], language="en")

    monkeypatch.setattr(asr, "_transcribe_sequential", fake_sequential)

    result = asr.transcribe(
        "fake.wav",
        str(out_json),
        parallel=True,
        max_workers=4,
        devices=["cpu", "cpu", "cpu", "cpu"],
        min_audio_duration_sec=300.0,
    )

    assert result.segments[0].text == "short clip"
    assert observed["sequential"] == 1


def test_transcribe_clip_task_extracts_clip_and_rebases_timestamps(monkeypatch):
    observed = {}

    def fake_extract(audio_path, start_sec, end_sec):
        observed["extract"] = (audio_path, start_sec, end_sec)
        return "/tmp/clip.wav"

    def fake_transcribe_window(audio_path, model_size="small", device=None,
                               start_offset_sec=0.0, duration_sec=None, compute_type=None):
        observed["window"] = {
            "audio_path": audio_path,
            "model_size": model_size,
            "device": device,
            "start_offset_sec": start_offset_sec,
            "duration_sec": duration_sec,
            "compute_type": compute_type,
        }
        return Transcript(segments=[
            ASRSegment(id="seg_local", start=1.5, end=4.0, text="clip text"),
        ], language="en")

    monkeypatch.setattr(asr, "_extract_audio_clip", fake_extract)
    monkeypatch.setattr(asr, "_transcribe_window", fake_transcribe_window)
    monkeypatch.setattr(asr.os, "unlink", lambda path: observed.setdefault("deleted", []).append(path))

    result = asr._transcribe_clip_task(
        "full.wav",
        "small",
        "cpu",
        (3, 120.0, 180.0),
        compute_type="int8",
    )

    assert observed["extract"] == ("full.wav", 120.0, 180.0)
    assert observed["window"] == {
        "audio_path": "/tmp/clip.wav",
        "model_size": "small",
        "device": "cpu",
        "start_offset_sec": 0.0,
        "duration_sec": None,
        "compute_type": "int8",
    }
    assert observed["deleted"] == ["/tmp/clip.wav"]
    assert result["index"] == 3
    transcript = Transcript.model_validate(result["transcript"])
    assert transcript.language == "en"
    assert transcript.segments[0].start == 121.5
    assert transcript.segments[0].end == 124.0
