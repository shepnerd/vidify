from types import SimpleNamespace

from agent.core.schemas import ASRSegment, Transcript, VideoMetadata, FrameSet, FrameStrategy
from agent.extensions.skills.timeline_builder import (
    _build_transcript_blocks,
    _heuristic_timeline_from_blocks,
    _parse_label_response,
    _parse_timeline_response,
    build_timeline,
)


def test_parse_timeline_response_from_fenced_json():
    text = """```json
    {
      "chapters": [{"start": 0, "end": 12.5, "title": "Intro", "summary": "Opening setup"}],
      "events": [{"start": 4, "end": 5, "text": "Logo slide", "evidence": {"frame_ids": ["f_1"]}}]
    }
    ```"""

    result = _parse_timeline_response(text)

    assert result == {
        "chapters": [{"start": 0.0, "end": 12.5, "title": "Intro", "summary": "Opening setup"}],
        "events": [{
            "start": 4.0,
            "end": 5.0,
            "text": "Logo slide",
            "evidence": {"asr_segment_ids": [], "frame_ids": ["f_1"]},
        }],
    }


def test_parse_timeline_response_from_pythonish_dict():
    text = (
        "Here is the timeline:\n"
        "{'chapters': [{'start': '0', 'end': '60', 'title': 'Opening', 'summary': 'Host intro'}], "
        "'events': [{'start': '15', 'end': '18', 'text': 'Two speakers on set', 'evidence': None}]}"
    )

    result = _parse_timeline_response(text)

    assert result == {
        "chapters": [{"start": 0.0, "end": 60.0, "title": "Opening", "summary": "Host intro"}],
        "events": [{
            "start": 15.0,
            "end": 18.0,
            "text": "Two speakers on set",
            "evidence": {"asr_segment_ids": [], "frame_ids": []},
        }],
    }


def test_build_transcript_blocks_groups_consecutive_segments():
    transcript = Transcript(segments=[
        ASRSegment(id="s0", start=0.0, end=30.0, text="Intro to AGI definitions and framing."),
        ASRSegment(id="s1", start=30.0, end=60.0, text="The guest explains how DeepMind thinks about AGI."),
        ASRSegment(id="s2", start=75.0, end=110.0, text="Discussion shifts to timing and probability estimates."),
        ASRSegment(id="s3", start=110.0, end=145.0, text="They compare forecasts and historical expectations."),
    ])

    blocks = _build_transcript_blocks(transcript, target_block_sec=90.0, min_block_sec=40.0)

    assert len(blocks) == 2
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 60.0
    assert blocks[0]["asr_segment_ids"] == ["s0", "s1"]
    assert blocks[1]["start"] == 75.0
    assert blocks[1]["end"] == 145.0
    assert blocks[1]["asr_segment_ids"] == ["s2", "s3"]


def test_heuristic_timeline_from_blocks_returns_non_empty_timeline():
    blocks = [
        {
            "id": "block_000",
            "start": 0.0,
            "end": 120.0,
            "title_hint": "intro / agi / definitions",
            "summary_hint": "Intro section on AGI definitions and framing.",
            "keywords": ["agi", "definitions", "framing"],
            "word_count": 120,
            "asr_segment_ids": ["s0", "s1"],
        },
        {
            "id": "block_001",
            "start": 120.0,
            "end": 260.0,
            "title_hint": "timing / forecasts / progress",
            "summary_hint": "Forecasts, timing, and progress discussion.",
            "keywords": ["timing", "forecasts", "progress"],
            "word_count": 180,
            "asr_segment_ids": ["s2", "s3"],
        },
        {
            "id": "block_002",
            "start": 260.0,
            "end": 420.0,
            "title_hint": "safety / governance / economics",
            "summary_hint": "Safety, governance, and economic implications.",
            "keywords": ["safety", "governance", "economics"],
            "word_count": 210,
            "asr_segment_ids": ["s4", "s5"],
        },
    ]
    frames = FrameSet(items=[], strategy=FrameStrategy(type="skipped", params={}))

    timeline = _heuristic_timeline_from_blocks(
        blocks,
        frames,
        {"target_chapters": 2, "target_events": 3},
    )

    assert len(timeline["chapters"]) == 2
    assert len(timeline["events"]) == 3
    assert timeline["events"][0]["evidence"]["asr_segment_ids"]
    assert len(timeline["events"][0]["text"]) <= 180


def test_parse_label_response_reads_chapter_labels():
    labels = _parse_label_response(
        '{"chapters":[{"index":0,"title":"AGI Definitions","summary":"How the guest defines AGI."}]}'
    )

    assert labels == [{
        "index": 0,
        "title": "AGI Definitions",
        "summary": "How the guest defines AGI.",
    }]


def test_build_timeline_falls_back_to_heuristic_when_model_output_is_empty(monkeypatch):
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 3:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=(
                        '{"chapters":['
                        '{"index":0,"title":"Opening Framing","summary":"The opening introduction and setup."},'
                        '{"index":1,"title":"AGI And Forecasts","summary":"Definitions, capabilities, and timelines."}'
                        ']}'
                    )))]
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(
        "agent.extensions.skills.timeline_builder.make_client",
        lambda base_url: FakeClient(),
    )

    metadata = VideoMetadata(duration_sec=600.0, fps=25.0, width=640, height=360, has_audio=True)
    transcript = Transcript(segments=[
        ASRSegment(id="s0", start=0.0, end=80.0, text="Opening and guest introduction."),
        ASRSegment(id="s1", start=80.0, end=180.0, text="Discussion of AGI definitions and capabilities."),
        ASRSegment(id="s2", start=180.0, end=320.0, text="Forecasts, risks, and scaling laws."),
        ASRSegment(id="s3", start=320.0, end=520.0, text="Governance, economics, and long-term impact."),
    ])
    frames = FrameSet(items=[], strategy=FrameStrategy(type="skipped", params={}))

    timeline = build_timeline(
        metadata,
        transcript,
        frames,
        model_name="qwen3.5",
        base_url="http://localhost:8000/v1",
    )

    assert timeline["chapters"]
    assert timeline["events"]
    assert timeline["chapters"][0]["title"] == "Opening Framing"
