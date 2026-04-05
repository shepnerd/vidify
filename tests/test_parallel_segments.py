# tests/test_parallel_segments.py
"""Tests for parallel segment processing.

Covers: segmentation logic, merge functions, segment worker, parallel executor,
workflow integration, and the pluggable segmentor interface.

Self-contained — no vLLM, GPU, or external services required.
"""
import os
import time
import tempfile
import threading
import pytest
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from agent.core.schemas import (
    FrameSet, FrameItem, FrameStrategy, Transcript, ASRSegment,
    VideoMetadata, VideoAsset, VideoSource,
)


# ---------------------------------------------------------------------------
# 1. Segmentation logic
# ---------------------------------------------------------------------------

class TestSplitVideoIntoSegments:
    def test_basic_split(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            segs = split_video_into_segments(900.0, tmp, segment_duration=300)
            assert len(segs) == 3
            assert segs[0].start_sec == 0.0
            assert segs[0].end_sec == 300.0
            assert segs[1].start_sec == 300.0
            assert segs[1].end_sec == 600.0
            assert segs[2].start_sec == 600.0
            assert segs[2].end_sec == 900.0

    def test_short_video_single_segment(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            segs = split_video_into_segments(120.0, tmp, segment_duration=300)
            assert len(segs) == 1
            assert segs[0].start_sec == 0.0
            assert segs[0].end_sec == 120.0

    def test_tiny_tail_merged(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            # 620s / 300s = [0-300, 300-600, 600-620] → tail 20s < 30s min → merge
            segs = split_video_into_segments(620.0, tmp, segment_duration=300,
                                              min_segment_duration=30)
            assert len(segs) == 2
            assert segs[-1].end_sec == 620.0

    def test_tail_not_merged_when_long_enough(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            # 650s / 300s = [0-300, 300-600, 600-650] → tail 50s > 30s → keep
            segs = split_video_into_segments(650.0, tmp, segment_duration=300,
                                              min_segment_duration=30)
            assert len(segs) == 3
            assert segs[-1].end_sec == 650.0

    def test_zero_duration(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            segs = split_video_into_segments(0.0, tmp, segment_duration=300)
            assert segs == []

    def test_cache_dirs_created(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            segs = split_video_into_segments(600.0, tmp, segment_duration=300)
            for seg in segs:
                assert os.path.isdir(seg.cache_dir)

    def test_segment_indices_sequential(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            segs = split_video_into_segments(1500.0, tmp, segment_duration=300)
            indices = [s.index for s in segs]
            assert indices == list(range(len(segs)))


# ---------------------------------------------------------------------------
# 2. Segmentor interface
# ---------------------------------------------------------------------------

class TestSegmentorInterface:
    def test_get_default_segmentor(self):
        from agent.core.segment import get_segmentor, DurationSegmentor
        seg = get_segmentor("duration", segment_duration=300)
        assert isinstance(seg, DurationSegmentor)

    def test_unknown_segmentor_raises(self):
        from agent.core.segment import get_segmentor
        with pytest.raises(ValueError, match="Unknown segmentor"):
            get_segmentor("nonexistent")

    def test_register_custom_segmentor(self):
        from agent.core.segment import (
            BaseSegmentor, register_segmentor, get_segmentor, VideoSegment,
        )

        class FixedSegmentor(BaseSegmentor):
            """Always returns exactly 2 segments, splitting at midpoint."""
            def segment(self, video_path, duration_sec, base_cache_dir):
                mid = duration_sec / 2
                return [
                    self._make_segment(0, 0, mid, base_cache_dir),
                    self._make_segment(1, mid, duration_sec, base_cache_dir),
                ]

        register_segmentor("fixed_test", FixedSegmentor)
        seg = get_segmentor("fixed_test")
        assert isinstance(seg, FixedSegmentor)

        with tempfile.TemporaryDirectory() as tmp:
            result = seg.segment("/fake.mp4", 100.0, tmp)
            assert len(result) == 2
            assert result[0].end_sec == 50.0
            assert result[1].start_sec == 50.0

    def test_register_non_subclass_raises(self):
        from agent.core.segment import register_segmentor
        with pytest.raises(TypeError):
            register_segmentor("bad", dict)

    def test_convenience_wrapper_uses_registry(self):
        from agent.core.segment import split_video_into_segments
        with tempfile.TemporaryDirectory() as tmp:
            segs = split_video_into_segments(
                600.0, tmp, segment_duration=200, segmentor_name="duration",
            )
            assert len(segs) == 3


# ---------------------------------------------------------------------------
# 3. Merge functions
# ---------------------------------------------------------------------------

class TestMergeFunctions:
    def _make_frameset(self, start_ts, count, prefix="seg"):
        items = [
            FrameItem(id=f"{prefix}_{i}", ts=start_ts + i * 2.0, path=f"/tmp/{prefix}_{i}.jpg")
            for i in range(count)
        ]
        return FrameSet(items=items, strategy=FrameStrategy(type="scene", params={}))

    def test_merge_framesets_sorted_and_reindexed(self):
        from agent.core.segment import merge_framesets, VideoSegment
        fs1 = self._make_frameset(0.0, 3, "a")
        fs2 = self._make_frameset(10.0, 3, "b")
        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=10, end_sec=20, cache_dir="/tmp/s1"),
        ]
        merged = merge_framesets([fs1, fs2], segs)
        assert len(merged.items) == 6
        # Check sorted by timestamp
        timestamps = [f.ts for f in merged.items]
        assert timestamps == sorted(timestamps)
        # Check re-indexed
        ids = [f.id for f in merged.items]
        assert ids == [f"f_{i:06d}" for i in range(6)]

    def test_merge_framesets_empty(self):
        from agent.core.segment import merge_framesets, VideoSegment
        empty = FrameSet(items=[], strategy=FrameStrategy(type="scene", params={}))
        segs = [VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir="/tmp")]
        merged = merge_framesets([empty], segs)
        assert len(merged.items) == 0

    def test_merge_ocr_results(self):
        from agent.core.segment import merge_ocr_results
        r1 = {"frame_a.jpg": ["hello"]}
        r2 = {"frame_b.jpg": ["world"]}
        merged = merge_ocr_results([r1, r2])
        assert "frame_a.jpg" in merged
        assert "frame_b.jpg" in merged

    def test_merge_ocr_handles_empty(self):
        from agent.core.segment import merge_ocr_results
        merged = merge_ocr_results([{}, {}, {}])
        assert merged == {}

    def test_merge_object_results(self):
        from agent.core.segment import merge_object_results
        r1 = {"frame_a.jpg": [{"class": "person"}]}
        r2 = {"frame_b.jpg": [{"class": "car"}]}
        merged = merge_object_results([r1, r2])
        assert len(merged) == 2

    def test_merge_emotion_adjusts_audio_timestamps(self):
        from agent.core.segment import merge_emotion_results, VideoSegment
        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=300, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=300, end_sec=600, cache_dir="/tmp/s1"),
        ]
        r1 = {"audio_emotions": [{"start": 10, "end": 20, "emotion": "happy"}]}
        r2 = {"audio_emotions": [{"start": 5, "end": 15, "emotion": "sad"}]}
        merged = merge_emotion_results([r1, r2], segs)
        audio = merged["audio_emotions"]
        assert len(audio) == 2
        # First segment: offset 0 → unchanged
        assert audio[0]["start"] == 10
        assert audio[0]["end"] == 20
        # Second segment: offset 300
        assert audio[1]["start"] == 305
        assert audio[1]["end"] == 315

    def test_merge_emotion_frame_level(self):
        from agent.core.segment import merge_emotion_results, VideoSegment
        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=300, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=300, end_sec=600, cache_dir="/tmp/s1"),
        ]
        r1 = {"frame_emotions": {"a.jpg": {"happy": 0.9}}}
        r2 = {"frame_emotions": {"b.jpg": {"sad": 0.8}}}
        merged = merge_emotion_results([r1, r2], segs)
        assert "a.jpg" in merged["frame_emotions"]
        assert "b.jpg" in merged["frame_emotions"]

    def test_merge_segment_results_full(self):
        from agent.core.segment import merge_segment_results, VideoSegment
        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=10, end_sec=20, cache_dir="/tmp/s1"),
        ]
        outputs = [
            {
                "frames": FrameSet(
                    items=[FrameItem(id="f_0", ts=1.0, path="/a.jpg")],
                    strategy=FrameStrategy(type="scene", params={})
                ).model_dump(),
                "ocr": {"a.jpg": ["text1"]},
                "objects": {"a.jpg": [{"class": "cat"}]},
                "emotions": {},
            },
            {
                "frames": FrameSet(
                    items=[FrameItem(id="f_0", ts=11.0, path="/b.jpg")],
                    strategy=FrameStrategy(type="scene", params={})
                ).model_dump(),
                "ocr": {"b.jpg": ["text2"]},
                "objects": {},
                "emotions": {},
            },
        ]
        merged = merge_segment_results(outputs, segs)
        assert len(merged["frames"].items) == 2
        assert "a.jpg" in merged["ocr"]
        assert "b.jpg" in merged["ocr"]


# ---------------------------------------------------------------------------
# 4. run_segments_parallel
# ---------------------------------------------------------------------------

class TestRunSegmentsParallel:
    def test_basic_parallel_execution(self):
        from agent.core.parallel import run_segments_parallel
        from agent.core.segment import VideoSegment

        def worker(segment, value=0):
            return {"index": segment.index, "value": value}

        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=10, end_sec=20, cache_dir="/tmp/s1"),
        ]
        results = run_segments_parallel(segs, worker, {"value": 42}, max_workers=2)
        assert len(results) == 2
        assert results[0]["index"] == 0
        assert results[0]["value"] == 42
        assert results[1]["index"] == 1

    def test_results_ordered_by_segment_index(self):
        from agent.core.parallel import run_segments_parallel
        from agent.core.segment import VideoSegment
        import time

        def slow_worker(segment):
            # Segment 0 finishes last to test ordering
            time.sleep(0.1 if segment.index == 0 else 0.01)
            return {"index": segment.index}

        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=10, end_sec=20, cache_dir="/tmp/s1"),
            VideoSegment(index=2, start_sec=20, end_sec=30, cache_dir="/tmp/s2"),
        ]
        results = run_segments_parallel(segs, slow_worker, {}, max_workers=3)
        # Even though seg 0 finishes last, results should be ordered
        assert [r["index"] for r in results] == [0, 1, 2]

    def test_failed_segment_returns_empty_dict(self):
        from agent.core.parallel import run_segments_parallel
        from agent.core.segment import VideoSegment

        def failing_worker(segment):
            if segment.index == 1:
                raise RuntimeError("boom")
            return {"ok": True}

        segs = [
            VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir="/tmp/s0"),
            VideoSegment(index=1, start_sec=10, end_sec=20, cache_dir="/tmp/s1"),
        ]
        results = run_segments_parallel(segs, failing_worker, {}, max_workers=2)
        assert results[0] == {"ok": True}
        assert results[1] == {}  # failed segment

    def test_empty_segments_list(self):
        from agent.core.parallel import run_segments_parallel
        results = run_segments_parallel([], lambda s: {}, {})
        assert results == []

    def test_actual_concurrency(self):
        """Verify segments actually run concurrently, not sequentially."""
        from agent.core.parallel import run_segments_parallel
        from agent.core.segment import VideoSegment

        def slow_worker(segment):
            time.sleep(0.2)
            return {"done": True}

        segs = [
            VideoSegment(index=i, start_sec=i * 10, end_sec=(i + 1) * 10, cache_dir=f"/tmp/s{i}")
            for i in range(4)
        ]
        start = time.time()
        results = run_segments_parallel(segs, slow_worker, {}, max_workers=4)
        elapsed = time.time() - start
        # 4 segments × 0.2s = 0.8s sequential, but should be ~0.2s parallel
        assert elapsed < 0.6, f"Expected parallel execution but took {elapsed:.2f}s"
        assert all(r["done"] for r in results)


# ---------------------------------------------------------------------------
# 5. Segment worker
# ---------------------------------------------------------------------------

class TestSegmentWorker:
    def test_process_segment_no_frames(self):
        """Empty frame sampling should return empty results gracefully."""
        from agent.core.segment_worker import process_segment
        from agent.core.segment import VideoSegment

        asset = MagicMock()
        asset.local_path = "/fake/video.mp4"
        seg = VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir=tempfile.mkdtemp())
        strategy = FrameStrategy(type="scene", params={"max_frames": 10})

        with patch("agent.core.segment_worker.sample_frames") as mock_sample:
            mock_sample.return_value = FrameSet(items=[], strategy=strategy)
            result = process_segment(segment=seg, asset=asset, strategy=strategy)

        assert result["frames"]["items"] == []
        assert result["ocr"] == {}
        assert result["objects"] == {}
        assert result["emotions"] == {}

    def test_process_segment_with_frames_no_captioning(self):
        """Frames present but captioning disabled — only analysis skills run."""
        from agent.core.segment_worker import process_segment
        from agent.core.segment import VideoSegment

        asset = MagicMock()
        seg = VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir=tempfile.mkdtemp())
        strategy = FrameStrategy(type="scene", params={"max_frames": 10})

        fake_frames = FrameSet(
            items=[FrameItem(id="f_0", ts=1.0, path="/tmp/fake.jpg")],
            strategy=strategy,
        )

        with patch("agent.core.segment_worker.sample_frames", return_value=fake_frames), \
             patch("agent.core.segment_worker.extract_text_from_video_frames", return_value={"a": "b"}), \
             patch("agent.core.segment_worker.analyze_emotions", return_value={"emo": "happy"}):
            result = process_segment(
                segment=seg, asset=asset, strategy=strategy,
                need_captioning=False, audio_path="/tmp/audio.wav",
            )

        assert len(result["frames"]["items"]) == 1
        assert result["ocr"] == {"a": "b"}
        assert result["emotions"] == {"emo": "happy"}

    def test_process_segment_captioning_parallel_with_analysis(self):
        """Captioning and analysis should run concurrently (both in parallel pool)."""
        from agent.core.segment_worker import process_segment
        from agent.core.segment import VideoSegment

        execution_order = []
        lock = threading.Lock()

        asset = MagicMock()
        seg = VideoSegment(index=0, start_sec=0, end_sec=10, cache_dir=tempfile.mkdtemp())
        strategy = FrameStrategy(type="scene", params={"max_frames": 10})

        fake_frames = FrameSet(
            items=[FrameItem(id="f_0", ts=1.0, path="/tmp/fake.jpg")],
            strategy=strategy,
        )

        def mock_caption(frames_dump, *args, **kwargs):
            with lock:
                execution_order.append(("caption_start", time.time()))
            time.sleep(0.15)
            frames = FrameSet(**frames_dump)
            for item in frames.items:
                item.caption = "test caption"
            with lock:
                execution_order.append(("caption_end", time.time()))
            return frames.model_dump()

        def mock_ocr(frame_paths):
            with lock:
                execution_order.append(("ocr_start", time.time()))
            time.sleep(0.15)
            with lock:
                execution_order.append(("ocr_end", time.time()))
            return {"ocr_result": True}

        with patch("agent.core.segment_worker.sample_frames", return_value=fake_frames), \
             patch("agent.core.segment_worker._run_captioning", side_effect=mock_caption), \
             patch("agent.core.segment_worker.extract_text_from_video_frames", side_effect=mock_ocr), \
             patch("agent.core.segment_worker.detect_objects_in_video_frames", None):
            result = process_segment(
                segment=seg, asset=asset, strategy=strategy,
                need_captioning=True, llm_model="test", llm_base_url="http://x",
            )

        # Both should have run
        assert result["ocr"] == {"ocr_result": True}
        # Check that caption result was used
        assert result["frames"]["items"][0]["caption"] == "test caption"

        # Verify parallelism: caption and OCR started at nearly the same time
        starts = {name: t for name, t in execution_order if name.endswith("_start")}
        if "caption_start" in starts and "ocr_start" in starts:
            gap = abs(starts["caption_start"] - starts["ocr_start"])
            assert gap < 0.1, f"Caption and OCR should start near-simultaneously, gap={gap:.3f}s"


# ---------------------------------------------------------------------------
# 6. Workflow integration: _extract_transcript helper
# ---------------------------------------------------------------------------

class TestExtractTranscript:
    def test_returns_subtitle_transcript(self):
        from agent.extensions.workflows.detailed import _extract_transcript

        asset = MagicMock()
        asset.subtitle_tracks = [MagicMock()]
        asset.cache_dir = tempfile.mkdtemp()
        meta = MagicMock()
        meta.has_audio = False

        fake_transcript = Transcript(
            segments=[ASRSegment(id="s0", start=0, end=5, text="hello")],
            language="en",
        )

        with patch("agent.extensions.workflows.detailed.load_best_subtitle",
                    return_value=fake_transcript):
            transcript, audio_path = _extract_transcript(asset, meta, "small")

        assert transcript.language == "en"
        assert len(transcript.segments) == 1
        assert audio_path is None

    def test_falls_back_to_asr(self):
        from agent.extensions.workflows.detailed import _extract_transcript

        asset = MagicMock()
        asset.subtitle_tracks = []
        asset.cache_dir = tempfile.mkdtemp()
        meta = MagicMock()
        meta.has_audio = True

        fake_transcript = Transcript(
            segments=[ASRSegment(id="a0", start=0, end=10, text="world")],
            language="zh",
        )

        with patch("agent.extensions.workflows.detailed.extract_audio", return_value="/tmp/audio.wav"), \
             patch("agent.extensions.workflows.detailed.transcribe", return_value=fake_transcript):
            transcript, audio_path = _extract_transcript(asset, meta, "small")

        assert transcript.language == "zh"
        assert audio_path == "/tmp/audio.wav"

    def test_no_audio_no_subtitles(self):
        from agent.extensions.workflows.detailed import _extract_transcript

        asset = MagicMock()
        asset.subtitle_tracks = []
        asset.cache_dir = tempfile.mkdtemp()
        meta = MagicMock()
        meta.has_audio = False

        transcript, audio_path = _extract_transcript(asset, meta, "small")
        assert len(transcript.segments) == 0
        assert audio_path is None


# ---------------------------------------------------------------------------
# 7. Workflow integration: _run_sequential captioning ∥ analysis
# ---------------------------------------------------------------------------

class TestRunSequentialParallelism:
    def test_captioning_and_analysis_run_in_parallel(self):
        """In _run_sequential, captioning should overlap with OCR/detection/emotion."""
        from agent.extensions.workflows.detailed import _run_sequential
        from agent.core.schemas import ContentSufficiency

        asset = MagicMock()
        asset.cache_dir = tempfile.mkdtemp()

        frames = FrameSet(
            items=[FrameItem(id="f_0", ts=1.0, path="/tmp/f.jpg")],
            strategy=FrameStrategy(type="scene", params={}),
        )
        sufficiency = ContentSufficiency(
            asr_coverage_ratio=0.1, transcript_word_count=10,
            has_subtitles=False, has_content_metadata=False,
            is_sufficient=False, reason="low coverage",
        )

        timing = {}

        def mock_caption(*args, **kwargs):
            timing["caption_start"] = time.time()
            time.sleep(0.15)
            timing["caption_end"] = time.time()
            return frames  # return original frames as "captioned"

        def mock_ocr(paths):
            timing["ocr_start"] = time.time()
            time.sleep(0.15)
            timing["ocr_end"] = time.time()
            return {"ocr": True}

        with patch("agent.extensions.workflows.detailed.caption_frames", side_effect=mock_caption), \
             patch("agent.extensions.workflows.detailed.supports_video", return_value=False), \
             patch("agent.extensions.workflows.detailed.extract_text_from_video_frames", side_effect=mock_ocr), \
             patch("agent.extensions.workflows.detailed.detect_objects_in_video_frames", None), \
             patch("agent.extensions.workflows.detailed.analyze_emotions", return_value={}):
            result_frames, ocr, obj, emo = _run_sequential(
                asset=asset, frames=frames, sufficiency=sufficiency,
                llm_model="test", llm_base_url="http://x",
                direct_model=False, model_path=None, tokenizer_path=None,
                audio_path=None, wf_cfg={},
            )

        assert ocr == {"ocr": True}
        # Check captioning and OCR overlapped
        if "caption_start" in timing and "ocr_start" in timing:
            gap = abs(timing["caption_start"] - timing["ocr_start"])
            assert gap < 0.1, f"Expected parallel but gap={gap:.3f}s"

    def test_sufficient_transcript_skips_captioning(self):
        """When transcript is sufficient, captioning should not run."""
        from agent.extensions.workflows.detailed import _run_sequential
        from agent.core.schemas import ContentSufficiency

        asset = MagicMock()
        asset.cache_dir = tempfile.mkdtemp()

        frames = FrameSet(
            items=[FrameItem(id="f_0", ts=1.0, path="/tmp/f.jpg")],
            strategy=FrameStrategy(type="scene", params={}),
        )
        sufficiency = ContentSufficiency(
            asr_coverage_ratio=0.8, transcript_word_count=200,
            has_subtitles=True, has_content_metadata=True,
            is_sufficient=True, reason="good coverage",
        )

        caption_called = False

        def mock_caption(*args, **kwargs):
            nonlocal caption_called
            caption_called = True

        with patch("agent.extensions.workflows.detailed.caption_frames", side_effect=mock_caption), \
             patch("agent.extensions.workflows.detailed.extract_text_from_video_frames", return_value={}), \
             patch("agent.extensions.workflows.detailed.detect_objects_in_video_frames", None), \
             patch("agent.extensions.workflows.detailed.analyze_emotions", return_value={}):
            _run_sequential(
                asset=asset, frames=frames, sufficiency=sufficiency,
                llm_model="test", llm_base_url="http://x",
                direct_model=False, model_path=None, tokenizer_path=None,
                audio_path=None, wf_cfg={},
            )

        assert not caption_called


# ---------------------------------------------------------------------------
# 8. Frame sampler time range
# ---------------------------------------------------------------------------

class TestFrameSamplerTimeRange:
    def test_time_flags_built_correctly(self):
        """Verify FFmpeg command includes -ss and -to for segment processing."""
        from agent.extensions.skills.frame_sampler import sample_frames

        asset = MagicMock()
        asset.local_path = "/fake/video.mp4"
        asset.metadata = MagicMock()
        asset.metadata.duration_sec = 600.0

        captured_cmd = []
        original_run = None

        def capture_run(cmd):
            captured_cmd.extend(cmd)
            # Create a fake frame file so glob finds something
            pass

        strategy = FrameStrategy(type="fps", params={"fps": 1.0, "max_frames": 10})

        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent.extensions.skills.frame_sampler._run", side_effect=capture_run):
            try:
                sample_frames(asset, tmp, strategy, start_sec=300.0, end_sec=600.0)
            except Exception:
                pass  # glob may find no files, that's ok

        # Check that -ss and -to are in the command
        assert "-ss" in captured_cmd, f"Expected -ss in cmd: {captured_cmd}"
        ss_idx = captured_cmd.index("-ss")
        assert captured_cmd[ss_idx + 1] == "300.0"

        assert "-to" in captured_cmd, f"Expected -to in cmd: {captured_cmd}"
        to_idx = captured_cmd.index("-to")
        assert captured_cmd[to_idx + 1] == "300.0"  # relative: 600-300=300

    def test_no_time_flags_when_none(self):
        """Without start_sec/end_sec, no -ss/-to flags."""
        from agent.extensions.skills.frame_sampler import sample_frames

        asset = MagicMock()
        asset.local_path = "/fake/video.mp4"
        asset.metadata = MagicMock()
        asset.metadata.duration_sec = 100.0

        captured_cmd = []

        def capture_run(cmd):
            captured_cmd.extend(cmd)

        strategy = FrameStrategy(type="fps", params={"fps": 1.0, "max_frames": 10})

        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent.extensions.skills.frame_sampler._run", side_effect=capture_run):
            try:
                sample_frames(asset, tmp, strategy)
            except Exception:
                pass

        assert "-ss" not in captured_cmd
        assert "-to" not in captured_cmd


# ---------------------------------------------------------------------------
# 9. Workflow-level audio/ASR ∥ frame sampling
# ---------------------------------------------------------------------------

class TestWorkflowAudioFrameParallelism:
    def test_audio_and_frames_overlap(self):
        """Audio extraction and frame sampling should run concurrently after probe."""
        timing = {}

        def mock_extract_transcript(asset, meta, whisper_model):
            timing["transcript_start"] = time.time()
            time.sleep(0.15)
            timing["transcript_end"] = time.time()
            return Transcript(segments=[], language=None), None

        def mock_sample_frames(asset, frames_dir, strategy, **kwargs):
            timing["frames_start"] = time.time()
            time.sleep(0.15)
            timing["frames_end"] = time.time()
            return FrameSet(items=[], strategy=strategy)

        asset = MagicMock()
        asset.id = "test"
        asset.local_path = "/fake.mp4"
        asset.cache_dir = tempfile.mkdtemp()
        asset.content_metadata = None
        asset.subtitle_tracks = []
        asset.source = MagicMock()
        asset.source.model_dump.return_value = {"type": "local", "uri": "/fake.mp4"}

        meta = MagicMock()
        meta.duration_sec = 60.0
        meta.has_audio = True
        meta.content = None
        meta.model_dump.return_value = {"duration_sec": 60, "fps": 30, "width": 1920, "height": 1080, "has_audio": True}

        sufficiency = MagicMock()
        sufficiency.is_sufficient = True
        sufficiency.model_dump.return_value = {}

        with patch("agent.extensions.workflows.detailed.probe_video", return_value=meta), \
             patch("agent.extensions.workflows.detailed._extract_transcript", side_effect=mock_extract_transcript), \
             patch("agent.extensions.workflows.detailed.sample_frames", side_effect=mock_sample_frames), \
             patch("agent.extensions.workflows.detailed.assess_sufficiency", return_value=sufficiency), \
             patch("agent.extensions.workflows.detailed.extract_text_from_video_frames", return_value={}), \
             patch("agent.extensions.workflows.detailed.detect_objects_in_video_frames", None), \
             patch("agent.extensions.workflows.detailed.analyze_emotions", return_value={}), \
             patch("agent.extensions.workflows.detailed.build_timeline", return_value={}), \
             patch("agent.extensions.workflows.detailed.translate_asr_results", return_value=[]), \
             patch("agent.extensions.workflows.detailed.save_analysis"):
            from agent.extensions.workflows.detailed import wf_detailed
            wf_detailed(asset)

        # Verify both started at roughly the same time
        if "transcript_start" in timing and "frames_start" in timing:
            gap = abs(timing["transcript_start"] - timing["frames_start"])
            assert gap < 0.1, f"Expected parallel but gap={gap:.3f}s"
