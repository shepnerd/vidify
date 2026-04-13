import os
import subprocess
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from agent.core.schemas import Transcript, ASRSegment
from agent.extensions.utils.cache import write_json, exists_nonempty, read_json
from agent.core.retry import retry_with_backoff

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    WhisperProcessor = None
    WhisperForConditionalGeneration = None

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:
    FasterWhisperModel = None

import logging

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}


def resolve_whisper_model(model_size: str = "small") -> str:
    """Resolve a Whisper model identifier to a local path or HF repo id."""
    from agent.config import get_model_path

    if os.path.isdir(str(model_size)):
        return str(model_size)

    model_id = get_model_path(f"whisper-{model_size}")
    if os.path.isdir(model_id):
        return model_id

    faster_model_id = get_model_path(f"faster-whisper-{model_size}")
    if os.path.isdir(faster_model_id):
        return faster_model_id

    return f"openai/whisper-{model_size}"


def has_local_whisper_model(model_size: str = "small") -> bool:
    """Return True when the requested Whisper model exists under models/."""
    model_id = resolve_whisper_model(model_size)
    return os.path.isdir(model_id) and not str(model_id).startswith("openai/")


def _is_faster_whisper_dir(model_id: str) -> bool:
    return os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "model.bin"))


def _resolve_backend(model_size: str = "small") -> tuple[str, str]:
    model_id = resolve_whisper_model(model_size)
    if _is_faster_whisper_dir(model_id):
        if FasterWhisperModel is None:
            raise ImportError("faster-whisper is not installed")
        return "faster-whisper", model_id

    if torch is None or WhisperProcessor is None or WhisperForConditionalGeneration is None:
        from agent.config import get_model_path
        fallback_dir = get_model_path(f"faster-whisper-{model_size}")
        if _is_faster_whisper_dir(fallback_dir) and FasterWhisperModel is not None:
            return "faster-whisper", fallback_dir
        raise ImportError("ASR dependencies are not installed")

    return "transformers", model_id


def _detect_device():
    """Pick best available device: NPU > CUDA > CPU."""
    if torch is None:
        raise ImportError("torch is required for ASR")
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return torch.device("npu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import json
    return float(json.loads(result.stdout)["format"]["duration"])


def _normalize_faster_whisper_device(device: str = None) -> tuple[str, int | None]:
    device = (device or "cpu").strip()
    if device.startswith("cuda"):
        if ":" in device:
            return "cuda", int(device.split(":", 1)[1])
        return "cuda", 0
    if device.startswith("cpu"):
        return "cpu", 0
    if device.startswith("npu"):
        logger.warning("faster-whisper does not support %s; falling back to cpu", device)
        return "cpu", 0
    return device, 0


def _get_model(model_size: str = "small", device: str = None, compute_type: str = None):
    """Lazy-init Whisper model per (backend, model_size, device)."""
    backend, model_id = _resolve_backend(model_size)
    resolved_device = device
    if backend == "transformers":
        resolved_device = torch.device(device) if device else _detect_device()
    else:
        fw_device, fw_index = _normalize_faster_whisper_device(device)
        resolved_device = f"{fw_device}:{fw_index}" if fw_index is not None else fw_device

    cache_key = (backend, model_id, str(resolved_device), compute_type or "")
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    logger.info("Loading %s ASR model %s on %s", backend, model_id, resolved_device)
    if backend == "transformers":
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        dtype = torch.float16 if resolved_device.type in ("cuda", "npu") else torch.float32
        model = model.to(resolved_device).to(dtype)
        model.eval()
        _MODEL_CACHE[cache_key] = (backend, processor, model, resolved_device)
    else:
        fw_device, fw_index = _normalize_faster_whisper_device(device)
        fw_compute = compute_type or ("float16" if fw_device == "cuda" else "int8")
        model = FasterWhisperModel(model_id, device=fw_device, device_index=fw_index, compute_type=fw_compute)
        _MODEL_CACHE[cache_key] = (backend, None, model, resolved_device)

    return _MODEL_CACHE[cache_key]


def _load_audio(audio_path: str, sr: int = 16000,
                offset_sec: float = 0.0,
                duration_sec: float = None) -> np.ndarray:
    """Load audio via librosa (already a dependency for emotion_analysis)."""
    import librosa
    audio, _ = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
        offset=max(0.0, float(offset_sec or 0.0)),
        duration=None if duration_sec is None else max(0.0, float(duration_sec)),
    )
    return audio


def _extract_audio_clip(audio_path: str, start_sec: float, end_sec: float) -> str:
    """Materialize an audio shard for backend-agnostic parallel ASR workers."""
    duration_sec = max(0.0, float(end_sec) - float(start_sec))
    if duration_sec <= 0:
        raise ValueError("audio clip duration must be positive")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        clip_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{float(start_sec):.3f}",
                "-t",
                f"{duration_sec:.3f}",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                clip_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        try:
            os.unlink(clip_path)
        except OSError:
            pass
        raise

    return clip_path


def _chunk_audio(audio: np.ndarray, sr: int = 16000, chunk_sec: float = 30.0):
    """Split audio into chunks for Whisper's 30-second context window."""
    chunk_len = int(sr * chunk_sec)
    for start in range(0, len(audio), chunk_len):
        yield start / sr, audio[start:start + chunk_len]


def _detect_available_devices() -> list[str]:
    if torch is None:
        return ["cpu"]
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return [f"npu:{i}" for i in range(torch.npu.device_count())]
    except ImportError:
        pass

    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    return ["cpu"]


def _normalize_devices(devices) -> list[str]:
    if devices is None:
        return []
    if isinstance(devices, str):
        return [d.strip() for d in devices.split(",") if d.strip()]
    return [str(d).strip() for d in devices if str(d).strip()]


def _resolve_worker_devices(device: str = None, devices=None, max_workers: int = None) -> list[str]:
    requested = _normalize_devices(devices)
    if device:
        requested = [device]
    if not requested:
        requested = _detect_available_devices()

    if max_workers is None or max_workers < 1:
        max_workers = len(requested) or 1

    if len(requested) == 1 and requested[0] == "cpu":
        return ["cpu"] * max_workers

    return requested[:max_workers]


def _plan_audio_ranges(duration_sec: float,
                       segment_duration_sec: float = 300.0,
                       min_segment_duration_sec: float = 30.0) -> list[tuple[int, float, float]]:
    if duration_sec <= 0:
        return []

    segs = []
    start = 0.0
    index = 0
    segment_duration_sec = max(30.0, float(segment_duration_sec))
    min_segment_duration_sec = max(1.0, float(min_segment_duration_sec))

    while start < duration_sec:
        end = min(duration_sec, start + segment_duration_sec)
        segs.append((index, start, end))
        index += 1
        start = end

    if len(segs) >= 2:
        last_index, last_start, last_end = segs[-1]
        if (last_end - last_start) < min_segment_duration_sec:
            prev_index, prev_start, _ = segs[-2]
            segs[-2] = (prev_index, prev_start, last_end)
            segs.pop()

    return segs


def _transcribe_window(audio_path: str,
                       model_size: str = "small",
                       device: str = None,
                       start_offset_sec: float = 0.0,
                       duration_sec: float = None,
                       chunk_sec: float = 30.0,
                       compute_type: str = None) -> Transcript:
    backend, processor, model, dev = _get_model(model_size, device=device, compute_type=compute_type)

    if backend == "faster-whisper":
        segments, info = model.transcribe(
            audio_path,
            language=None,
            task="transcribe",
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=False,
            initial_prompt=None,
        )
        segs = []
        seg_idx = 0
        max_end = None if duration_sec is None else start_offset_sec + duration_sec
        for segment in segments:
            seg_start = max(start_offset_sec, float(segment.start))
            seg_end = float(segment.end)
            if seg_end <= start_offset_sec:
                continue
            if max_end is not None and seg_start >= max_end:
                break
            if max_end is not None:
                seg_end = min(seg_end, max_end)
            text = (segment.text or "").strip()
            if not text:
                continue
            segs.append(ASRSegment(
                id=f"seg_{seg_idx:06d}",
                start=seg_start,
                end=seg_end,
                text=text,
                confidence=None,
            ))
            seg_idx += 1
        return Transcript(segments=segs, language=getattr(info, "language", None))

    audio = _load_audio(audio_path, offset_sec=start_offset_sec, duration_sec=duration_sec)
    dtype = torch.float16 if dev.type in ("cuda", "npu") else torch.float32

    segs = []
    seg_idx = 0
    detected_lang = None

    for chunk_start, chunk in _chunk_audio(audio, chunk_sec=chunk_sec):
        if len(chunk) == 0:
            continue
        input_features = processor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(dev).to(dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=448,
                language=None,
                task="transcribe",
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if not text:
            continue

        if detected_lang is None:
            try:
                forced_ids = model.config.forced_decoder_ids
                if forced_ids:
                    lang_token_id = forced_ids[0][1]
                    detected_lang = processor.tokenizer.decode([lang_token_id]).strip("<|>")
            except Exception:
                detected_lang = "unknown"

        chunk_duration = len(chunk) / 16000
        segs.append(ASRSegment(
            id=f"seg_{seg_idx:06d}",
            start=start_offset_sec + chunk_start,
            end=start_offset_sec + chunk_start + chunk_duration,
            text=text,
            confidence=None,
        ))
        seg_idx += 1

    return Transcript(segments=segs, language=detected_lang)


def _merge_transcripts(transcripts: list[Transcript]) -> Transcript:
    merged_segments = []
    language = None
    for transcript in transcripts:
        if transcript is None:
            continue
        if language is None and transcript.language:
            language = transcript.language
        merged_segments.extend(transcript.segments)

    merged_segments.sort(key=lambda s: (s.start, s.end, s.id))
    for idx, seg in enumerate(merged_segments):
        seg.id = f"seg_{idx:06d}"

    return Transcript(segments=merged_segments, language=language)


def _transcribe_clip_task(audio_path: str,
                          model_size: str,
                          device: str,
                          range_info: tuple[int, float, float],
                          compute_type: str = None) -> dict:
    index, start_sec, end_sec = range_info
    clip_path = _extract_audio_clip(audio_path, start_sec, end_sec)
    try:
        transcript = _transcribe_window(
            clip_path,
            model_size=model_size,
            device=device,
            start_offset_sec=0.0,
            duration_sec=None,
            compute_type=compute_type,
        )
    finally:
        try:
            os.unlink(clip_path)
        except OSError:
            pass

    for segment in transcript.segments:
        segment.start += start_sec
        segment.end += start_sec

    return {
        "index": index,
        "device": device,
        "transcript": transcript.model_dump(),
    }


def _run_parallel_transcribe(audio_path: str,
                             model_size: str,
                             worker_devices: list[str],
                             ranges: list[tuple[int, float, float]],
                             compute_type: str = None) -> Transcript:
    if not ranges:
        return Transcript(segments=[], language=None)

    if len(ranges) == 1 or len(worker_devices) <= 1:
        return _transcribe_window(
            audio_path,
            model_size=model_size,
            device=worker_devices[0] if worker_devices else None,
            compute_type=compute_type,
        )

    results = {}
    mp_ctx = mp.get_context("spawn")
    max_workers = min(len(worker_devices), len(ranges))
    logger.info(
        "Running parallel ASR over %d audio clips with %d workers on devices=%s",
        len(ranges), max_workers, worker_devices[:max_workers],
    )

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        future_map = {}
        for index, range_info in enumerate(ranges):
            device = worker_devices[index % len(worker_devices)]
            future = executor.submit(
                _transcribe_clip_task,
                audio_path,
                model_size,
                device,
                range_info,
                compute_type,
            )
            future_map[future] = range_info[0]

        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = Transcript.model_validate(future.result()["transcript"])

    ordered = [results[idx] for idx, _, _ in ranges if idx in results]
    return _merge_transcripts(ordered)


def _transcribe_sequential(audio_path: str,
                           model_size: str = "small",
                           device: str = None,
                           compute_type: str = None) -> Transcript:
    return _transcribe_window(audio_path, model_size=model_size, device=device, compute_type=compute_type)


@retry_with_backoff(max_retries=2, base_delay=3.0, max_delay=30.0,
                    retryable_exceptions=(RuntimeError, OSError, MemoryError))
def transcribe(audio_path: str, out_json_path: str,
               model_size: str = "small",
               device: str = None,
               compute_type: str = None,
               parallel: bool = False,
               max_workers: int = None,
               devices=None,
               segment_duration_sec: float = 300.0,
               min_audio_duration_sec: float = 300.0,
               min_segment_duration_sec: float = 30.0) -> Transcript:
    if exists_nonempty(out_json_path):
        return Transcript.model_validate(read_json(out_json_path))

    tr = None
    if parallel:
        audio_duration = _get_audio_duration(audio_path)
        worker_devices = _resolve_worker_devices(
            device=device,
            devices=devices,
            max_workers=max_workers,
        )
        ranges = _plan_audio_ranges(
            duration_sec=audio_duration,
            segment_duration_sec=segment_duration_sec,
            min_segment_duration_sec=min_segment_duration_sec,
        )
        if audio_duration >= min_audio_duration_sec and len(ranges) > 1 and len(worker_devices) > 1:
            if compute_type is None:
                tr = _run_parallel_transcribe(
                    audio_path,
                    model_size=model_size,
                    worker_devices=worker_devices,
                    ranges=ranges,
                )
            else:
                tr = _run_parallel_transcribe(
                    audio_path,
                    model_size=model_size,
                    worker_devices=worker_devices,
                    ranges=ranges,
                    compute_type=compute_type,
                )
        else:
            logger.info(
                "Parallel ASR not used: duration=%.1fs clips=%d workers=%d",
                audio_duration, len(ranges), len(worker_devices),
            )

    if tr is None:
        if compute_type is None:
            tr = _transcribe_sequential(audio_path, model_size=model_size, device=device)
        else:
            tr = _transcribe_sequential(audio_path, model_size=model_size, device=device, compute_type=compute_type)

    write_json(out_json_path, tr.model_dump())
    return tr
