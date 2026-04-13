import os
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from agent.core.schemas import Transcript, ASRSegment
from agent.extensions.utils.cache import write_json, exists_nonempty
from agent.core.retry import retry_with_backoff

import logging

logger = logging.getLogger(__name__)

_processor = None
_model = None
_device = None


def resolve_whisper_model(model_size: str = "small") -> str:
    """Resolve a Whisper model identifier to a local path or HF repo id."""
    from agent.config import get_model_path

    model_id = get_model_path(f"whisper-{model_size}")
    if os.path.isdir(model_id):
        return model_id
    return f"openai/whisper-{model_size}"


def has_local_whisper_model(model_size: str = "small") -> bool:
    """Return True when the requested Whisper model exists under models/."""
    return os.path.isdir(resolve_whisper_model(model_size))


def _detect_device():
    """Pick best available device: NPU > CUDA > CPU."""
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return torch.device("npu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_model(model_size: str = "small"):
    """Lazy-init Whisper model on the best available device."""
    global _processor, _model, _device
    if _model is not None:
        return _processor, _model, _device

    _device = _detect_device()
    model_id = resolve_whisper_model(model_size)

    logger.info("Loading Whisper model %s on %s", model_id, _device)
    _processor = WhisperProcessor.from_pretrained(model_id)
    _model = WhisperForConditionalGeneration.from_pretrained(model_id)
    dtype = torch.float16 if _device.type in ("cuda", "npu") else torch.float32
    _model = _model.to(_device).to(dtype)
    _model.eval()

    return _processor, _model, _device


def _load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio via librosa (already a dependency for emotion_analysis)."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def _chunk_audio(audio: np.ndarray, sr: int = 16000, chunk_sec: float = 30.0):
    """Split audio into chunks for Whisper's 30-second context window."""
    chunk_len = int(sr * chunk_sec)
    for start in range(0, len(audio), chunk_len):
        yield start / sr, audio[start:start + chunk_len]


@retry_with_backoff(max_retries=2, base_delay=3.0, max_delay=30.0,
                    retryable_exceptions=(RuntimeError, OSError, MemoryError))
def transcribe(audio_path: str, out_json_path: str,
               model_size: str = "small",
               device: str = None,
               compute_type: str = None) -> Transcript:
    if exists_nonempty(out_json_path):
        pass

    processor, model, dev = _get_model(model_size)
    # Override device if explicitly specified
    if device is not None:
        dev = torch.device(device)
        model.to(dev)

    audio = _load_audio(audio_path)
    dtype = torch.float16 if dev.type in ("cuda", "npu") else torch.float32

    segs = []
    seg_idx = 0
    detected_lang = None

    for chunk_start, chunk in _chunk_audio(audio):
        input_features = processor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(dev).to(dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=448,
                language=None,  # auto-detect
                task="transcribe",
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if not text:
            continue

        # Detect language from first chunk
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
            start=chunk_start,
            end=chunk_start + chunk_duration,
            text=text,
            confidence=None,
        ))
        seg_idx += 1

    tr = Transcript(segments=segs, language=detected_lang)
    write_json(out_json_path, tr.model_dump())
    return tr
