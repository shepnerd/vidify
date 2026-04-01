# agent/skills/asr.py
import os
from faster_whisper import WhisperModel
from agent.core.schemas import Transcript, ASRSegment
from agent.extensions.utils.cache import write_json, exists_nonempty
from agent.core.retry import retry_with_backoff


def _best_device():
    """Pick CUDA if available, otherwise CPU with int8 quantization."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


@retry_with_backoff(max_retries=2, base_delay=3.0, max_delay=30.0,
                    retryable_exceptions=(RuntimeError, OSError, MemoryError))
def transcribe(audio_path: str, out_json_path: str,
               model_size: str = "small",
               device: str = None,
               compute_type: str = None) -> Transcript:
    if exists_nonempty(out_json_path):
        # 简化：读取就略了，你可自行实现 read_json -> Transcript
        pass

    if device is None or compute_type is None:
        auto_dev, auto_ct = _best_device()
        device = device or auto_dev
        compute_type = compute_type or auto_ct

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, vad_filter=True)

    segs = []
    for i, s in enumerate(segments):
        segs.append(ASRSegment(
            id=f"seg_{i:06d}",
            start=float(s.start), end=float(s.end),
            text=s.text.strip(),
            confidence=getattr(s, "avg_logprob", None)
        ))

    tr = Transcript(segments=segs, language=getattr(info, "language", None))
    write_json(out_json_path, tr.model_dump())
    return tr
