# agent/skills/asr.py
import os
from faster_whisper import WhisperModel
from agent.core.schemas import Transcript, ASRSegment
from agent.extensions.utils.cache import write_json, exists_nonempty

def transcribe(audio_path: str, out_json_path: str,
               model_size: str = "small",
               device: str = "cuda",
               compute_type: str = "float16") -> Transcript:
    if exists_nonempty(out_json_path):
        # 简化：读取就略了，你可自行实现 read_json -> Transcript
        pass

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