# agent/skills/audio_extract.py
import os, subprocess
from agent.extensions.utils.cache import ensure_dir, exists_nonempty

def extract_audio(video, out_path: str) -> str:
    ensure_dir(os.path.dirname(out_path))
    if exists_nonempty(out_path):
        return out_path
    cmd = ["ffmpeg", "-y", "-i", video.local_path,
           "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", out_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return out_path