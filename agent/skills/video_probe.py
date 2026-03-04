# agent/skills/video_probe.py
import json, subprocess
from agent.schemas import VideoMetadata

def probe_video(local_path: str) -> VideoMetadata:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=index,codec_type,width,height,r_frame_rate",
        "-of", "json", local_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    obj = json.loads(p.stdout)

    duration = float(obj["format"]["duration"])
    streams = obj.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v:
        raise RuntimeError("No video stream")

    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    r = v.get("r_frame_rate", "0/1")
    num, den = r.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    return VideoMetadata(duration_sec=duration, fps=fps, width=w, height=h, has_audio=(a is not None))