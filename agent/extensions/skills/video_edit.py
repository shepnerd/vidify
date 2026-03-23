# agent/skills/video_edit.py
import os, subprocess
from agent.extensions.utils.cache import ensure_dir

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

def export_highlight_clips(video, highlights, out_dir: str):
    ensure_dir(out_dir)
    out = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"clip_{i:02d}_{h.start:.1f}-{h.end:.1f}.mp4")
        cmd = ["ffmpeg", "-y", "-ss", str(h.start), "-to", str(h.end), "-i", video.local_path,
               "-c", "copy", out_path]
        _run(cmd)
        h.output_path = out_path
        out.append(h)
    return out

def export_highlight_reel(clips, out_path: str) -> dict:
    """
    把 clips 按顺序拼成一个 reel。返回 reel_timeline：每段在 reel 中的起止。
    MVP：使用 concat demuxer（要求编码参数一致；若不一致需 re-encode）。
    """
    ensure_dir(os.path.dirname(out_path))
    list_path = out_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c.output_path}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path]
    _run(cmd)

    # 生成 reel 映射（MVP：用 clip 时长估算；后续可 ffprobe 精确）
    cur = 0.0
    reel_timeline = []
    for c in clips:
        dur = float(c.end - c.start)
        c.reel_start = cur
        c.reel_end = cur + dur
        reel_timeline.append({"clip": os.path.basename(c.output_path), "reel_start": cur, "reel_end": cur + dur,
                              "src_start": c.start, "src_end": c.end})
        cur += dur

    return {"reel_path": out_path, "reel_timeline": reel_timeline}