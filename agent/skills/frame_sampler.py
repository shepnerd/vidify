# agent/skills/frame_sampler.py
import os, re, subprocess, glob
from agent.schemas import FrameSet, FrameItem, FrameStrategy
from agent.utils.cache import ensure_dir

PTS_RE = re.compile(r".*_(\d+)\.(jpg|png)$")

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")

def sample_frames(video, out_dir: str, strategy: FrameStrategy) -> FrameSet:
    ensure_dir(out_dir)
    # 清理旧产物（MVP 简化：直接复用也行；这里先不删）
    max_frames = int(strategy.params.get("max_frames", 128))

    if strategy.type == "fps":
        fps = float(strategy.params.get("fps", 1.0))
        # 文件名包含 frame number
        out_tpl = os.path.join(out_dir, "f_%06d.jpg")
        cmd = ["ffmpeg", "-y", "-i", video.local_path,
               "-vf", f"fps={fps},scale=256:144:force_original_aspect_ratio=decrease",
               "-q:v", "2", out_tpl]
        _run(cmd)

    elif strategy.type == "scene":
        th = float(strategy.params.get("scene_threshold", 0.3))
        out_tpl = os.path.join(out_dir, "f_%06d.jpg")
        # select 场景切分；scene 越大越“变化明显”
        vf = f"select='gt(scene,{th})',scale=256:144:force_original_aspect_ratio=decrease"
        cmd = ["ffmpeg", "-y", "-i", video.local_path,
               "-vf", vf, "-vsync", "vfr", "-q:v", "2", out_tpl]
        _run(cmd)
    else:
        raise ValueError(f"Unknown strategy: {strategy.type}")

    paths = sorted(glob.glob(os.path.join(out_dir, "f_*.jpg")))
    items = []
    for i, pth in enumerate(paths[:max_frames]):
        m = PTS_RE.match(pth)
        frame_num = int(m.group(1)) if m else i
        if strategy.type == "fps":
            fps = float(strategy.params.get("fps", 1.0))
            ts = frame_num / fps
        else:  # scene
            # Approximate timestamp based on frame number
            ts = frame_num * (video.metadata.duration_sec / len(paths)) if video.metadata else frame_num * 1.0
        fid = f"f_{i:04d}"
        items.append(FrameItem(id=fid, ts=ts, path=pth))
    return FrameSet(items=items, strategy=strategy)