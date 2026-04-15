# agent/skills/frame_sampler.py
import glob
import os
import re
import subprocess
from agent.core.schemas import FrameSet, FrameItem, FrameStrategy
from agent.extensions.utils.cache import ensure_dir

PTS_RE = re.compile(r".*_(\d+)\.(jpg|png)$")

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")


def _clip_duration_sec(video, start_sec: float = None, end_sec: float = None) -> float | None:
    if start_sec is not None and end_sec is not None:
        return max(0.0, float(end_sec) - float(start_sec))
    if end_sec is not None:
        return max(0.0, float(end_sec))
    if getattr(video, "metadata", None) is not None:
        duration = getattr(video.metadata, "duration_sec", None)
        if duration is not None:
            return max(0.0, float(duration) - float(start_sec or 0.0))
    return None


def _resolve_frame_budget(strategy: FrameStrategy, duration_sec: float | None) -> int:
    max_frames = max(1, int(strategy.params.get("max_frames", 128)))
    if not strategy.params.get("adaptive_by_duration"):
        return max_frames

    min_frames = max(1, int(strategy.params.get("min_frames", min(16, max_frames))))
    min_frames = min(min_frames, max_frames)
    if duration_sec is None:
        return max_frames

    target = int(round(float(duration_sec)))
    target = max(min_frames, target)
    return min(max_frames, target)

def sample_frames(video, out_dir: str, strategy: FrameStrategy,
                  start_sec: float = None, end_sec: float = None) -> FrameSet:
    """Sample frames from a video using the given strategy.

    Args:
        video: VideoAsset with local_path and metadata.
        out_dir: Directory to write extracted frames.
        strategy: FrameStrategy (fps or scene).
        start_sec: If set, only process video from this timestamp (for segment parallelism).
        end_sec: If set, only process video up to this timestamp.
    """
    ensure_dir(out_dir)
    clip_duration = _clip_duration_sec(video, start_sec=start_sec, end_sec=end_sec)
    max_frames = _resolve_frame_budget(strategy, clip_duration)

    # Build FFmpeg time-range flags for segment processing
    time_flags = []
    if start_sec is not None:
        time_flags += ["-ss", str(start_sec)]
    if end_sec is not None:
        if start_sec is not None:
            time_flags += ["-to", str(end_sec - start_sec)]  # -to is relative when after -ss before -i
        else:
            time_flags += ["-to", str(end_sec)]

    ts_offset = start_sec if start_sec is not None else 0.0

    if strategy.type == "fps":
        fps = float(strategy.params.get("fps", 1.0))
        out_tpl = os.path.join(out_dir, "f_%06d.jpg")
        cmd = ["ffmpeg", "-y"] + time_flags + ["-i", video.local_path,
               "-vf", f"fps={fps},scale=256:144:force_original_aspect_ratio=decrease",
               "-q:v", "2", out_tpl]
        _run(cmd)

    elif strategy.type == "scene":
        th = float(strategy.params.get("scene_threshold", 0.3))
        out_tpl = os.path.join(out_dir, "f_%06d.jpg")
        vf = f"select='gt(scene,{th})',scale=256:144:force_original_aspect_ratio=decrease"
        cmd = ["ffmpeg", "-y"] + time_flags + ["-i", video.local_path,
               "-vf", vf, "-vsync", "vfr", "-q:v", "2", out_tpl]
        _run(cmd)
    else:
        raise ValueError(f"Unknown strategy: {strategy.type}")

    paths = sorted(glob.glob(os.path.join(out_dir, "f_*.jpg")))
    items = []

    # Determine segment duration for timestamp estimation
    seg_duration = clip_duration

    for i, pth in enumerate(paths[:max_frames]):
        m = PTS_RE.match(pth)
        frame_num = int(m.group(1)) if m else i
        if strategy.type == "fps":
            fps = float(strategy.params.get("fps", 1.0))
            ts = ts_offset + frame_num / fps
        else:  # scene
            # Approximate timestamp: distribute within segment range
            if seg_duration and len(paths) > 0:
                ts = ts_offset + frame_num * (seg_duration / len(paths))
            else:
                ts = ts_offset + frame_num * 1.0
        fid = f"f_{i:04d}"
        items.append(FrameItem(id=fid, ts=ts, path=pth))
    return FrameSet(items=items, strategy=strategy)
