# agent/extensions/utils/__init__.py
import json
import os
import subprocess


def unset_proxy():
    """Remove proxy env vars that interfere with local vLLM connections.

    Cluster service-mesh proxies (e.g. kubebrain) intercept multimodal
    POST payloads and corrupt them.  Call this early — before importing
    httpx / openai — to ensure direct connections to vLLM.
    """
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def split_video_segment(video_path: str, start: float, duration: float,
                        out_path: str, scale_w: int = 640) -> str:
    """Extract a video segment with re-encoding and downscale.

    Re-encoding (instead of ``-c copy``) ensures the output has correct
    keyframes and is readable by OpenCV / Qwen3-VL frame extraction.
    Downscaling to *scale_w* (default 640px wide) reduces MLLM token
    cost without losing semantic information.

    Returns *out_path*.
    """
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", str(start), "-i", video_path, "-t", str(duration),
         "-vf", f"scale={scale_w}:-2",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-an", out_path],
        capture_output=True, check=True,
    )
    return out_path


def get_short_clip(video_path: str, max_duration: float = 30.0,
                   scale_w: int = 640) -> str:
    """Return *video_path* if short enough, otherwise a bounded clip.

    The clip is cached next to the source so repeated calls are free.
    """
    duration = get_video_duration(video_path)
    if duration <= max_duration:
        return video_path

    clip_name = f"clip_{int(max_duration)}s_{scale_w}w.mp4"
    clip_path = os.path.join(os.path.dirname(video_path), clip_name)
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return clip_path

    split_video_segment(video_path, 0, max_duration, clip_path, scale_w=scale_w)
    return clip_path


def make_video_content(video_path: str) -> dict:
    """Build the OpenAI-compatible video content block for vLLM.

    vLLM expects ``{"type": "video_url", "video_url": {"url": "file://..."}}``,
    NOT ``{"type": "video", "video": "file://..."}``.
    """
    return {"type": "video_url", "video_url": {"url": f"file://{video_path}"}}
