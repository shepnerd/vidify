# agent/extensions/utils/__init__.py
import base64
import io
import json
import os
import subprocess


def unset_proxy():
    """Remove proxy env vars that interfere with local vLLM connections.

    Cluster service-mesh proxies can intercept multimodal POST payloads
    and corrupt them.  Call this early — before importing httpx / openai —
    to ensure direct connections to vLLM.
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
    keyframes and is readable by OpenCV / Qwen3.5 frame extraction.
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
                   scale_w: int = 640, cache_dir: str = None) -> str:
    """Return *video_path* if short enough, otherwise a bounded clip.

    The clip is cached in *cache_dir* (or next to the source) so repeated
    calls are free.
    """
    duration = get_video_duration(video_path)
    if duration <= max_duration:
        return video_path

    parent = cache_dir or os.path.dirname(video_path)
    clip_name = f"clip_{int(max_duration)}s_{scale_w}w.mp4"
    clip_path = os.path.join(parent, clip_name)
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return clip_path

    split_video_segment(video_path, 0, max_duration, clip_path, scale_w=scale_w)
    return clip_path


def get_short_audio(video_path: str, max_duration: float = 60.0,
                    cache_dir: str = None) -> str:
    """Extract a short mono 16 kHz WAV from *video_path*.

    The result is cached next to the source (or in *cache_dir*) so
    repeated calls are free.
    """
    parent = cache_dir or os.path.dirname(video_path)
    audio_path = os.path.join(parent, f"audio_{int(max_duration)}s.wav")
    if os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-t", str(max_duration),
         "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", audio_path],
        capture_output=True, check=True,
    )
    return audio_path


def extract_frames(video_path: str, out_dir: str, fps: float = 0.2,
                   max_frames: int = 8) -> list:
    """Extract frames from *video_path* at *fps*.

    Returns a list of dicts ``{"path": ..., "ts": ..., "id": ...}``.
    """
    import glob as _globmod
    os.makedirs(out_dir, exist_ok=True)
    out_tpl = os.path.join(out_dir, "f_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps={fps},scale=256:144:force_original_aspect_ratio=decrease",
         "-q:v", "2", out_tpl],
        capture_output=True, check=True,
    )
    paths = sorted(_globmod.glob(os.path.join(out_dir, "f_*.jpg")))[:max_frames]
    frames = []
    for i, p in enumerate(paths):
        ts = (i + 1) / fps
        frames.append({"path": p, "ts": ts, "id": f"f_{i:04d}"})
    return frames


def make_video_content(video_path: str) -> dict:
    """Build the OpenAI-compatible video content block for vLLM.

    vLLM expects ``{"type": "video_url", "video_url": {"url": "file://..."}}``,
    NOT ``{"type": "video", "video": "file://..."}``.
    """
    return {"type": "video_url", "video_url": {"url": f"file://{video_path}"}}


def img_to_data_url(path: str, max_w: int = 256, max_h: int = 144,
                    fmt: str = "JPEG", quality: int = 85) -> str:
    """Convert an image to a base64 data URL, optionally downscaling."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"
