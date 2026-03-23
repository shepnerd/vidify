# agent/skills/video_download.py
import os, subprocess
from agent.extensions.utils.cache import ensure_dir, exists_nonempty

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")

def download_youtube(uri: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "source.mp4")
    if exists_nonempty(out_path):
        return out_path
    # yt-dlp 输出到固定文件名：先下到模板名再转名或直接用 -o
    cmd = ["yt-dlp", uri, "-f", "bv*+ba/b", "-o", out_path]
    _run(cmd)
    return out_path

def download_generic(uri: str, out_dir: str) -> str:
    # MVP：仍用 yt-dlp 试一下；后续再按站点规则扩展
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "source.mp4")
    if exists_nonempty(out_path):
        return out_path
    cmd = ["yt-dlp", uri, "-o", out_path]
    _run(cmd)
    return out_path