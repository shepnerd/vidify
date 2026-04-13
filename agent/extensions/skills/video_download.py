# agent/skills/video_download.py
import os, subprocess, json, glob
from agent.extensions.utils.cache import ensure_dir, exists_nonempty
from agent.core.schemas import ContentMetadata, SubtitleTrack

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")


def _subtitle_args_for_uri(uri: str) -> list[str]:
    # Bilibili exposes "danmaku" XML alongside subtitles. Exclude it so yt-dlp
    # does not try to run subtitle conversion on a non-subtitle XML payload.
    if "bilibili.com" in uri or "b23.tv" in uri:
        return ["--sub-langs", "all,-danmaku"]
    return []

def parse_info_json(info_json_path: str) -> ContentMetadata | None:
    if not info_json_path or not os.path.exists(info_json_path):
        return None
    with open(info_json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return ContentMetadata(
        title=info.get("title"),
        description=info.get("description"),
        uploader=info.get("uploader") or info.get("channel"),
        upload_date=info.get("upload_date"),
        tags=info.get("tags") or [],
        categories=info.get("categories") or [],
        duration_from_source=info.get("duration"),
        view_count=info.get("view_count"),
    )

def find_subtitle_files(out_dir: str) -> list[SubtitleTrack]:
    tracks = []
    for ext in ("vtt", "srt"):
        for path in glob.glob(os.path.join(out_dir, f"*.{ext}")):
            fname = os.path.basename(path)
            # yt-dlp naming: source.LANG.ext or source.LANG.ext for auto
            # Detect language and source type from filename
            parts = fname.rsplit(".", 2)  # e.g. ["source.en", "vtt"] or ["source", "en", "vtt"]
            lang = "unknown"
            source_type = "manual"
            name_no_ext = fname.rsplit(f".{ext}", 1)[0]  # e.g. "source.en" or "source.en.auto"
            # Remove the base "source" prefix
            suffix = name_no_ext.replace("source", "").strip(".")
            if suffix:
                # Could be "en", "en.auto", "zh-Hans", etc.
                if ".auto" in suffix or "-auto" in suffix:
                    source_type = "auto"
                    lang = suffix.replace(".auto", "").replace("-auto", "").strip(".")
                else:
                    lang = suffix
            if lang:
                tracks.append(SubtitleTrack(
                    language=lang, source=source_type, format=ext, path=path
                ))
    return tracks

def download_youtube(uri: str, out_dir: str) -> dict:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "source.mp4")
    info_json_path = os.path.join(out_dir, "source.info.json")

    if not exists_nonempty(out_path):
        cmd = [
            "yt-dlp", uri,
            "-f", "bv*+ba/b",
            "-o", out_path,
            "--write-info-json",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", "en,zh,ja,ko,es,fr,de,pt,ru",
            "--sub-format", "vtt",
            "--convert-subs", "vtt",
        ]
        _run(cmd)

    content_metadata = parse_info_json(info_json_path)
    subtitle_tracks = find_subtitle_files(out_dir)
    if content_metadata:
        content_metadata.subtitles = subtitle_tracks

    return {
        "video_path": out_path,
        "info_json_path": info_json_path if os.path.exists(info_json_path) else None,
        "content_metadata": content_metadata,
        "subtitle_tracks": subtitle_tracks,
    }

def download_generic(uri: str, out_dir: str) -> dict:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "source.mp4")
    info_json_path = os.path.join(out_dir, "source.info.json")

    if not exists_nonempty(out_path):
        cmd = [
            "yt-dlp", uri,
            "-o", out_path,
            "--write-info-json",
            "--write-subs", "--write-auto-subs",
            *_subtitle_args_for_uri(uri),
            "--sub-format", "vtt",
            "--convert-subs", "vtt",
        ]
        _run(cmd)

    content_metadata = parse_info_json(info_json_path)
    subtitle_tracks = find_subtitle_files(out_dir)
    if content_metadata:
        content_metadata.subtitles = subtitle_tracks

    return {
        "video_path": out_path,
        "info_json_path": info_json_path if os.path.exists(info_json_path) else None,
        "content_metadata": content_metadata,
        "subtitle_tracks": subtitle_tracks,
    }
