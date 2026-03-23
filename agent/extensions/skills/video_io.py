# agent/skills/video_io.py
import os
from agent.core.schemas import VideoAsset, VideoSource
from agent.extensions.utils.cache import sha1, ensure_dir
from agent.skills.video_download import download_youtube, download_generic

def load_video(source_type: str, uri: str, cache_root: str) -> VideoAsset:
    vid = sha1(f"{source_type}:{uri}")
    cache_dir = ensure_dir(os.path.join(cache_root, "videos", vid))

    if source_type == "local":
        local_path = os.path.abspath(uri)
        if not os.path.exists(local_path):
            raise FileNotFoundError(local_path)
        return VideoAsset(id=vid, source=VideoSource(type="local", uri=uri),
                          local_path=local_path, cache_dir=cache_dir)

    if source_type == "youtube":
        local_path = download_youtube(uri, cache_dir)
        return VideoAsset(id=vid, source=VideoSource(type="youtube", uri=uri),
                          local_path=local_path, cache_dir=cache_dir)

    if source_type == "url":
        local_path = download_generic(uri, cache_dir)
        return VideoAsset(id=vid, source=VideoSource(type="url", uri=uri),
                          local_path=local_path, cache_dir=cache_dir)

    raise ValueError(f"Unknown source_type: {source_type}")