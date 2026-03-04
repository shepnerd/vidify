# agent/skills/persist.py
import os
from agent.utils.cache import write_json, read_json, ensure_dir

def analysis_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "analysis.json")

def save_analysis(cache_dir: str, obj: dict) -> str:
    path = analysis_path(cache_dir)
    ensure_dir(os.path.dirname(path))
    write_json(path, obj)
    return path

def load_analysis(cache_dir: str) -> dict:
    return read_json(analysis_path(cache_dir))