import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_default_config() -> Dict[str, Any]:
    return {
        "llm_base_url": "http://localhost:8000/v1",
        "llm_model": "qwen-vl",
        "embed_base_url": "http://localhost:8000/v1",
        "embed_model": "qwen-embed",
        "cache_root": "./cache",
        "max_frames": 128,
        "whisper_model": "small",
        "chunk_sec": 20,
        "top_k": 5,
        "max_clips": 5,
        "also_make_reel": True,
        "include_web_search": False,
        "google_api_key": None,
        "google_search_engine_id": None,
        "direct_model": False,
        "model_path": "/models/qwen-vl",
        "tokenizer_path": None
    }