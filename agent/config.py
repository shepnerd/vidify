import yaml
import os
from typing import Dict, Any

# ── Central model directory ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")


def get_model_path(subpath: str) -> str:
    """Resolve a model path under MODEL_ROOT.

    Returns the full path if it exists under models/, otherwise returns
    *subpath* unchanged so the calling library can fall back to its own
    download / cache logic (e.g. HuggingFace model IDs).
    """
    full = os.path.join(MODEL_ROOT, subpath)
    if os.path.exists(full):
        return full
    return subpath

def get_config() -> Dict[str, Any]:
    """Return the merged models config (used by skills for model paths)."""
    return load_models_config()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_models_config(models_path: str = "models.yaml") -> Dict[str, Any]:
    if os.path.exists(models_path):
        with open(models_path, 'r') as f:
            return yaml.safe_load(f)
    return get_default_models_config()

def load_workflows_config(workflows_path: str = "workflows.yaml") -> Dict[str, Any]:
    if os.path.exists(workflows_path):
        with open(workflows_path, 'r') as f:
            return yaml.safe_load(f)
    return get_default_workflows_config()

def get_default_config() -> Dict[str, Any]:
    return {
        "llm_base_url": "http://localhost:8000/v1",
        "llm_model": "qwen3.5-9b",
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
        "model_path": None,
        "tokenizer_path": None,
        "log_format": "text",  # "json" for structured logging
    }

def get_default_models_config() -> Dict[str, Any]:
    return {
        "mllm": {
            "heavy": {
                "model_name": "qwen3.5-9b",
                "base_url": "http://localhost:8000/v1",
                "max_tokens": 512,
                "temperature": 0.7
            },
            "light": {
                "model_name": "qwen3.5-4b",
                "base_url": "http://localhost:8000/v1",
                "max_tokens": 256,
                "temperature": 0.5
            },
            "mla": {
                "model_name": "qwen3-mla",
                "base_url": "http://localhost:8001/v1",
                "max_tokens": 512,
                "temperature": 0.7
            }
        },
        "ocr": {
            "engine": "paddleocr",
            "lang": "ch",
            "use_angle_cls": True
        },
        "object_detection": {
            "model": "yolov8n.pt",
            "conf_threshold": 0.5
        },
        "asr": {
            "model": "whisper",
            "size": "small",
            "language": None
        },
        "emotion_analysis": {
            "audio_model": "wav2vec2-emotion",
            "visual_model": "fer"
        },
        "translation": {
            "source_lang": "en",
            "target_lang": "zh",
            "model": "helsinki-nlp"
        }
    }

def get_default_workflows_config() -> Dict[str, Any]:
    return {
        "brief": {
            "use_asr": True,
            "max_frames": 64,
            "min_frames": 16,
            "adaptive_frame_sampling": True,
            "include_web_search": False,
            "asr_first": True,
            "min_coverage_ratio": 0.3,
            "min_word_count": 50,
            "force_visual": False,
            "prefer_subtitles_over_asr": True,
            "parallel_asr": {
                "enabled": False,
                "max_workers": 4,
                "segment_duration": 240,
                "min_audio_duration": 300,
                "min_segment_duration": 30,
            },
        },
        "detailed": {
            "use_advanced_skills": True,
            "max_frames": 128,
            "min_frames": 16,
            "adaptive_frame_sampling": True,
            "heavy_interval": 5,
            "include_web_search": False,
            "asr_first": True,
            "min_coverage_ratio": 0.3,
            "min_word_count": 50,
            "force_visual": False,
            "prefer_subtitles_over_asr": True,
            "max_parallel_skills": 3,
            "parallel_asr": {
                "enabled": False,
                "max_workers": 4,
                "segment_duration": 240,
                "min_audio_duration": 300,
                "min_segment_duration": 30,
            },
        },
        "live_stream": {
            "source": "webcam",
            "resolution": [640, 480],
            "fps": 1,
            "heavy_interval": 5,
            "similarity_threshold": 0.9,
            "min_segment_frames": 3,
            "max_segment_frames": 16
        }
    }
