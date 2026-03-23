import yaml
import os
from typing import Dict, Any

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

def get_default_models_config() -> Dict[str, Any]:
    return {
        "mllm": {
            "heavy": {
                "model_name": "qwen-vl-7b",
                "base_url": "http://localhost:8000/v1",
                "max_tokens": 512,
                "temperature": 0.7
            },
            "light": {
                "model_name": "qwen-vl-1b",
                "base_url": "http://localhost:8000/v1",
                "max_tokens": 256,
                "temperature": 0.5
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
            "include_web_search": False
        },
        "detailed": {
            "use_advanced_skills": True,
            "max_frames": 128,
            "heavy_interval": 5
        },
        "live_stream": {
            "source": "webcam",
            "resolution": [640, 480],
            "fps": 1,
            "heavy_interval": 5
        }
    }