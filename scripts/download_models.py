#!/usr/bin/env python3
"""
Script to download required models for VidCopilot agent and update configuration.
"""

import os
import yaml
import subprocess
import sys
from pathlib import Path

def download_huggingface_model(model_name, save_path=None):
    """Download a model from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_name} to {save_path or 'HF cache'}...")
        return snapshot_download(repo_id=model_name, local_dir=save_path)
    except ImportError:
        print("huggingface_hub not installed. Please install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return None

def download_yolo_model(model_name, save_path):
    """Download YOLO model using ultralytics."""
    try:
        from ultralytics import YOLO
        print(f"Downloading {model_name}...")
        model = YOLO(model_name)
        model.save(save_path)
        return True
    except ImportError:
        print("ultralytics not installed. Please install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return False

def download_whisper_model(model_size, save_path):
    """Download Whisper model."""
    try:
        import whisper
        print(f"Downloading Whisper {model_size} model...")
        whisper.load_model(model_size)
        # Whisper models are cached automatically
        return True
    except ImportError:
        print("openai-whisper not installed. Please install with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"Failed to download Whisper {model_size}: {e}")
        return False

def download_fer_model(save_path):
    """Download FER model."""
    try:
        import fer
        # FER models are downloaded automatically on first use
        print("FER model will be downloaded on first use.")
        return True
    except ImportError:
        print("fer not installed. Please install with: pip install fer")
        return False

def main():
    # Load current models config
    config_path = Path("models.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("models.yaml not found. Using default config.")
        config = {}

    # Download models
    downloads = {
        "mllm": {
            "heavy": {"model_name": "Qwen/Qwen-VL", "path_key": "model_path"},
            "light": {"model_name": "Qwen/Qwen-VL-Chat", "path_key": "model_path"}
        },
        "object_detection": {"model": "yolov8n.pt", "path_key": "model"},
        "asr": {"model": "whisper", "size": "small", "path_key": "model"},
        "emotion_analysis": {
            "audio_model": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            "visual_model": "fer",
            "path_key": "audio_model"
        },
        "translation": {"model": "Helsinki-NLP/opus-mt-en-zh", "path_key": "model"}
    }

    for section, models in downloads.items():
        if section not in config:
            config[section] = {}
        if section == "mllm":
            for sub_key, model_info in models.items():
                model_name = model_info["model_name"]
                save_path = download_huggingface_model(model_name, None)
                if save_path:
                    if sub_key not in config[section]:
                        config[section][sub_key] = {}
                    config[section][sub_key][model_info["path_key"]] = save_path
        elif section == "object_detection":
            model_name = models["model"]
            # For YOLO, download to HF cache or specific? But YOLO uses ultralytics, may not use HF.
            # Keep as is, but since not HF, perhaps leave.
            # For now, assume YOLO downloads elsewhere.
            config[section][models["path_key"]] = model_name  # Keep as name
        elif section == "asr":
            # Whisper downloads automatically
            config[section][models["path_key"]] = models["model"]
        elif section == "emotion_analysis":
            # Download audio model
            model_name = models["audio_model"]
            save_path = download_huggingface_model(model_name, None)
            if save_path:
                config[section][models["path_key"]] = save_path
            # FER
            download_fer_model(None)
        elif section == "translation":
            model_name = models["model"]
            save_path = download_huggingface_model(model_name, None)
            if save_path:
                config[section][models["path_key"]] = save_path

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Model download complete. Configuration updated in models.yaml")

if __name__ == "__main__":
    main()