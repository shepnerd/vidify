#!/usr/bin/env python3
"""
Download all required models for VidCopilot and organize them.

HuggingFace models stay in the HF cache (managed by huggingface_hub).
Non-HF models (PaddleOCR, YOLOv8) are placed under models/ in the project root.

Usage:
    python scripts/download_models.py          # download all
    python scripts/download_models.py --only whisper paddleocr yolo
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_whisper(size: str = "small"):
    """Download Whisper model via HuggingFace transformers (stays in HF cache or models/)."""
    local_dir = MODELS_DIR / f"whisper-{size}"
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[whisper] Already present: {local_dir}")
        return str(local_dir)
    try:
        from huggingface_hub import snapshot_download
        repo = f"openai/whisper-{size}"
        print(f"[whisper] Downloading {repo} to {local_dir} ...")
        path = snapshot_download(repo, local_dir=str(local_dir))
        print(f"[whisper] OK: {path}")
        return path
    except Exception as e:
        print(f"[whisper] FAILED: {e}")
        return None


def download_paddleocr():
    """Download PaddleOCR models and copy to models/paddleocr/."""
    paddle_dir = MODELS_DIR / "paddleocr"
    # Check if already present
    if all((paddle_dir / d / "inference.pdmodel").exists() for d in ("det", "rec", "cls")):
        print("[paddleocr] Already present in models/paddleocr/")
        return True

    try:
        # Initialize PaddleOCR to trigger downloads to ~/.paddleocr/
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        print("[paddleocr] Initializing PaddleOCR (downloads to ~/.paddleocr/) ...")
        PaddleOCR(use_angle_cls=True, lang="ch")

        # Copy to models/paddleocr/
        src_map = {
            "det": Path.home() / ".paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer",
            "rec": Path.home() / ".paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer",
            "cls": Path.home() / ".paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
        }
        for name, src in src_map.items():
            dst = paddle_dir / name
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"[paddleocr] Copied {name}: {src} -> {dst}")
            else:
                print(f"[paddleocr] WARNING: source not found: {src}")
        return True
    except Exception as e:
        print(f"[paddleocr] FAILED: {e}")
        return False


def download_yolo(model_name: str = "yolov8n.pt"):
    """Download YOLO model to models/ and pre-cache ultralytics assets."""
    dst = MODELS_DIR / model_name
    if dst.exists():
        print(f"[yolo] Already present: {dst}")
    else:
        try:
            from ultralytics import YOLO
            print(f"[yolo] Downloading {model_name} ...")
            model = YOLO(model_name)
            # ultralytics downloads to CWD; move to models/
            src = Path(model_name)
            if src.exists() and src != dst:
                shutil.move(str(src), str(dst))
            print(f"[yolo] OK: {dst}")
        except Exception as e:
            print(f"[yolo] FAILED: {e}")
            return False

    # Pre-cache the Arial.ttf font that ultralytics downloads at plot time
    try:
        from ultralytics.utils import ASSETS_URL  # noqa: F401
        from ultralytics.utils.downloads import attempt_download_asset
        from ultralytics import settings as ul_settings
        font_dir = Path(ul_settings.get("datasets_dir", Path.home() / "datasets"))
        # Trigger font download so it's cached for offline use
        from ultralytics.utils.plotting import Annotator
        import numpy as np
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        Annotator(dummy)
        print("[yolo] Font/assets pre-cached for offline use")
    except Exception as e:
        print(f"[yolo] Font pre-cache skipped: {e}")

    return True


def download_emotion():
    """Download emotion analysis model (HF-managed)."""
    try:
        from huggingface_hub import snapshot_download
        repo = "superb/wav2vec2-base-superb-er"
        print(f"[emotion] Downloading {repo} (HF-managed) ...")
        path = snapshot_download(repo)
        print(f"[emotion] OK: {path}")
        return path
    except Exception as e:
        print(f"[emotion] FAILED: {e}")
        return None


def download_translation():
    """Download translation models (HF-managed)."""
    try:
        from huggingface_hub import snapshot_download
        for repo in ("Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"):
            print(f"[translation] Downloading {repo} (HF-managed) ...")
            path = snapshot_download(repo)
            print(f"[translation] OK: {path}")
        return True
    except Exception as e:
        print(f"[translation] FAILED: {e}")
        return False


ALL_MODELS = ["whisper", "paddleocr", "yolo", "emotion", "translation"]

DOWNLOAD_MAP = {
    "whisper": download_whisper,
    "paddleocr": download_paddleocr,
    "yolo": download_yolo,
    "emotion": download_emotion,
    "translation": download_translation,
}


def main():
    parser = argparse.ArgumentParser(description="Download VidCopilot models")
    parser.add_argument("--only", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                        help="Download only specific models (default: all)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Models directory: {MODELS_DIR}")
    print(f"Downloading: {', '.join(args.only)}\n")

    for name in args.only:
        DOWNLOAD_MAP[name]()
        print()

    # Summary
    print("=" * 50)
    print("Model inventory:")
    print("=" * 50)
    print(f"\n  models/ ({MODELS_DIR}):")
    for p in sorted(MODELS_DIR.rglob("*")):
        if p.is_file():
            rel = p.relative_to(MODELS_DIR)
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"    {rel}  ({size_mb:.1f} MB)")

    print(f"\n  HuggingFace cache ({Path.home() / '.cache/huggingface/hub'}):")
    hf_dir = Path.home() / ".cache/huggingface/hub"
    if hf_dir.exists():
        for d in sorted(hf_dir.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                print(f"    {d.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
