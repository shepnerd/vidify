import os
# Disable OneDNN/MKL-DNN to avoid potential conflicts with PaddlePaddle
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import json
import subprocess
import sys
import tempfile
import numpy as np
from typing import List, Dict, Any
from agent.config import get_model_path

_ocr = None


def _get_ocr():
    """Lazy-init PaddleOCR so import doesn't crash if models are missing."""
    global _ocr
    if _ocr is None:
        try:
            import paddle
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            det_model_dir=get_model_path("paddleocr/det"),
            rec_model_dir=get_model_path("paddleocr/rec"),
            cls_model_dir=get_model_path("paddleocr/cls"),
            enable_mkldnn=False,
        )
    return _ocr


def _ocr_in_subprocess(frame_path: str) -> List[Dict[str, Any]]:
    """Run OCR in a subprocess to avoid OneDNN conflicts with CTranslate2."""
    script = f'''
import os, sys, json
os.environ["FLAGS_use_mkldnn"] = "0"
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))})
from agent.extensions.skills.ocr import extract_text_from_frame
result = extract_text_from_frame({repr(frame_path)})
print(json.dumps(result, ensure_ascii=False))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return []
    for line in result.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return []

def extract_text_from_frame(frame_path: str) -> List[Dict[str, Any]]:
    """
    从单帧图像中提取文本。

    Args:
        frame_path (str): 帧图像路径。

    Returns:
        List[Dict]: 检测到的文本列表，每个包含文本、位置、置信度。
    """
    img = cv2.imread(frame_path)
    if img is None:
        return []

    results = _get_ocr().ocr(img, cls=True)
    if not results or not results[0]:
        return []

    texts = []
    for line in results[0]:
        bbox, (text, confidence) = line
        texts.append({
            'text': text,
            'bbox': bbox,
            'confidence': confidence
        })
    return texts

def extract_text_from_video_frames(frame_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    从视频帧列表中提取文本。

    Args:
        frame_paths (List[str]): 帧路径列表。

    Returns:
        Dict: 帧路径到文本列表的映射。
    """
    ocr_results = {}
    for path in frame_paths:
        ocr_results[path] = extract_text_from_frame(path)
    return ocr_results
