import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any

# 初始化PaddleOCR（支持中英文，可扩展多语言）
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 默认中文，可改为'en'或'multilingual'

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

    results = ocr.ocr(img, cls=True)
    if not results or not results[0]:
        return []

    texts = []
    for line in results[0]:
        bbox, (text, confidence) = line
        texts.append({
            'text': text,
            'bbox': bbox,  # 边界框坐标
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