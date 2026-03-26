import cv2
from typing import List, Dict, Any
from agent.config import get_model_path

_model = None

def _get_model():
    """Lazy-init YOLO so import doesn't crash if model file is missing."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(get_model_path("yolov8n.pt"))
    return _model

def detect_objects_in_frame(frame_path: str) -> List[Dict[str, Any]]:
    """
    在单帧中检测物体。

    Args:
        frame_path (str): 帧图像路径。

    Returns:
        List[Dict]: 检测到的物体列表，每个包含类别、置信度、边界框。
    """
    results = _get_model()(frame_path)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
            })
    return detections

def detect_objects_in_video_frames(frame_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    在视频帧列表中检测物体。

    Args:
        frame_paths (List[str]): 帧路径列表。

    Returns:
        Dict: 帧路径到检测结果的映射。
    """
    detection_results = {}
    for path in frame_paths:
        detection_results[path] = detect_objects_in_frame(path)
    return detection_results
