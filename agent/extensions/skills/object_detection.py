import logging
import cv2
from typing import List, Dict, Any
from agent.config import get_model_path

logger = logging.getLogger(__name__)

_model = None
_ultralytics_available = None

def _check_ultralytics():
    global _ultralytics_available
    if _ultralytics_available is None:
        try:
            import ultralytics  # noqa: F401
            _ultralytics_available = True
        except ImportError:
            _ultralytics_available = False
    return _ultralytics_available

def _get_model():
    """Lazy-init YOLO so import doesn't crash if ultralytics is not installed."""
    global _model
    if _model is None:
        if not _check_ultralytics():
            raise ImportError(
                "ultralytics is required for object detection but is not installed. "
                "Install it with: pip install vidify[detection]"
            )
        import os
        # Prevent ultralytics from reaching the network in offline environments.
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        from ultralytics import settings as ul_settings
        # Disable analytics and auto-update checks (avoids network calls)
        ul_settings.update({"sync": False})
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
