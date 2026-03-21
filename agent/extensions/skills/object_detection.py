from ultralytics import YOLO
import cv2
from typing import List, Dict, Any

# 初始化YOLO模型（使用预训练的YOLOv8）
model = YOLO('yolov8n.pt')  # nano版本，轻量级；可改为'yolov8s.pt'等

def detect_objects_in_frame(frame_path: str) -> List[Dict[str, Any]]:
    """
    在单帧中检测物体。

    Args:
        frame_path (str): 帧图像路径。

    Returns:
        List[Dict]: 检测到的物体列表，每个包含类别、置信度、边界框。
    """
    results = model(frame_path)
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