import cv2
import asyncio
from typing import Callable, Any, Tuple, Dict
from agent.extensions.skills.vision_caption import caption_frame  # 假设有caption函数
from agent.extensions.skills.object_detection import detect_objects_in_frame
from agent.config import load_models_config, load_workflows_config

def process_live_stream(source: str = None, stream_url: str = None, callback: Callable[[Dict[str, Any]], None] = None,
                        resolution: Tuple[int, int] = None, fps: int = None, heavy_interval: int = None):
    """
    处理直播流或摄像头。

    Args:
        source (str): 'stream' 或 'webcam'。
        stream_url (str): 流URL，如果source='stream'。
        callback (Callable): 处理每帧后的回调函数。
        resolution (Tuple): 分辨率 (width, height)。
        fps (int): 处理频率，每秒帧数。
        heavy_interval (int): 每多少帧用大模型。
    """
    # Load configurations
    models_config = load_models_config()
    workflows_config = load_workflows_config()
    
    # Use config defaults if not provided
    if source is None:
        source = workflows_config.get('live_stream', {}).get('source', 'webcam')
    if resolution is None:
        res_config = workflows_config.get('live_stream', {}).get('resolution', [640, 480])
        resolution = tuple(res_config)
    if fps is None:
        fps = workflows_config.get('live_stream', {}).get('fps', 1)
    if heavy_interval is None:
        heavy_interval = workflows_config.get('live_stream', {}).get('heavy_interval', 5)
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif source == 'stream':
        cap = cv2.VideoCapture(stream_url)
    else:
        raise ValueError("Invalid source")

    if not cap.isOpened():
        raise ValueError("Cannot open source")

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 按fps处理
        if frame_count % (30 // fps) == 0:  # 假设30fps视频
            frame_path = f"/tmp/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

            # SlowFast策略
            if frame_count % heavy_interval == 0:
                analysis = analyze_frame_heavy(frame_path)
            else:
                analysis = analyze_frame_light(frame_path)
            callback(analysis)

    cap.release()

def analyze_frame_heavy(frame_path: str) -> Dict[str, Any]:
    """
    用大模型分析帧（7B）。
    """
    models_config = load_models_config()
    heavy_model = models_config.get('mllm', {}).get('heavy', {})
    # TODO: Use heavy model for captioning
    caption = caption_frame(frame_path, model_name=heavy_model.get('model_name'), base_url=heavy_model.get('base_url'))
    objects = detect_objects_in_frame(frame_path)
    # 添加OCR等
    from agent.extensions.skills.ocr import extract_text_from_frame
    ocr = extract_text_from_frame(frame_path)
    return {
        'caption': caption,
        'objects': objects,
        'ocr': ocr,
        'model': 'heavy'
    }

def analyze_frame_light(frame_path: str) -> Dict[str, Any]:
    """
    用轻量模型分析帧（<1B MLLM + OCR + 轻量检测）。
    """
    models_config = load_models_config()
    light_model = models_config.get('mllm', {}).get('light', {})
    # TODO: Use light model for captioning
    caption = "Light caption placeholder"  # TODO: 实现轻量caption，使用light_model
    # 轻量对象检测：使用YOLO nano或简化
    objects = []  # TODO: 实现轻量检测
    # 轻量OCR：使用快速OCR
    from agent.extensions.skills.ocr import extract_text_from_frame
    ocr = extract_text_from_frame(frame_path)  # 假设轻量
    return {
        'caption': caption,
        'objects': objects,
        'ocr': ocr,
        'model': 'light'
    }

async def process_live_stream_async(source: str = None, stream_url: str = None, callback: Callable[[Dict[str, Any]], None] = None,
                                    resolution: Tuple[int, int] = None, fps: int = None, heavy_interval: int = None):
    """
    异步处理直播流。
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_live_stream, source, stream_url, callback, resolution, fps, heavy_interval)