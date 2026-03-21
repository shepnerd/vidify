import cv2
import asyncio
from typing import Callable, Any
from agent.skills.frame_sampler import sample_frames  # 假设有帧采样函数
from agent.skills.vision_caption import caption_frame  # 假设有caption函数

def process_live_stream(stream_url: str, callback: Callable[[Dict[str, Any]], None]):
    """
    处理直播流。

    Args:
        stream_url (str): 流URL。
        callback (Callable): 处理每帧后的回调函数。
    """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError("Cannot open stream")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 每隔一定帧处理一次
        if frame_count % 30 == 0:  # 例如每30帧处理一次
            # 保存帧到临时文件或内存
            frame_path = f"/tmp/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

            # 分析帧
            analysis = analyze_frame(frame_path)
            callback(analysis)

    cap.release()

def analyze_frame(frame_path: str) -> Dict[str, Any]:
    """
    分析单帧。

    Args:
        frame_path (str): 帧路径。

    Returns:
        Dict: 分析结果。
    """
    # 示例：caption和对象检测
    from agent.skills.vision_caption import caption_frame
    from agent.skills.object_detection import detect_objects_in_frame

    caption = caption_frame(frame_path)
    objects = detect_objects_in_frame(frame_path)

    return {
        'caption': caption,
        'objects': objects
    }

async def process_live_stream_async(stream_url: str, callback: Callable[[Dict[str, Any]], None]):
    """
    异步处理直播流。
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_live_stream, stream_url, callback)