import asyncio
import concurrent.futures
from typing import List, Dict, Any, Callable
from agent.orchestrator import run  # 假设orchestrator有run函数

def process_video_batch(video_uris: List[str], mode: str, cfg: Dict[str, Any]) -> List[Any]:
    """
    批量处理多个视频。

    Args:
        video_uris (List[str]): 视频URI列表。
        mode (str): 处理模式。
        cfg (Dict): 配置字典。

    Returns:
        List[Any]: 处理结果列表。
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 限制并发数
        futures = [executor.submit(process_single_video, uri, mode, cfg) for uri in video_uris]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing video: {e}")
                results.append(None)
    return results

def process_single_video(uri: str, mode: str, cfg: Dict[str, Any]) -> Any:
    """
    处理单个视频。

    Args:
        uri (str): 视频URI。
        mode (str): 模式。
        cfg (Dict): 配置。

    Returns:
        Any: 处理结果。
    """
    from agent.skills.video_io import load_video
    asset = load_video(cfg.get('source_type', 'youtube'), uri, cfg.get('cache_root', './cache'))
    return run(asset, mode, cfg)

# 队列管理（简单实现）
class VideoQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def add_video(self, uri: str, mode: str, cfg: Dict[str, Any]):
        await self.queue.put((uri, mode, cfg))

    async def process_queue(self):
        while True:
            uri, mode, cfg = await self.queue.get()
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, process_single_video, uri, mode, cfg)
                print(f"Processed {uri}: {result}")
            except Exception as e:
                print(f"Error processing {uri}: {e}")
            finally:
                self.queue.task_done()