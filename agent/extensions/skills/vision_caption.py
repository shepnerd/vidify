import json, math, subprocess
from agent.extensions.models.vllm_openai_client import make_client, _is_qwen35
from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body
from agent.extensions.utils import (
    get_video_duration, split_video_segment, make_video_content, img_to_data_url,
)


def _make_direct_client(model_path: str = None, tokenizer_path: str = None):
    from agent.extensions.models.direct_model_loader import make_direct_client
    return make_direct_client(model_path, tokenizer_path)

def supports_video(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen" in name or "video" in name

def caption_frames(frames, model_name: str, base_url: str,
                   max_frames: int = 128, batch_size: int = 8,
                   max_w: int = 256, max_h: int = 144,
                   direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> "FrameSet":
    """
    frames: FrameSet(items=[FrameItem...])
    return: FrameSet with FrameItem.caption filled
    """
    if direct_model:
        client = _make_direct_client(model_path, tokenizer_path)
    else:
        client = make_client(base_url)

    if direct_model:
        client = _make_direct_client(model_path, tokenizer_path)
        # For direct model, process one by one
        batch_size = 1
    else:
        client = make_client(base_url)

    items = frames.items[:max_frames]
    id2item = {it.id: it for it in items}
    previous_summary = ""

    for bi in range(0, len(items), batch_size):
        batch = items[bi:bi+batch_size]

        if direct_model:
            # Single frame processing
            it = batch[0]
            prompt = f"请生成一句中文描述。{f' 前面的描述：{previous_summary}' if previous_summary else ''}"
            image_urls = [f"file://{it.path}"]
            text = client.chat_with_images(model_name, prompt, image_urls, max_tokens=100, temperature=0.2)
            # Assume text is the caption
            try:
                id2item[it.id].caption = text.strip()
                previous_summary += text.strip() + " "
            except:
                id2item[it.id].caption = None
        else:
            content = [{"type": "text", "text": (
                f"你将收到多张视频关键帧。请逐帧生成一句中文描述。{f' 前面的描述：{previous_summary}' if previous_summary else ''}\n"
                "要求：只输出严格 JSON 数组，每个元素含 {frame_id, caption}，不要输出多余文本。\n"
            )}]

            for it in batch:
                data_url = img_to_data_url(it.path, max_w=max_w, max_h=max_h)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
                content.append({"type": "text", "text": f"frame_id={it.id}"})

            kwargs = {}
            if _is_qwen35(model_name):
                kwargs["extra_body"] = make_no_thinking_extra_body()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_completion_tokens=800,
                **kwargs,
            )
            text = strip_thinking(resp.choices[0].message.content.strip())

            # 解析并对齐
            try:
                arr = json.loads(text)
                for obj in arr:
                    fid = obj.get("frame_id")
                    cap = obj.get("caption")
                    if fid in id2item and cap:
                        id2item[fid].caption = cap
                        previous_summary += cap + " "
            except Exception:
                # MVP：失败则给本批打空，后续可做重试/回退到单帧
                for it in batch:
                    if not it.caption:
                        it.caption = None

    frames.items[:len(items)] = items
    return frames

def caption_video(video_path: str, model_name: str, base_url: str, max_duration: int = 60,
                  direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> list:
    """
    Caption video by segments if too long.
    Returns list of dict: {"start": float, "end": float, "caption": str}
    """
    # Get duration
    duration = get_video_duration(video_path)

    if duration <= max_duration:
        # Direct process
        if direct_model:
            client = _make_direct_client(model_path, tokenizer_path)
            prompt = "请生成视频的中文描述。"
            text = client.chat_with_images(model_name, prompt, [f"file://{video_path}"], max_tokens=500, temperature=0.2)
        else:
            client = make_client(base_url)
            content = [{"type": "text", "text": "请生成视频的中文描述。"}, make_video_content(video_path)]
            kwargs = {}
            if _is_qwen35(model_name):
                kwargs["extra_body"] = make_no_thinking_extra_body()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_completion_tokens=500,
                **kwargs,
            )
            text = strip_thinking(resp.choices[0].message.content.strip())
        return [{"start": 0, "end": duration, "caption": text}]

    else:
        # Split and process segments
        segments = []
        start = 0.0
        summaries = []
        while start < duration:
            end = min(start + max_duration, duration)
            segment_path = f"{video_path}_seg_{int(start)}_{int(end)}.mp4"
            split_video_segment(video_path, start, end - start, segment_path)
            # Process segment
            prev_summary = summaries[-1] if summaries else ""
            if direct_model:
                client = _make_direct_client(model_path, tokenizer_path)
                prompt = f"请生成这段视频的中文描述。{f' 前面的总结：{prev_summary}' if prev_summary else ''}"
                text = client.chat_with_images(model_name, prompt, [f"file://{segment_path}"], max_tokens=300, temperature=0.2)
            else:
                client = make_client(base_url)
                content = [{"type": "text", "text": f"请生成这段视频的中文描述。{f' 前面的总结：{prev_summary}' if prev_summary else ''}"},
                           make_video_content(segment_path)]
                kwargs = {}
                if _is_qwen35(model_name):
                    kwargs["extra_body"] = make_no_thinking_extra_body()
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                    max_completion_tokens=300,
                    **kwargs,
                )
                text = strip_thinking(resp.choices[0].message.content.strip())
            segments.append({"start": start, "end": end, "caption": text})
            summaries.append(text)
            start = end
        return segments

def caption_video_as_frameset(video_path: str, model_name: str, base_url: str, max_duration: int = 60,
                              direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> "FrameSet":
    from agent.core.schemas import FrameItem, FrameSet
    segments = caption_video(video_path, model_name, base_url, max_duration, direct_model, model_path, tokenizer_path)
    items = []
    for i, seg in enumerate(segments):
        path = f"{video_path}_seg_{int(seg['start'])}_{int(seg['end'])}.mp4" if seg['start'] > 0 else video_path
        item = FrameItem(id=f"seg_{i}", ts=seg["start"], path=path, caption=seg["caption"])
        items.append(item)
    from agent.core.schemas import FrameStrategy
    return FrameSet(items=items, strategy=FrameStrategy(type="scene", params={"source": "video_caption"}))
