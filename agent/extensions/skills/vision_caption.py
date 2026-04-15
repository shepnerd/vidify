import json
from agent.extensions.models.vllm_openai_client import make_client, _is_qwen35
from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body
from agent.extensions.utils import (
    get_video_duration, split_video_segment, make_video_content, img_to_data_url,
)


def _make_direct_client(model_path: str = None, tokenizer_path: str = None):
    from agent.extensions.models.direct_model_loader import make_direct_client
    return make_direct_client(model_path, tokenizer_path)


def _fmt_seconds(sec: float | None) -> str:
    if sec is None:
        return "unknown"
    return f"{float(sec):.1f}s"


def _frame_sampling_text(frames, total_items: int) -> str:
    strategy_type = getattr(getattr(frames, "strategy", None), "type", None)
    if strategy_type == "fps":
        fps = frames.strategy.params.get("fps")
        if fps is not None:
            return f"按约 {float(fps):.2f} fps 均匀抽样，共 {total_items} 帧。"
    if strategy_type == "scene":
        return f"按场景变化抽样，共 {total_items} 帧。"
    return f"共抽样 {total_items} 帧。"


def _build_frame_batch_prompt(frames, batch, previous_summary: str = "",
                              video_duration_sec: float = None,
                              sampled_frame_count: int = None) -> str:
    total_items = sampled_frame_count or len(frames.items)
    timestamps = "；".join(f"{it.id}={_fmt_seconds(it.ts)}" for it in batch)
    prompt = [
        "你将收到多张来自同一视频的关键帧。",
        f"源视频总时长约 {_fmt_seconds(video_duration_sec)}。",
        _frame_sampling_text(frames, total_items),
        f"本批包含 {len(batch)} 帧，时间戳如下：{timestamps}。",
        "请结合时间顺序逐帧生成一句中文描述。",
    ]
    if previous_summary:
        prompt.append(f"前面的描述：{previous_summary}")
    prompt.append("要求：只输出严格 JSON 数组，每个元素含 {frame_id, caption}，不要输出多余文本。")
    return "\n".join(prompt)


def _build_single_frame_prompt(frame, previous_summary: str = "",
                               video_duration_sec: float = None,
                               sampled_frame_count: int = None) -> str:
    prompt = [
        "请为这张视频关键帧生成一句中文描述。",
        f"该帧时间戳约 {_fmt_seconds(frame.ts)}。",
        f"源视频总时长约 {_fmt_seconds(video_duration_sec)}。",
    ]
    if sampled_frame_count:
        prompt.append(f"该视频本次共抽样 {sampled_frame_count} 帧。")
    if previous_summary:
        prompt.append(f"前面的描述：{previous_summary}")
    return " ".join(prompt)


def _build_video_prompt(base_instruction: str,
                        clip_duration_sec: float,
                        source_duration_sec: float = None,
                        segment_start_sec: float = None,
                        segment_end_sec: float = None,
                        previous_summary: str = "") -> str:
    lines = [
        base_instruction,
        f"当前输入片段时长约 {_fmt_seconds(clip_duration_sec)}。",
    ]
    if source_duration_sec is not None:
        lines.append(f"源视频总时长约 {_fmt_seconds(source_duration_sec)}。")
    if segment_start_sec is not None and segment_end_sec is not None:
        lines.append(
            f"当前片段对应源视频时间范围约 {_fmt_seconds(segment_start_sec)} 到 {_fmt_seconds(segment_end_sec)}。"
        )
    if previous_summary:
        lines.append(f"前面的总结：{previous_summary}")
    return " ".join(lines)


def supports_video(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen" in name or "video" in name

def caption_frames(frames, model_name: str, base_url: str,
                   max_frames: int = 128, batch_size: int = 8,
                   max_w: int = 256, max_h: int = 144,
                   direct_model: bool = False, model_path: str = None, tokenizer_path: str = None,
                   video_duration_sec: float = None, sampled_frame_count: int = None) -> "FrameSet":
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
            prompt = _build_single_frame_prompt(
                it,
                previous_summary=previous_summary,
                video_duration_sec=video_duration_sec,
                sampled_frame_count=sampled_frame_count or len(items),
            )
            image_urls = [f"file://{it.path}"]
            text = client.chat_with_images(model_name, prompt, image_urls, max_tokens=100, temperature=0.2)
            # Assume text is the caption
            try:
                id2item[it.id].caption = text.strip()
                previous_summary += text.strip() + " "
            except Exception:
                id2item[it.id].caption = None
        else:
            content = [{"type": "text", "text": _build_frame_batch_prompt(
                frames,
                batch,
                previous_summary=previous_summary,
                video_duration_sec=video_duration_sec,
                sampled_frame_count=sampled_frame_count or len(items),
            )}]

            for it in batch:
                data_url = img_to_data_url(it.path, max_w=max_w, max_h=max_h)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
                content.append({"type": "text", "text": f"frame_id={it.id}, timestamp={_fmt_seconds(it.ts)}"})

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
                  direct_model: bool = False, model_path: str = None, tokenizer_path: str = None,
                  source_duration_sec: float = None) -> list:
    """
    Caption video by segments if too long.
    Returns list of dict: {"start": float, "end": float, "caption": str}
    """
    # Get duration
    duration = get_video_duration(video_path)
    source_duration_sec = duration if source_duration_sec is None else source_duration_sec

    if duration <= max_duration:
        # Direct process
        if direct_model:
            client = _make_direct_client(model_path, tokenizer_path)
            prompt = _build_video_prompt(
                "请生成视频的中文描述。",
                clip_duration_sec=duration,
                source_duration_sec=source_duration_sec,
                segment_start_sec=0.0,
                segment_end_sec=duration,
            )
            text = client.chat_with_images(model_name, prompt, [f"file://{video_path}"], max_tokens=500, temperature=0.2)
        else:
            client = make_client(base_url)
            content = [{"type": "text", "text": _build_video_prompt(
                "请生成视频的中文描述。",
                clip_duration_sec=duration,
                source_duration_sec=source_duration_sec,
                segment_start_sec=0.0,
                segment_end_sec=duration,
            )}, make_video_content(video_path)]
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
                prompt = _build_video_prompt(
                    "请生成这段视频的中文描述。",
                    clip_duration_sec=end - start,
                    source_duration_sec=source_duration_sec,
                    segment_start_sec=start,
                    segment_end_sec=end,
                    previous_summary=prev_summary,
                )
                text = client.chat_with_images(model_name, prompt, [f"file://{segment_path}"], max_tokens=300, temperature=0.2)
            else:
                client = make_client(base_url)
                content = [{"type": "text", "text": _build_video_prompt(
                    "请生成这段视频的中文描述。",
                    clip_duration_sec=end - start,
                    source_duration_sec=source_duration_sec,
                    segment_start_sec=start,
                    segment_end_sec=end,
                    previous_summary=prev_summary,
                )},
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
                              direct_model: bool = False, model_path: str = None, tokenizer_path: str = None,
                              source_duration_sec: float = None) -> "FrameSet":
    from agent.core.schemas import FrameItem, FrameSet
    segments = caption_video(
        video_path,
        model_name,
        base_url,
        max_duration,
        direct_model,
        model_path,
        tokenizer_path,
        source_duration_sec=source_duration_sec,
    )
    items = []
    for i, seg in enumerate(segments):
        path = f"{video_path}_seg_{int(seg['start'])}_{int(seg['end'])}.mp4" if seg['start'] > 0 else video_path
        item = FrameItem(id=f"seg_{i}", ts=seg["start"], path=path, caption=seg["caption"])
        items.append(item)
    from agent.core.schemas import FrameStrategy
    return FrameSet(items=items, strategy=FrameStrategy(type="scene", params={"source": "video_caption"}))
