import base64, io, json, math
from PIL import Image
from openai import OpenAI

def _resize_limit(img: Image.Image, max_w=512, max_h=256) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img

def _img_to_data_url(path: str, max_w=512, max_h=256, fmt="JPEG", quality=85) -> str:
    img = Image.open(path).convert("RGB")
    img = _resize_limit(img, max_w=max_w, max_h=max_h)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def caption_frames(frames, model_name: str, base_url: str,
                   max_frames: int = 128, batch_size: int = 8,
                   max_w: int = 512, max_h: int = 256) -> "FrameSet":
    """
    frames: FrameSet(items=[FrameItem...])
    return: FrameSet with FrameItem.caption filled
    """
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    items = frames.items[:max_frames]
    id2item = {it.id: it for it in items}

    for bi in range(0, len(items), batch_size):
        batch = items[bi:bi+batch_size]

        content = [{"type": "text", "text": (
            "你将收到多张视频关键帧。请逐帧生成一句中文描述。\n"
            "要求：只输出严格 JSON 数组，每个元素含 {frame_id, caption}，不要输出多余文本。\n"
        )}]

        for it in batch:
            data_url = _img_to_data_url(it.path, max_w=max_w, max_h=max_h)
            content.append({"type": "image", "image": data_url})  # data-url image [2]
            content.append({"type": "text", "text": f"frame_id={it.id}"})

        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_completion_tokens=800,
        )
        text = resp.choices[0].message.content.strip()

        # 解析并对齐
        try:
            arr = json.loads(text)
            for obj in arr:
                fid = obj.get("frame_id")
                cap = obj.get("caption")
                if fid in id2item and cap:
                    id2item[fid].caption = cap
        except Exception:
            # MVP：失败则给本批打空，后续可做重试/回退到单帧
            for it in batch:
                if not it.caption:
                    it.caption = None

    frames.items[:len(items)] = items
    return frames