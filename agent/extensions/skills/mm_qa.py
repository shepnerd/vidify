# agent/skills/multimodal_qa.py
from agent.extensions.models.vllm_openai_client import make_client

def video_frames_qa(frame_items, question: str,
                    model_name: str, base_url: str,
                    max_images: int = 12,
                    temperature: float = 0.2,
                    max_tokens: int = 800,
                    extra_body: dict | None = None) -> str:
    """
    frame_items: List[FrameItem] (must have .path)
    A方案：image_url.url 直接传本地路径；需要 vLLM serve 时设置 allowed-local-media-path [1]
    """
    client = make_client(base_url)
    imgs = frame_items[:max_images]

    content = [{"type": "text", "text": (
        "你将收到多张视频关键帧。请根据图片回答问题。"
        "如果无法从图片确定，请明确说明不确定，并指出缺少哪些信息。\n"
        f"问题：{question}"
    )}]
    for it in imgs:
        content.append({"type": "image_url", "image_url": {"url": it.path}})
        content.append({"type": "text", "text": f"(frame_id={it.id}, ts={it.ts:.1f}s)"})

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        extra_body=extra_body or {}  # vLLM 可透传 top_k 等 [1]
    )
    return resp.choices[0].message.content