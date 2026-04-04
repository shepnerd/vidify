# agent/skills/timeline_builder.py
import json
from openai import OpenAI
from agent.extensions.models.vllm_openai_client import make_client, _is_qwen35
from agent.extensions.models.direct_model_loader import make_direct_client
from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body

def build_timeline(metadata, transcript, frames, model_name: str, base_url: str,
                   content_metadata=None,
                   direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> dict:
    if direct_model:
        client = make_direct_client(model_path, tokenizer_path)
    else:
        client = make_client(base_url)

    frame_snips = [{"ts": f.ts, "id": f.id, "cap": f.caption} for f in frames.items if f.caption][:128]
    asr_snips = [{"start": s.start, "end": s.end, "id": s.id, "text": s.text} for s in transcript.segments][:400]

    # Build content metadata context if available
    content_context = None
    if content_metadata:
        cm = content_metadata if isinstance(content_metadata, dict) else content_metadata.model_dump()
        content_context = {
            k: v for k, v in {
                "title": cm.get("title"),
                "description": (cm.get("description") or "")[:500] or None,
                "uploader": cm.get("uploader"),
                "tags": cm.get("tags"),
                "categories": cm.get("categories"),
            }.items() if v
        }

    # Adjust task description based on available data
    if not frame_snips and asr_snips:
        task_desc = ("生成视频结构化时间线。注意：没有视觉帧数据，请主要根据语音转录文本和视频元数据来构建时间线。")
    elif frame_snips and not asr_snips:
        task_desc = ("生成视频结构化时间线。注意：没有语音转录数据，请主要根据视觉帧描述来构建时间线。")
    else:
        task_desc = "生成视频结构化时间线"

    payload = {
        "task": task_desc,
        "requirements": {
            "output": {
                "chapters": [{"start": "sec", "end": "sec", "title": "str", "summary": "str"}],
                "events": [{"start": "sec", "end": "sec", "text": "str", "evidence": {"asr_segment_ids": [], "frame_ids": []}}]
            }
        },
        "video_metadata": metadata.model_dump(),
        "frames": frame_snips,
        "asr_segments": asr_snips
    }
    if content_context:
        payload["content_info"] = content_context

    if direct_model:
        prompt = f"请根据以下信息生成视频结构化时间线。输出必须是严格的JSON格式，不要包含其他文本。\n\n{json.dumps(payload, ensure_ascii=False)}"
        text = client.chat_with_images(model_name, prompt, [], max_tokens=1200, temperature=0.2)
    else:
        kwargs = {}
        if _is_qwen35(model_name):
            kwargs["extra_body"] = make_no_thinking_extra_body()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}],
            temperature=0.2,
            max_completion_tokens=1200,
            **kwargs,
        )
        text = strip_thinking(resp.choices[0].message.content.strip())

    # Try to extract JSON from the response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return {"chapters": [], "events": []}
