# agent/skills/timeline_builder.py
import json
from openai import OpenAI
from agent.models.vllm_openai_client import make_client
from agent.models.direct_model_loader import make_direct_client

def build_timeline(metadata, transcript, frames, model_name: str, base_url: str,
                   direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> dict:
    if direct_model:
        client = make_direct_client(model_path, tokenizer_path)
    else:
        client = make_client(base_url)

    # 控制上下文：只取前 N 个 frame caption
    frame_snips = [{"ts": f.ts, "id": f.id, "cap": f.caption} for f in frames.items if f.caption][:128]
    asr_snips = [{"start": s.start, "end": s.end, "id": s.id, "text": s.text} for s in transcript.segments][:400]

    prompt = {
        "task": "生成视频结构化时间线",
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

    if direct_model:
        text = client.chat_with_images(model_name, json.dumps(prompt, ensure_ascii=False), [], max_tokens=1200, temperature=0.2)
    else:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]}],
            temperature=0.2,
            max_completion_tokens=1200,
        )
        text = resp.choices[0].message.content.strip()
    return json.loads(text)