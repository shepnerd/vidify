# agent/skills/highlights.py
import json
from openai import OpenAI
from agent.extensions.models.vllm_openai_client import make_client
from agent.extensions.models.direct_model_loader import make_direct_client
from agent.core.schemas import HighlightClip

def detect_highlights(transcript, timeline: dict, model_name: str, base_url: str,
                      max_clips: int = 5,
                      direct_model: bool = False,
                      model_path: str = None,
                      tokenizer_path: str = None) -> list[HighlightClip]:
    if direct_model:
        client = make_direct_client(model_path, tokenizer_path)
        payload = {
            "task": "从时间线与转录中选取高光片段（信息密度高/关键结论/转折点）",
            "constraints": {"max_clips": max_clips, "min_len_sec": 10, "max_len_sec": 90},
            "timeline": timeline,
            "asr_segments": [s.model_dump() for s in transcript.segments[:800]],
            "output_schema": [{"start": "sec", "end": "sec", "reason": "str"}]
        }
        text = client.chat_with_images(model_name, json.dumps(payload, ensure_ascii=False), [], max_tokens=800, temperature=0.2)
    else:
        client = make_client(base_url)
        payload = {
            "task": "从时间线与转录中选取高光片段（信息密度高/关键结论/转折点）",
            "constraints": {"max_clips": max_clips, "min_len_sec": 10, "max_len_sec": 90},
            "timeline": timeline,
            "asr_segments": [s.model_dump() for s in transcript.segments[:800]],
            "output_schema": [{"start": "sec", "end": "sec", "reason": "str"}]
        }

        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}],
            temperature=0.2,
            max_completion_tokens=800,
        )
        text = resp.choices[0].message.content.strip()
    arr = json.loads(text)
    return [HighlightClip(start=o["start"], end=o["end"], reason=o["reason"], output_path="") for o in arr]