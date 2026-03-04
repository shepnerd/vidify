# agent/skills/highlights.py
import json
from openai import OpenAI
from agent.schemas import HighlightClip

def detect_highlights(transcript, timeline: dict, model_name: str, base_url: str,
                      max_clips: int = 5) -> list[HighlightClip]:
    client = OpenAI(base_url=base_url, api_key="EMPTY")

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
    arr = json.loads(resp.choices[0].message.content.strip())
    return [HighlightClip(start=o["start"], end=o["end"], reason=o["reason"], output_path="") for o in arr]