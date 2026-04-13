import ast
import json
import logging
import math
import re
from collections import Counter

from agent.extensions.models.vllm_openai_client import make_client, _is_qwen35
from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "they", "have", "about", "what",
    "when", "where", "which", "will", "would", "there", "their", "them", "then",
    "than", "into", "your", "you", "yeah", "well", "just", "like", "really", "very",
    "kind", "sort", "because", "could", "should", "being", "been", "also", "only",
    "over", "under", "more", "most", "some", "much", "many", "such", "through",
    "after", "before", "around", "again", "today", "still", "think", "going", "know",
    "it's", "dont", "didnt", "cant", "wont", "thats", "were", "our", "one", "two",
    "three", "four", "five", "first", "second", "third", "thing", "things", "make",
    "made", "gets", "get", "got", "going", "come", "comes", "coming", "look", "looks",
    "want", "wants", "wanted", "need", "needs", "used", "using", "use", "part", "parts",
    "good", "great", "best", "better", "maybe", "yes", "okay", "right", "mean", "meaning",
    "said", "says", "saying", "asked", "asking", "tell", "tells", "told", "talk", "talking",
    "talked", "question", "questions", "answer", "answers", "host", "guest", "episode",
    "show", "video", "clip", "section", "lot", "lots", "world", "people", "person", "years",
    "year", "time", "times", "day", "days", "work", "works", "working", "done", "does",
    "doing", "did", "has", "had", "having", "can", "see",
}


def _extract_json_candidate(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                return chunk
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text


def _coerce_num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _tokenize_terms(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", (text or "").lower())
    return [term for term in terms if term not in _STOPWORDS]


def _summarize_block_text(text: str, max_chars: int = 280) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip()
    return cut or text[:max_chars].strip()


def _keywords_for_text(text: str, top_k: int = 4) -> list[str]:
    counts = Counter(_tokenize_terms(text))
    return [term for term, _ in counts.most_common(top_k)]


def _make_block_title(text: str, keywords: list[str]) -> str:
    strong_keywords = [word for word in keywords if len(word) >= 4 and word not in _STOPWORDS]
    if len(strong_keywords) >= 2:
        return " / ".join(strong_keywords[:3])

    summary = _summarize_block_text(text, max_chars=96)
    summary = re.sub(r"^(yeah|well|so|now|look|okay|right|and|but)\b[\s,.:;-]*", "", summary, flags=re.IGNORECASE)
    words = summary.split()
    if len(words) > 10:
        summary = " ".join(words[:10])
    return summary.strip(" ,.:;-") or "untitled segment"


def _clean_title_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    text = re.sub(r"^(yeah|well|so|now|look|okay|right|and|but)\b[\s,.:;-]*", "", text, flags=re.IGNORECASE)
    text = text.strip(" ,.:;-")
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    return text or "Untitled Chapter"


def _normalize_timeline(raw) -> dict:
    if not isinstance(raw, dict):
        return {"chapters": [], "events": []}

    chapters = []
    for chapter in raw.get("chapters", []) or []:
        if not isinstance(chapter, dict):
            continue
        start = _coerce_num(chapter.get("start"))
        end = _coerce_num(chapter.get("end"), start)
        if end < start:
            end = start
        title = str(chapter.get("title") or "").strip()
        summary = str(chapter.get("summary") or "").strip()
        if not title and not summary:
            continue
        chapters.append({
            "start": start,
            "end": end,
            "title": title or summary[:80] or "untitled chapter",
            "summary": summary or title,
        })

    events = []
    for event in raw.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        start = _coerce_num(event.get("start"))
        end = _coerce_num(event.get("end"), start)
        if end < start:
            end = start
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        evidence = event.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        events.append({
            "start": start,
            "end": end,
            "text": text,
            "evidence": {
                "asr_segment_ids": list(evidence.get("asr_segment_ids", []) or []),
                "frame_ids": list(evidence.get("frame_ids", []) or []),
            },
        })

    chapters.sort(key=lambda item: (item["start"], item["end"], item["title"]))
    events.sort(key=lambda item: (item["start"], item["end"], item["text"]))
    return {"chapters": chapters, "events": events}


def _parse_timeline_response(text: str) -> dict | None:
    candidate = _extract_json_candidate(text)
    if not candidate:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
            if isinstance(parsed, dict):
                return _normalize_timeline(parsed)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return None


def _extract_model_text(resp) -> str:
    if not getattr(resp, "choices", None):
        return ""
    message = getattr(resp.choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                part = item.get("text")
                if part:
                    parts.append(str(part))
            elif item:
                parts.append(str(item))
        content = "\n".join(parts)
    return strip_thinking((content or "").strip())


def _repair_timeline_response(client, model_name: str, raw_text: str, direct_model: bool) -> str:
    repair_prompt = (
        "Repair the following timeline into strict JSON.\n"
        "Return only one JSON object with top-level keys chapters and events.\n"
        "Every chapter needs start, end, title, summary.\n"
        "Every event needs start, end, text, evidence.\n"
        "All start/end values must be numeric seconds.\n\n"
        f"{raw_text or ''}"
    )
    if direct_model:
        return client.chat_with_images(model_name, repair_prompt, [], max_tokens=1200, temperature=0.0) or ""

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}],
        temperature=0.0,
        max_completion_tokens=1200,
        response_format={"type": "json_object"},
    )
    return _extract_model_text(resp)


def _parse_label_response(text: str) -> list[dict] | None:
    candidate = _extract_json_candidate(text)
    if not candidate:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

        if isinstance(parsed, dict):
            parsed = parsed.get("chapters")
        if not isinstance(parsed, list):
            continue

        labels = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            title = _clean_title_text(item.get("title"))
            summary = _summarize_block_text(str(item.get("summary") or title), max_chars=180)
            labels.append({"index": index, "title": title, "summary": summary})
        if labels:
            return labels
    return None


def _build_transcript_blocks(transcript,
                             target_block_sec: float = 150.0,
                             min_block_sec: float = 45.0,
                             max_block_chars: int = 420) -> list[dict]:
    segments = list(getattr(transcript, "segments", []) or [])
    if not segments:
        return []

    blocks = []
    cur_segments = []
    cur_texts = []
    cur_ids = []
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_segments, cur_texts, cur_ids, cur_start, cur_end
        if not cur_segments:
            return
        text = " ".join(cur_texts).strip()
        keywords = _keywords_for_text(text)
        blocks.append({
            "id": f"block_{len(blocks):03d}",
            "start": float(cur_start),
            "end": float(cur_end),
            "title_hint": _make_block_title(text, keywords),
            "summary_hint": _summarize_block_text(text, max_chars=max_block_chars),
            "keywords": keywords,
            "word_count": sum(len((seg.text or "").split()) for seg in cur_segments),
            "asr_segment_ids": list(cur_ids),
        })
        cur_segments = []
        cur_texts = []
        cur_ids = []
        cur_start = None
        cur_end = None

    for segment in segments:
        if cur_start is None:
            cur_start = segment.start

        gap = 0.0 if cur_end is None else max(0.0, segment.start - cur_end)
        projected_end = segment.end
        projected_duration = projected_end - cur_start

        should_split = False
        if cur_segments:
            if gap >= 12.0 and (cur_end - cur_start) >= min_block_sec:
                should_split = True
            elif projected_duration >= target_block_sec:
                should_split = True
            elif len(" ".join(cur_texts)) >= max_block_chars and (cur_end - cur_start) >= min_block_sec:
                should_split = True

        if should_split:
            flush()
            cur_start = segment.start

        cur_segments.append(segment)
        cur_texts.append((segment.text or "").strip())
        cur_ids.append(segment.id)
        cur_end = segment.end

    flush()
    return blocks


def _estimate_timeline_shape(duration_sec: float, block_count: int, transcript_word_count: int) -> dict:
    if block_count <= 0:
        return {"target_chapters": 0, "target_events": 0}

    duration_min = max(1.0, duration_sec / 60.0)
    density = transcript_word_count / duration_min
    chapter_guess = max(
        math.ceil(duration_min / 6.0),
        math.ceil(block_count / 3.0),
    )
    if density >= 160:
        chapter_guess += 1

    event_guess = max(
        math.ceil(duration_min / 2.5),
        math.ceil(block_count * 0.75),
    )

    return {
        "target_chapters": _clamp(chapter_guess, 1, min(12, block_count)),
        "target_events": _clamp(event_guess, 2, min(24, block_count)),
    }


def _select_frame_evidence(frames, start: float, end: float, max_items: int = 3) -> list[str]:
    items = []
    for frame in getattr(frames, "items", []) or []:
        if start <= getattr(frame, "ts", -1) <= end:
            items.append(getattr(frame, "id", None))
        if len(items) >= max_items:
            break
    return [item for item in items if item]


def _heuristic_timeline_from_blocks(blocks: list[dict], frames, target_shape: dict) -> dict:
    if not blocks:
        return {"chapters": [], "events": []}

    target_chapters = _clamp(target_shape.get("target_chapters", 1), 1, len(blocks))
    target_events = _clamp(target_shape.get("target_events", 2), 1, len(blocks))

    chapters = []
    blocks_per_chapter = math.ceil(len(blocks) / target_chapters)
    for start_idx in range(0, len(blocks), blocks_per_chapter):
        group = blocks[start_idx:start_idx + blocks_per_chapter]
        if not group:
            continue
        merged_text = " ".join(block["summary_hint"] for block in group).strip()
        merged_keywords = []
        for block in group:
            merged_keywords.extend(block.get("keywords", []))
        title = _make_block_title(merged_text, list(dict.fromkeys(merged_keywords)))
        chapters.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "title": title,
            "summary": _summarize_block_text(merged_text, max_chars=220) or title,
        })

    if len(chapters) > target_chapters:
        chapters = chapters[:target_chapters - 1] + [{
            "start": chapters[target_chapters - 1]["start"],
            "end": chapters[-1]["end"],
            "title": chapters[target_chapters - 1]["title"],
            "summary": chapters[target_chapters - 1]["summary"],
        }]

    event_indices = sorted({
        int(round(i * (len(blocks) - 1) / max(1, target_events - 1)))
        for i in range(target_events)
    })
    events = []
    for idx in event_indices:
        block = blocks[idx]
        text = _summarize_block_text(block["summary_hint"] or block["title_hint"], max_chars=180)
        events.append({
            "start": block["start"],
            "end": block["end"],
            "text": text,
            "evidence": {
                "asr_segment_ids": block.get("asr_segment_ids", [])[:6],
                "frame_ids": _select_frame_evidence(frames, block["start"], block["end"]),
            },
        })

    return _normalize_timeline({"chapters": chapters, "events": events})


def _overlapping_blocks(blocks: list[dict], start: float, end: float) -> list[dict]:
    overlaps = []
    for block in blocks:
        if block["end"] < start or block["start"] > end:
            continue
        overlaps.append(block)
    return overlaps


def _refine_timeline_labels(client, model_name: str, timeline: dict, blocks: list[dict], direct_model: bool) -> dict:
    chapters = list((timeline or {}).get("chapters") or [])
    if not chapters:
        return timeline

    chapter_context = []
    for index, chapter in enumerate(chapters):
        source_blocks = _overlapping_blocks(blocks, chapter["start"], chapter["end"])
        keywords = []
        for block in source_blocks:
            keywords.extend(block.get("keywords", []))
        chapter_context.append({
            "index": index,
            "start": chapter["start"],
            "end": chapter["end"],
            "current_title": chapter.get("title", ""),
            "current_summary": chapter.get("summary", ""),
            "keywords": list(dict.fromkeys(keywords))[:8],
            "source_outline": [
                {
                    "title_hint": block.get("title_hint"),
                    "summary_hint": _summarize_block_text(block.get("summary_hint", ""), max_chars=140),
                }
                for block in source_blocks[:4]
            ],
        })

    prompt = (
        "Rewrite the chapter labels for a transcript-first video timeline.\n"
        "Use semantic topics, not keyword lists. Titles should be concise, human-readable, and specific.\n"
        "Preserve chapter count, order, and time ranges. Return strict JSON only:\n"
        "{\"chapters\":[{\"index\":0,\"title\":\"...\",\"summary\":\"...\"}]}\n\n"
        f"{json.dumps({'chapters': chapter_context}, ensure_ascii=False)}"
    )

    if direct_model:
        raw = client.chat_with_images(model_name, prompt, [], max_tokens=1000, temperature=0.1) or ""
    else:
        kwargs = {}
        if _is_qwen35(model_name):
            kwargs["extra_body"] = make_no_thinking_extra_body()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            temperature=0.1,
            max_completion_tokens=1000,
            response_format={"type": "json_object"},
            **kwargs,
        )
        raw = _extract_model_text(resp)

    labels = _parse_label_response(raw)
    if not labels:
        return timeline

    by_index = {item["index"]: item for item in labels}
    refined_chapters = []
    for index, chapter in enumerate(chapters):
        label = by_index.get(index)
        if not label:
            refined_chapters.append({
                **chapter,
                "title": _clean_title_text(chapter.get("title")),
                "summary": _summarize_block_text(chapter.get("summary") or chapter.get("title"), max_chars=180),
            })
            continue
        refined_chapters.append({
            **chapter,
            "title": label["title"],
            "summary": label["summary"],
        })

    return {
        **timeline,
        "chapters": refined_chapters,
    }


def _build_bottom_up_payload(metadata, transcript, frames, content_metadata=None) -> tuple[dict, dict]:
    blocks = _build_transcript_blocks(transcript)
    transcript_word_count = sum(len((seg.text or "").split()) for seg in getattr(transcript, "segments", []) or [])
    target_shape = _estimate_timeline_shape(metadata.duration_sec, len(blocks), transcript_word_count)

    frame_snips = [
        {"ts": f.ts, "id": f.id, "cap": f.caption}
        for f in (getattr(frames, "items", []) or [])
        if getattr(f, "caption", None)
    ][:96]

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

    payload = {
        "task": (
            "Build a bottom-up video timeline. Use transcript semantics first. "
            "For podcast, interview, presentation, lecture, or panel content, rely primarily "
            "on transcript blocks and only use visual evidence as support."
        ),
        "rules": {
            "target_chapters": target_shape["target_chapters"],
            "target_events": target_shape["target_events"],
            "prefer_semantic_structure": True,
            "merge_adjacent_blocks_into_higher_level_chapters": True,
            "output": {
                "chapters": [{"start": "sec", "end": "sec", "title": "str", "summary": "str"}],
                "events": [{"start": "sec", "end": "sec", "text": "str", "evidence": {"asr_segment_ids": [], "frame_ids": []}}],
            },
        },
        "video_metadata": metadata.model_dump(),
        "transcript_outline": {
            "block_count": len(blocks),
            "word_count": transcript_word_count,
            "blocks": blocks[:24],
        },
        "frames": frame_snips,
    }
    if content_context:
        payload["content_info"] = content_context

    fallback = _heuristic_timeline_from_blocks(blocks, frames, target_shape)
    return payload, fallback, blocks


def build_timeline(metadata, transcript, frames, model_name: str, base_url: str,
                   content_metadata=None,
                   direct_model: bool = False, model_path: str = None, tokenizer_path: str = None) -> dict:
    payload, fallback, blocks = _build_bottom_up_payload(metadata, transcript, frames, content_metadata=content_metadata)

    if not payload["transcript_outline"]["blocks"] and not payload["frames"]:
        return {"chapters": [], "events": []}

    if direct_model:
        from agent.extensions.models.direct_model_loader import make_direct_client
        client = make_direct_client(model_path, tokenizer_path)
        prompt = (
            "Generate a structured video timeline from this bottom-up outline. "
            "Return strict JSON only.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        text = client.chat_with_images(model_name, prompt, [], max_tokens=1400, temperature=0.1) or ""
    else:
        client = make_client(base_url)
        kwargs = {}
        if _is_qwen35(model_name):
            kwargs["extra_body"] = make_no_thinking_extra_body()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "Generate a structured video timeline from this bottom-up outline. "
                        "Use transcript semantics first. Return strict JSON only.\n\n"
                        f"{json.dumps(payload, ensure_ascii=False)}"
                    ),
                }],
            }],
            temperature=0.1,
            max_completion_tokens=1400,
            response_format={"type": "json_object"},
            **kwargs,
        )
        text = _extract_model_text(resp)

    parsed = _parse_timeline_response(text)
    if parsed is not None and (parsed["chapters"] or parsed["events"]):
        return _refine_timeline_labels(client, model_name, parsed, blocks, direct_model=direct_model)

    logger.warning("Timeline JSON parse failed or empty; requesting repair pass")
    try:
        repaired = _repair_timeline_response(client, model_name, text, direct_model=direct_model)
        parsed = _parse_timeline_response(repaired)
        if parsed is not None and (parsed["chapters"] or parsed["events"]):
            return _refine_timeline_labels(client, model_name, parsed, blocks, direct_model=direct_model)
    except Exception as exc:
        logger.warning("Timeline repair pass failed: %s", exc)

    logger.warning("Falling back to heuristic timeline")
    return _refine_timeline_labels(client, model_name, fallback, blocks, direct_model=direct_model)
