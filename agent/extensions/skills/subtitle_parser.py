# agent/skills/subtitle_parser.py
import re
from typing import Optional, List
from agent.core.schemas import Transcript, ASRSegment, SubtitleTrack


def _parse_vtt_timestamp(ts: str) -> float:
    """Parse VTT timestamp 'HH:MM:SS.mmm' or 'MM:SS.mmm' to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


def _parse_srt_timestamp(ts: str) -> float:
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds."""
    ts = ts.strip().replace(",", ".")
    return _parse_vtt_timestamp(ts)


def _strip_tags(text: str) -> str:
    """Remove HTML/VTT tags like <b>, <i>, <c.color>, etc."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _dedup_segments(segments: List[ASRSegment], overlap_threshold: float = 0.5) -> List[ASRSegment]:
    """Merge segments with overlapping timestamps and similar text."""
    if not segments:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        # If this segment overlaps significantly with the previous one
        overlap = max(0, min(prev.end, seg.end) - max(prev.start, seg.start))
        prev_dur = prev.end - prev.start
        if prev_dur > 0 and overlap / prev_dur > overlap_threshold:
            # Keep the longer text
            if len(seg.text) > len(prev.text):
                merged[-1] = ASRSegment(
                    id=prev.id,
                    start=min(prev.start, seg.start),
                    end=max(prev.end, seg.end),
                    text=seg.text,
                    confidence=prev.confidence,
                )
            else:
                merged[-1] = ASRSegment(
                    id=prev.id,
                    start=min(prev.start, seg.start),
                    end=max(prev.end, seg.end),
                    text=prev.text,
                    confidence=prev.confidence,
                )
        else:
            merged.append(seg)
    return merged


def parse_vtt(vtt_path: str, confidence: float = 1.0) -> Transcript:
    """Parse a WebVTT file into a Transcript."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = []
    # Match VTT cues: timestamp line --> timestamp, followed by text lines
    cue_pattern = re.compile(
        r"(\d[\d:\.]+)\s*-->\s*(\d[\d:\.]+)[^\n]*\n((?:(?!\n\n|\d[\d:\.]+\s*-->).+\n?)+)",
        re.MULTILINE,
    )
    for i, m in enumerate(cue_pattern.finditer(content)):
        start = _parse_vtt_timestamp(m.group(1))
        end = _parse_vtt_timestamp(m.group(2))
        text = _strip_tags(m.group(3)).replace("\n", " ").strip()
        if text:
            segments.append(ASRSegment(
                id=f"sub_{i:06d}", start=start, end=end,
                text=text, confidence=confidence,
            ))

    segments = _dedup_segments(segments)
    # Re-number after dedup
    for i, seg in enumerate(segments):
        seg.id = f"sub_{i:06d}"
    return Transcript(segments=segments, language=None)


def parse_srt(srt_path: str, confidence: float = 1.0) -> Transcript:
    """Parse an SRT file into a Transcript."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = []
    # SRT blocks: index\nstart --> end\ntext\n\n
    block_pattern = re.compile(
        r"\d+\s*\n(\d[\d:,\.]+)\s*-->\s*(\d[\d:,\.]+)\s*\n((?:(?!\n\n|\d+\s*\n\d).+\n?)+)",
        re.MULTILINE,
    )
    for i, m in enumerate(block_pattern.finditer(content)):
        start = _parse_srt_timestamp(m.group(1))
        end = _parse_srt_timestamp(m.group(2))
        text = _strip_tags(m.group(3)).replace("\n", " ").strip()
        if text:
            segments.append(ASRSegment(
                id=f"sub_{i:06d}", start=start, end=end,
                text=text, confidence=confidence,
            ))

    segments = _dedup_segments(segments)
    for i, seg in enumerate(segments):
        seg.id = f"sub_{i:06d}"
    return Transcript(segments=segments, language=None)


def load_best_subtitle(subtitle_tracks: List[SubtitleTrack]) -> Optional[Transcript]:
    """Pick the best subtitle track and parse it into a Transcript.

    Priority: manual > auto. Among same source type, prefer en > zh > others.
    """
    if not subtitle_tracks:
        return None

    lang_priority = {"en": 0, "zh": 1, "ja": 2, "ko": 3}

    def sort_key(t: SubtitleTrack):
        source_score = 0 if t.source == "manual" else 1
        lang_score = lang_priority.get(t.language.split("-")[0], 99)
        return (source_score, lang_score)

    tracks_sorted = sorted(subtitle_tracks, key=sort_key)
    best = tracks_sorted[0]

    confidence = 1.0 if best.source == "manual" else 0.8
    if best.format == "vtt":
        transcript = parse_vtt(best.path, confidence=confidence)
    elif best.format == "srt":
        transcript = parse_srt(best.path, confidence=confidence)
    else:
        return None

    # Try to infer language from the track
    if best.language and best.language != "unknown":
        transcript.language = best.language.split("-")[0]

    return transcript if transcript.segments else None
