# agent/skills/content_sufficiency.py
import re
from agent.core.schemas import Transcript, VideoMetadata, ContentSufficiency


def _count_words(text: str) -> int:
    """Count words. For CJK text, count characters as words."""
    # Detect if text is primarily CJK
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", text))
    if cjk_chars > len(text) * 0.3:
        # CJK-dominant: count characters (excluding spaces/punctuation)
        return cjk_chars + len(re.findall(r"[a-zA-Z]+", text))
    # Latin-dominant: count space-separated words
    return len(text.split())


def assess_sufficiency(
    transcript: Transcript,
    metadata: VideoMetadata,
    min_coverage_ratio: float = 0.3,
    min_word_count: int = 50,
    force_visual: bool = False,
) -> ContentSufficiency:
    """Assess whether transcript provides enough information to skip visual processing.

    Pure heuristics, no LLM call. Fast and cheap.
    """
    duration = metadata.duration_sec if metadata.duration_sec > 0 else 1.0

    # Compute ASR coverage: total spoken time / video duration
    total_spoken = sum(max(0, s.end - s.start) for s in transcript.segments)
    coverage = min(total_spoken / duration, 1.0)

    # Compute word count
    all_text = " ".join(s.text for s in transcript.segments)
    word_count = _count_words(all_text)

    has_subs = any(
        s.confidence is not None and s.confidence >= 0.8
        for s in transcript.segments
    )
    has_content_meta = (
        metadata.content is not None
        and bool(metadata.content.title)
    )

    if force_visual:
        is_sufficient = False
        reason = "Visual processing forced by configuration."
    elif not transcript.segments:
        is_sufficient = False
        reason = "No transcript available."
    elif coverage >= min_coverage_ratio and word_count >= min_word_count:
        is_sufficient = True
        parts = []
        parts.append(f"ASR covers {coverage:.0%} of video duration")
        parts.append(f"{word_count} words in transcript")
        if has_subs:
            parts.append("subtitles available")
        if has_content_meta:
            parts.append("video metadata available")
        reason = "Transcript is sufficient: " + ", ".join(parts) + "."
    else:
        reasons = []
        if coverage < min_coverage_ratio:
            reasons.append(f"low speech coverage ({coverage:.0%} < {min_coverage_ratio:.0%})")
        if word_count < min_word_count:
            reasons.append(f"low word count ({word_count} < {min_word_count})")
        is_sufficient = False
        reason = "Transcript insufficient: " + ", ".join(reasons) + "."

    return ContentSufficiency(
        asr_coverage_ratio=round(coverage, 3),
        transcript_word_count=word_count,
        has_subtitles=has_subs,
        has_content_metadata=has_content_meta,
        is_sufficient=is_sufficient,
        reason=reason,
    )
