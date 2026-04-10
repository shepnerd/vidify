# agent/chat.py
"""Interactive video chat CLI — explore a video through conversation.

Usage:
    python agent/main.py chat youtube "https://youtu.be/xxx"
    python agent/main.py chat local video.mp4 --chat-api-base https://api.openai.com/v1 --chat-model gpt-4o
"""
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.core.schemas import VideoAsset, FrameItem, FrameSet, FrameStrategy
from agent.extensions.skills.persist import load_analysis
from agent.extensions.models.vllm_openai_client import make_client, _is_qwen35
from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body

logger = logging.getLogger(__name__)

# ── Intent detection keywords ───────────────────────────────────────────

_VISUAL_KEYWORDS = {
    "look", "see", "show", "display", "screen", "board", "slide",
    "image", "picture", "color", "wear", "face", "background",
    "written", "text on", "equation", "diagram", "chart", "graph",
    "logo", "sign", "gesture", "scene", "appear", "visible",
    "看", "显示", "屏幕", "画面", "图", "公式", "黑板", "白板",
    "穿", "颜色", "长什么样", "幻灯片",
}

_EMOTION_KEYWORDS = {
    "mood", "happy", "sad", "angry", "emotion", "feel", "feeling",
    "expression", "excited", "nervous", "confident", "anxious",
    "enthusiastic", "passionate", "calm", "tense", "frustrated",
    "smile", "laugh", "cry", "frown",
    "情绪", "心情", "开心", "难过", "生气", "紧张", "激动",
    "高兴", "表情", "笑", "哭",
}


def _format_timestamp(sec: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    sec = int(sec)
    if sec >= 3600:
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}"
    m, s = divmod(sec, 60)
    return f"{m}:{s:02d}"


class VideoChat:
    """Interactive video Q&A session backed by cached analysis + on-demand skills."""

    def __init__(self, asset: VideoAsset, analysis: dict,
                 chat_client: OpenAI, chat_model: str,
                 vision_client: Optional[OpenAI] = None,
                 vision_model: Optional[str] = None):
        self.asset = asset
        self.analysis = analysis
        self.chat_client = chat_client
        self.chat_model = chat_model
        # Vision client may differ from chat (e.g. chat=Claude, vision=local vLLM)
        self.vision_client = vision_client or chat_client
        self.vision_model = vision_model or chat_model
        self.history: list[dict] = []
        self.visual_cache: dict[str, list[dict]] = {}  # key → frame captions
        self.console = Console(stderr=True)

        # Pre-extract commonly used data
        self.timeline = analysis.get("timeline", {})
        self.chapters = (self.timeline.get("chapters", [])
                         if isinstance(self.timeline, dict) else [])
        self.events = (self.timeline.get("events", [])
                       if isinstance(self.timeline, dict) else [])
        self.asr_segments = (analysis.get("asr", {}).get("segments", [])
                             if isinstance(analysis.get("asr"), dict) else [])
        self.content_meta = analysis.get("content_metadata") or {}
        self.video_meta = analysis.get("video", {})
        self.sufficiency = analysis.get("sufficiency", {})

    # ── Public API ──────────────────────────────────────────────────────

    def get_summary(self) -> str:
        """Build a structured summary from analysis data for LLM to present."""
        parts = []

        # Video info
        title = self.content_meta.get("title", "")
        uploader = self.content_meta.get("uploader", "")
        duration = self.video_meta.get("duration_sec", 0)
        if title:
            parts.append(f"**Title**: {title}")
        if uploader:
            parts.append(f"**Creator**: {uploader}")
        if duration:
            parts.append(f"**Duration**: {_format_timestamp(duration)}")

        desc = self.content_meta.get("description", "")
        if desc:
            parts.append(f"\n**Description** (excerpt):\n{desc[:500]}")

        # Timeline chapters
        if self.chapters:
            parts.append("\n**Chapters**:")
            for ch in self.chapters:
                start = _format_timestamp(ch.get("start", 0))
                end = _format_timestamp(ch.get("end", 0))
                parts.append(f"  [{start}-{end}] **{ch.get('title', '')}** — {ch.get('summary', '')}")

        # Sufficiency info
        if self.sufficiency.get("is_sufficient"):
            word_count = self.sufficiency.get("transcript_word_count", 0)
            coverage = self.sufficiency.get("asr_coverage_ratio", 0)
            parts.append(f"\n*Analysis based on audio transcript ({word_count} words, {coverage:.0%} coverage). "
                         f"Visual analysis available on demand.*")

        return "\n".join(parts)

    def answer(self, question: str, force_visual: bool = False) -> str:
        """Answer a user question, triggering on-demand skills when needed."""
        intents = self._classify_intent(question)
        if force_visual:
            intents.add("visual")

        # Find relevant time ranges
        relevant_ranges = self._find_relevant_segments(question)

        # Gather on-demand context
        extra_context = []

        if "visual" in intents and relevant_ranges:
            self.console.print("[dim]Triggering targeted visual analysis...[/dim]", highlight=False)
            visual_data = self._get_visual_context(relevant_ranges)
            if visual_data:
                extra_context.append(("Visual observations", visual_data))

        # Build context for LLM
        context = self._build_context(question, relevant_ranges, extra_context)

        # Call LLM
        answer_text = self._call_llm(question, context)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer_text})

        return answer_text

    def get_timeline_display(self) -> str:
        """Format timeline for display."""
        if not self.chapters:
            return "No timeline chapters available."
        lines = []
        for ch in self.chapters:
            start = _format_timestamp(ch.get("start", 0))
            end = _format_timestamp(ch.get("end", 0))
            lines.append(f"[{start}-{end}] {ch.get('title', '')}")
            if ch.get("summary"):
                lines.append(f"    {ch['summary']}")
        return "\n".join(lines)

    # ── Intent classification ───────────────────────────────────────────

    def _classify_intent(self, question: str) -> set:
        q_lower = question.lower()
        intents = {"factual"}
        if any(kw in q_lower for kw in _VISUAL_KEYWORDS):
            intents.add("visual")
        if any(kw in q_lower for kw in _EMOTION_KEYWORDS):
            intents.add("emotion")
        return intents

    # ── Relevant segment lookup ─────────────────────────────────────────

    def _find_relevant_segments(self, question: str) -> list[tuple[float, float]]:
        """Find time ranges relevant to the question using timeline chapters."""
        q_lower = question.lower()
        ranges = []

        # Check chapter titles and summaries for keyword overlap
        for ch in self.chapters:
            title = ch.get("title", "").lower()
            summary = ch.get("summary", "").lower()
            # Simple word overlap score
            q_words = set(re.findall(r'\w+', q_lower))
            ch_words = set(re.findall(r'\w+', title + " " + summary))
            overlap = len(q_words & ch_words - {"the", "a", "an", "is", "are", "was", "were",
                                                  "in", "on", "at", "to", "for", "of", "and",
                                                  "or", "but", "not", "this", "that", "what",
                                                  "how", "why", "when", "where", "who", "do",
                                                  "does", "did"})
            if overlap >= 1:
                ranges.append((ch.get("start", 0), ch.get("end", 0)))

        # If no chapter match, fall back to full video (first and last 20% of ASR)
        if not ranges and self.asr_segments:
            duration = self.video_meta.get("duration_sec", 0)
            if duration > 0:
                ranges.append((0, duration))

        return ranges[:5]  # Cap at 5 ranges

    # ── On-demand visual analysis ───────────────────────────────────────

    def _get_visual_context(self, ranges: list[tuple[float, float]]) -> list[dict]:
        """Extract and caption frames from specific time ranges (cached)."""
        results = []
        for start, end in ranges:
            cache_key = f"{start:.0f}-{end:.0f}"
            if cache_key in self.visual_cache:
                results.extend(self.visual_cache[cache_key])
                continue

            frames = self._extract_frames(start, end, max_frames=3)
            if not frames:
                continue

            # Caption frames using vision model
            captions = self._caption_frames(frames)
            self.visual_cache[cache_key] = captions
            results.extend(captions)

        return results

    def _extract_frames(self, start: float, end: float, max_frames: int = 3) -> list[dict]:
        """Extract frames from a time range using FFmpeg."""
        frames_dir = os.path.join(self.asset.cache_dir, "chat_frames")
        os.makedirs(frames_dir, exist_ok=True)

        duration = end - start
        if duration <= 0:
            duration = 10
        interval = max(duration / (max_frames + 1), 1)

        frames = []
        for i in range(max_frames):
            ts = start + interval * (i + 1)
            out_path = os.path.join(frames_dir, f"chat_{ts:.0f}.jpg")
            if not os.path.exists(out_path):
                cmd = [
                    "ffmpeg", "-y", "-ss", str(ts),
                    "-i", self.asset.local_path,
                    "-frames:v", "1",
                    "-vf", "scale=512:-1:force_original_aspect_ratio=decrease",
                    "-q:v", "2",
                    out_path,
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   check=True, timeout=30)
                except Exception:
                    continue
            if os.path.exists(out_path):
                frames.append({"ts": ts, "path": out_path})
        return frames

    def _caption_frames(self, frames: list[dict]) -> list[dict]:
        """Caption extracted frames using the vision model."""
        from agent.extensions.models.vllm_openai_client import chat_with_images
        results = []
        for frame in frames:
            try:
                caption = chat_with_images(
                    self.vision_client, self.vision_model,
                    "Describe what you see in this video frame in 1-2 sentences. "
                    "Focus on people's expressions, body language, and any text/graphics visible.",
                    [frame["path"]],
                    max_tokens=200, temperature=0.2,
                )
                results.append({
                    "ts": frame["ts"],
                    "ts_fmt": _format_timestamp(frame["ts"]),
                    "caption": caption,
                })
            except Exception as e:
                logger.warning("Failed to caption frame at %.0fs: %s", frame["ts"], e)
        return results

    # ── Context building ────────────────────────────────────────────────

    def _build_context(self, question: str, ranges: list[tuple[float, float]],
                       extra_context: list[tuple[str, list]]) -> str:
        """Assemble context from analysis data + on-demand results."""
        parts = []

        # Video metadata
        title = self.content_meta.get("title", "Unknown video")
        duration = _format_timestamp(self.video_meta.get("duration_sec", 0))
        parts.append(f"Video: {title} ({duration})")

        # Relevant ASR segments
        if ranges and self.asr_segments:
            relevant_text = []
            for start, end in ranges:
                for seg in self.asr_segments:
                    seg_start = seg.get("start", 0)
                    seg_end = seg.get("end", 0)
                    if seg_start >= start - 30 and seg_end <= end + 30:
                        ts = _format_timestamp(seg_start)
                        relevant_text.append(f"[{ts}] {seg.get('text', '')}")
            if relevant_text:
                parts.append("\nRelevant transcript:")
                parts.extend(relevant_text[:50])  # Cap transcript segments

        # Timeline chapters
        if self.chapters:
            parts.append("\nTimeline chapters:")
            for ch in self.chapters:
                start = _format_timestamp(ch.get("start", 0))
                parts.append(f"  [{start}] {ch.get('title', '')}: {ch.get('summary', '')}")

        # Extra context (visual, emotion)
        for label, items in extra_context:
            if items:
                parts.append(f"\n{label}:")
                for item in items:
                    if isinstance(item, dict):
                        ts = item.get("ts_fmt", "")
                        caption = item.get("caption", "")
                        parts.append(f"  [{ts}] {caption}")

        return "\n".join(parts)

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_llm(self, question: str, context: str) -> str:
        """Call the chat LLM with conversation history and context."""
        system_msg = (
            "You are a helpful video analysis assistant. You help users understand "
            "and explore video content through conversation. Answer based on the "
            "provided analysis context. If you need visual details that aren't "
            "available, say so — the user can request visual analysis.\n\n"
            "Be concise but informative. Use timestamps when referencing specific moments."
        )

        messages = [{"role": "system", "content": system_msg + f"\n\nVideo analysis context:\n{context}"}]

        # Add conversation history (last 10 turns)
        for turn in self.history[-10:]:
            messages.append(turn)

        messages.append({"role": "user", "content": question})

        kwargs = {}
        if _is_qwen35(self.chat_model):
            kwargs["extra_body"] = make_no_thinking_extra_body()

        try:
            resp = self.chat_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=1000,
                **kwargs,
            )
            text = resp.choices[0].message.content
            return strip_thinking(text).strip()
        except Exception as e:
            return f"Error calling LLM: {e}"


# ── REPL ────────────────────────────────────────────────────────────────

_HELP_TEXT = """\
**Commands:**
  `/summary`   — Show video summary
  `/timeline`  — Show timeline chapters
  `/visual`    — Force visual analysis for next question
  `/help`      — Show this help
  `/quit`      — Exit chat

Just type a question to ask about the video.\
"""


def run_chat_repl(chat: VideoChat):
    """Run the interactive chat REPL."""
    console = chat.console

    # Welcome banner
    title = chat.content_meta.get("title", "Video")
    duration = _format_timestamp(chat.video_meta.get("duration_sec", 0))
    console.print()
    console.print(Panel(
        f"[bold]{title}[/bold]\n"
        f"Duration: {duration}  |  "
        f"ASR: {chat.sufficiency.get('transcript_word_count', '?')} words  |  "
        f"Chapters: {len(chat.chapters)}",
        title="[bold cyan]VidCopilot Chat[/bold cyan]",
        border_style="cyan",
    ))

    # Show summary
    summary = chat.get_summary()
    if summary:
        console.print()
        console.print(Markdown(summary))

    console.print()
    console.print("[dim]Type a question or /help for commands. /quit to exit.[/dim]")
    console.print()

    force_visual_next = False

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue

        # Handle commands
        if question.startswith("/"):
            cmd = question.split()[0].lower()
            if cmd in ("/quit", "/q", "/exit"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/help":
                console.print(Markdown(_HELP_TEXT))
                continue
            elif cmd == "/summary":
                console.print(Markdown(chat.get_summary()))
                continue
            elif cmd == "/timeline":
                console.print(chat.get_timeline_display())
                continue
            elif cmd == "/visual":
                rest = question[len("/visual"):].strip()
                if rest:
                    # /visual <question> — answer with forced visual
                    question = rest
                    force_visual_next = True
                else:
                    force_visual_next = True
                    console.print("[dim]Visual analysis enabled for next question.[/dim]")
                    continue
            else:
                console.print(f"[dim]Unknown command: {cmd}. Type /help for commands.[/dim]")
                continue

        # Answer
        start_time = time.time()
        answer = chat.answer(question, force_visual=force_visual_next)
        elapsed = time.time() - start_time
        force_visual_next = False

        console.print()
        console.print(Markdown(answer))
        console.print(f"[dim]({elapsed:.1f}s)[/dim]")
        console.print()
