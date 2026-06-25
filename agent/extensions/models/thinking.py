# agent/extensions/models/thinking.py
"""Utilities for handling Qwen3.5 thinking mode responses.

Qwen3.5 models output ``<think>...</think>`` reasoning blocks by default
before the actual answer.  For VidCopilot's pipeline (structured JSON
outputs, captions, timeline generation) we almost always want the final
answer only.  This module provides helpers to strip or extract thinking
content.

Thinking mode can be disabled at request time by setting
``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``
when using vLLM/SGLang, but stripping post-hoc is more robust across
backends.
"""
import re
from typing import Tuple, Optional

# Matches the entire <think>...</think> block (possibly multiline).
_THINK_RE = re.compile(r"<think>\n?(.*?)</think>\n?\n?", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and return the answer only.

    >>> strip_thinking("<think>\\nLet me reason...\\n</think>\\n\\nThe answer is 42.")
    'The answer is 42.'
    >>> strip_thinking("No thinking here.")
    'No thinking here.'
    """
    return _THINK_RE.sub("", text).strip()


def extract_thinking(text: str) -> Tuple[Optional[str], str]:
    """Split a response into (thinking, answer).

    Returns ``(thinking_content, answer)`` where *thinking_content* is
    ``None`` if no ``<think>`` block was found.

    >>> thinking, answer = extract_thinking("<think>\\nreasoning\\n</think>\\n\\nresult")
    >>> thinking
    'reasoning'
    >>> answer
    'result'
    """
    m = _THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip()
        answer = _THINK_RE.sub("", text).strip()
        return thinking, answer
    return None, text.strip()


def make_no_thinking_extra_body() -> dict:
    """Return the ``extra_body`` dict that disables thinking for vLLM/SGLang.

    Usage::

        client.chat.completions.create(
            ...,
            extra_body=make_no_thinking_extra_body(),
        )
    """
    return {"chat_template_kwargs": {"enable_thinking": False}}
