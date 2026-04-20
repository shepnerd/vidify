"""LLM-based structured reflection generation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from agent.extensions.mra.schemas import BaseOutput, EvidenceBundle, ReflectionOutput
from agent.extensions.mra.prompts import build_reflection_prompt
from agent.extensions.mra.evidence_collector import summarize_evidence

logger = logging.getLogger(__name__)


def generate_reflection(query: str | None,
                        base_output: BaseOutput,
                        evidence: EvidenceBundle,
                        llm_model: str,
                        llm_base_url: str,
                        config: dict) -> ReflectionOutput:
    """Call the LLM to produce a structured reflection on the base output."""

    from agent.extensions.models.vllm_openai_client import (
        make_client, _is_qwen35,
    )
    from agent.extensions.models.thinking import (
        strip_thinking, make_no_thinking_extra_body,
    )

    evidence_summary = summarize_evidence(evidence)
    prompt = build_reflection_prompt(
        query, base_output.model_dump(), evidence_summary, config,
    )

    client = make_client(llm_base_url)
    kwargs: Dict[str, Any] = {}
    if _is_qwen35(llm_model):
        kwargs["extra_body"] = make_no_thinking_extra_body()

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=800,
            **kwargs,
        )
        raw_text = strip_thinking(resp.choices[0].message.content.strip())
    except Exception as exc:
        logger.warning("Reflection LLM call failed: %s", exc)
        return _fallback_reflection()

    return _parse_reflection(raw_text)


def _parse_reflection(raw_text: str) -> ReflectionOutput:
    """Parse and validate the LLM JSON output into a ReflectionOutput."""
    # Try to extract JSON from possibly wrapped text
    text = raw_text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return ReflectionOutput.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse reflection JSON: %s — raw: %.200s", exc, text)
        return _fallback_reflection()


def _fallback_reflection() -> ReflectionOutput:
    """Return a safe fallback when parsing or LLM call fails."""
    return ReflectionOutput(
        overall_self_confidence=0.5,
        claim_reviews=[],
        global_risk="medium",
    )
