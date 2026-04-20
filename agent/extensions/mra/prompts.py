"""Prompt templates for the MRA reflection and evidence-only re-reasoning."""

from __future__ import annotations

import json
from typing import Any, Dict


def build_reflection_prompt(query: str | None,
                            base_output_dict: dict,
                            evidence_summary: str,
                            config: dict) -> str:
    """Build the structured-reflection prompt for the LLM."""

    answer = base_output_dict.get("answer", "")[:300]
    claims_text = ""
    for c in base_output_dict.get("claims", []):
        cid = c.get("claim_id", "?")
        ctxt = c.get("text", "")[:120]
        conf = c.get("confidence", "?")
        span = c.get("span", [])
        claims_text += f"  - {cid}: \"{ctxt}\" (confidence={conf}, span={span})\n"

    error_types = json.dumps(config.get("supported_error_types", []))
    fix_types = json.dumps(config.get("supported_interventions", []))

    prompt = f"""You are a self-assessment module for a video understanding agent.

Given:
1. User query: {query or '(general analysis)'}
2. Agent answer: {answer}
3. Agent claims:
{claims_text}
4. Available evidence summary:
{evidence_summary}

Your job: review whether the answer may be wrong and output structured JSON.

Rules:
- Choose error_type ONLY from: {error_types}
- Choose proposed_fix ONLY from: {fix_types}
- Each suspicious claim must include: claim_id, status, error_type, time_span, objects, evidence_gap, proposed_fix, expected_change
- status must be one of: "possibly_wrong", "uncertain_review", "likely_correct"
- global_risk must be one of: "low", "medium", "high"
- If unsure, mark status as "uncertain_review"
- Do NOT write any text outside the JSON object

Output exactly this JSON schema:
{{
  "overall_self_confidence": <float 0-1>,
  "claim_reviews": [
    {{
      "claim_id": "<string>",
      "status": "<possibly_wrong|uncertain_review|likely_correct>",
      "error_type": "<from allowed list>",
      "time_span": [<start_sec>, <end_sec>] or null,
      "region_hint": "<string or null>",
      "objects": ["<obj_id>", ...],
      "evidence_gap": "<what evidence is missing>",
      "proposed_fix": ["<from allowed list>", ...],
      "expected_change": "<what should change after fix>"
    }}
  ],
  "global_risk": "<low|medium|high>"
}}"""
    return prompt


def build_evidence_only_rereason_prompt(query: str | None,
                                        claim: dict,
                                        evidence_subset: dict) -> str:
    """Build a prompt that forces grounded reasoning from evidence only."""

    claim_text = claim.get("text", "")
    claim_id = claim.get("claim_id", "?")
    evidence_str = json.dumps(evidence_subset, ensure_ascii=False, default=str)[:2000]

    prompt = f"""You are a strictly grounded reasoning module.

You are given:
- Question: {query or '(verify the following claim)'}
- Target claim ({claim_id}): "{claim_text}"
- Verified evidence (ONLY use this):
{evidence_str}

Rules:
- Use ONLY the provided evidence to evaluate the claim
- Do NOT infer facts not directly supported by the evidence
- If evidence is insufficient, answer "uncertain"
- You MUST cite which evidence items you rely on

Output JSON:
{{
  "answer": "<your assessment of the claim>",
  "confidence": <float 0-1>,
  "evidence_used": ["<list of evidence keys you relied on>"],
  "unsupported_parts": ["<aspects of the claim with no evidence>"]
}}"""
    return prompt
