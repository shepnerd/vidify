"""Delta evaluator: compare before/after intervention results."""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.extensions.mra.schemas import BaseOutput, Delta, ReflectionOutput

logger = logging.getLogger(__name__)


def evaluate_delta(base_output: BaseOutput,
                   reflection: ReflectionOutput,
                   new_output: BaseOutput,
                   new_reflection: ReflectionOutput | None,
                   updated_evidence: Any,
                   intervention: dict) -> Delta:
    """Compare intervention before/after and evaluate reflection predictive validity."""

    review = intervention.get("review", {})
    target_claim_id = review.get("claim_id", "")

    # Answer changed?
    answer_changed = base_output.answer != new_output.answer

    # Confidence delta for target claim
    old_conf = _get_claim_conf(base_output, target_claim_id)
    new_conf = _get_claim_conf(new_output, target_claim_id)
    conf_delta = new_conf - old_conf

    # Expected change matched?
    expected = review.get("expected_change", "").lower()
    matched = _match_expected_change(expected, old_conf, new_conf, answer_changed)

    # Reflection became more specific?
    more_specific = False
    if new_reflection:
        old_spec = _reflection_specificity(reflection)
        new_spec = _reflection_specificity(new_reflection)
        more_specific = new_spec >= old_spec

    return Delta(
        answer_changed=answer_changed,
        claim_conf_delta=conf_delta,
        expected_change_matched=matched,
        reflection_became_more_specific=more_specific,
    )


def _get_claim_conf(output: BaseOutput, claim_id: str) -> float:
    for c in output.claims:
        if c.claim_id == claim_id:
            return c.confidence
    return output.answer_confidence


def _match_expected_change(expected: str,
                            old_conf: float,
                            new_conf: float,
                            answer_changed: bool) -> bool:
    """Simple keyword matching on expected_change vs actual changes."""
    if not expected:
        return False

    # Check directional keywords
    if any(kw in expected for kw in ("decrease", "lower", "reduce", "drop")):
        return new_conf < old_conf
    if any(kw in expected for kw in ("increase", "higher", "improve", "rise")):
        return new_conf > old_conf
    if any(kw in expected for kw in ("change", "different", "revise", "update")):
        return answer_changed or abs(new_conf - old_conf) > 0.1
    if any(kw in expected for kw in ("clarif", "disambigu", "confirm")):
        return abs(new_conf - old_conf) > 0.05

    # If confidence changed at all, consider it a partial match
    return abs(new_conf - old_conf) > 0.1


def _reflection_specificity(reflection: ReflectionOutput) -> float:
    """Score how specific a reflection is (more non-None fields = more specific)."""
    if not reflection.claim_reviews:
        return 0.0

    total = 0.0
    for review in reflection.claim_reviews:
        score = 0.0
        if review.time_span:
            score += 1.0
        if review.objects:
            score += 1.0
        if review.region_hint:
            score += 0.5
        if review.evidence_gap:
            score += 0.5
        if review.proposed_fix:
            score += 1.0
        if review.expected_change:
            score += 1.0
        total += score

    return total / len(reflection.claim_reviews)
