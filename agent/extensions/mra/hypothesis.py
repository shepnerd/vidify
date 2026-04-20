"""Competing hypotheses generation and scoring."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.extensions.mra.schemas import BaseOutput, EvidenceBundle, ReflectionOutput


def build_simple_hypotheses(reflection: ReflectionOutput,
                            base_output: BaseOutput,
                            evidence: EvidenceBundle) -> List[Dict[str, Any]]:
    """Generate competing hypotheses for each reviewed claim.

    MVP: two hypotheses per claim — "original is correct" vs "reflection is right".
    """
    hypotheses = []
    claim_map = {c.claim_id: c for c in base_output.claims}

    for review in reflection.claim_reviews:
        claim = claim_map.get(review.claim_id)
        claim_conf = claim.confidence if claim else 0.5

        # H1: the original answer is correct
        hypotheses.append({
            "id": f"H_{review.claim_id}_correct",
            "type": "original_correct",
            "claim_id": review.claim_id,
            "confidence": claim_conf,
            "reasoning": f"Original claim has confidence {claim_conf:.2f}",
        })

        # H2: the reflection is right, the claim is wrong
        ref_conf = reflection.overall_self_confidence
        hypotheses.append({
            "id": f"H_{review.claim_id}_wrong",
            "type": "reflection_supported",
            "claim_id": review.claim_id,
            "confidence": (1.0 - claim_conf) * ref_conf,
            "reasoning": (f"Reflection flags {review.error_type} "
                          f"with self-confidence {ref_conf:.2f}"),
            "error_type": review.error_type,
        })

    return hypotheses
