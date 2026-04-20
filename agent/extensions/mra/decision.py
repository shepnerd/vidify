"""Final accept / revise / abstain decision logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent.extensions.mra.schemas import (
    AuditResult, BaseOutput, Delta, MRAResult, MRAStatus,
    ReflectionOutput,
)

logger = logging.getLogger(__name__)


def finalize_decision(base_output: BaseOutput,
                      new_output: BaseOutput | None,
                      reflection: ReflectionOutput | None,
                      new_reflection: ReflectionOutput | None,
                      audit: AuditResult,
                      delta: Delta | None,
                      intervention_type: str | None,
                      config: dict,
                      raw_workflow_result: dict | None = None) -> MRAResult:
    """Produce the final MRA decision based on meta-trust and delta."""

    accept_threshold = config.get("meta_trust_accept", 0.75)
    uncertain_threshold = config.get("meta_trust_uncertain", 0.45)
    mts = audit.meta_trust

    # Adjust meta_trust based on delta (if intervention was run)
    if delta is not None:
        if delta.expected_change_matched:
            mts = min(1.0, mts + 0.1)
        if delta.reflection_became_more_specific:
            mts = min(1.0, mts + 0.05)

    final_output = new_output if new_output is not None else base_output

    if mts >= accept_threshold:
        if new_output is not None and delta and delta.answer_changed:
            status: MRAStatus = "accept_revised_answer"
        else:
            status = "accept_original_with_higher_confidence"
    elif mts >= uncertain_threshold:
        status = "keep_uncertain"
        final_output.answer_confidence = min(final_output.answer_confidence, 0.5)
    else:
        status = "abstain_or_escalate"
        final_output.answer_confidence = min(final_output.answer_confidence, 0.3)

    audit_log = {
        "meta_trust_raw": audit.meta_trust,
        "meta_trust_adjusted": mts,
        "groundedness": audit.groundedness,
        "attribution_validity": audit.attribution_validity,
        "fix_validity": audit.fix_validity,
        "status": status,
    }
    if delta:
        audit_log["delta"] = delta.model_dump()
    if intervention_type:
        audit_log["intervention_type"] = intervention_type

    logger.info("MRA decision: status=%s, meta_trust=%.2f (raw=%.2f)",
                status, mts, audit.meta_trust)

    return MRAResult(
        status=status,
        final_output=final_output,
        base_output=base_output,
        reflection=reflection,
        new_reflection=new_reflection,
        audit=audit,
        delta=delta,
        intervention_type=intervention_type,
        audit_log=audit_log,
        raw_workflow_result=raw_workflow_result or {},
    )


def finalize_early(base_output: BaseOutput,
                   reflection: ReflectionOutput | None,
                   audit: AuditResult | None,
                   status: MRAStatus,
                   raw_workflow_result: dict | None = None) -> MRAResult:
    """Quick finalization when no intervention is needed."""
    audit_log: Dict[str, Any] = {"status": status}
    if audit:
        audit_log.update({
            "meta_trust": audit.meta_trust,
            "groundedness": audit.groundedness,
            "attribution_validity": audit.attribution_validity,
            "fix_validity": audit.fix_validity,
        })

    return MRAResult(
        status=status,
        final_output=base_output,
        base_output=base_output,
        reflection=reflection,
        audit=audit,
        audit_log=audit_log,
        raw_workflow_result=raw_workflow_result or {},
    )
