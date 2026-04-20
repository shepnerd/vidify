"""Pydantic v2 models for the Meta-Reflective Auditor."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error taxonomy & intervention vocabulary (MVP)
# ---------------------------------------------------------------------------

ErrorType = Literal[
    "visual_ambiguity",
    "temporal_boundary_error",
    "tracking_failure",
    "ocr_ambiguity",
    "language_prior_bias",
]

InterventionType = Literal[
    "dense_frame_resample",
    "zoom_region",
    "rerun_tracker_or_detector",
    "evidence_only_rereason",
]

ReviewStatus = Literal["possibly_wrong", "uncertain_review", "likely_correct"]

RiskLevel = Literal["low", "medium", "high"]

MRAStatus = Literal[
    "accepted_without_intervention",
    "no_intervention_possible",
    "accept_revised_answer",
    "accept_original_with_higher_confidence",
    "keep_uncertain",
    "abstain_or_escalate",
]


# ---------------------------------------------------------------------------
# Base output protocol — wraps any workflow result
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    claim_id: str
    type: str = "general"
    text: str
    span: Optional[List[float]] = None
    objects: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    support_refs: Dict[str, Any] = Field(default_factory=dict)


class BaseOutput(BaseModel):
    answer: str
    answer_confidence: float = 0.5
    claims: List[Claim] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_workflow_result(cls, result: dict, mode: str,
                            query: str | None = None) -> "BaseOutput":
        """Adapt a raw workflow result dict into the MRA base-output protocol."""
        if mode == "ask":
            return cls._from_ask_result(result, query)
        return cls._from_analysis_result(result)

    @classmethod
    def _from_analysis_result(cls, result: dict) -> "BaseOutput":
        timeline = result.get("timeline", "")
        if isinstance(timeline, dict):
            answer = json.dumps(timeline, ensure_ascii=False)[:500]
        elif isinstance(timeline, list):
            answer = " ".join(
                ch.get("summary", ch.get("title", ""))
                for ch in timeline if isinstance(ch, dict)
            )[:500]
        else:
            answer = str(timeline)[:500]

        frames = result.get("frames", {})
        items = frames.get("items", []) if isinstance(frames, dict) else []

        # Synthesize a single claim from the timeline
        claims = []
        if answer:
            duration = result.get("video", {}).get("duration_sec")
            claims.append(Claim(
                claim_id="c0",
                type="summary",
                text=answer[:200],
                span=[0, duration] if duration else None,
                confidence=0.6,
                support_refs={
                    "frames": [it.get("id", f"f_{i}") for i, it in enumerate(items[:8])],
                },
            ))

        sampled = [it.get("id") for it in items if it.get("id")]
        trace = {
            "sampled_frames": sampled[:32],
            "used_modules": ["video_llm"],
        }
        return cls(answer=answer, answer_confidence=0.6, claims=claims, trace=trace)

    @classmethod
    def _from_ask_result(cls, result: dict, query: str | None = None) -> "BaseOutput":
        inner = result.get("result", {})
        if isinstance(inner, dict):
            answer = inner.get("answer", str(inner))
        else:
            answer = str(inner)

        evidence = []
        if isinstance(inner, dict):
            evidence = inner.get("evidence", [])

        hits = result.get("hits", [])
        spans = [(h.get("start", 0), h.get("end", 0)) for h in hits if "start" in h]

        claims = [Claim(
            claim_id="c0",
            type="qa_answer",
            text=answer[:200],
            span=[spans[0][0], spans[-1][1]] if spans else None,
            confidence=0.65,
            support_refs={"hits": hits[:5], "evidence": evidence[:5]},
        )]

        trace = {"query": query, "used_modules": ["rag_faiss", "video_llm"]}
        return cls(answer=answer, answer_confidence=0.65, claims=claims, trace=trace)


# ---------------------------------------------------------------------------
# Reflection output
# ---------------------------------------------------------------------------

class ClaimReview(BaseModel):
    claim_id: str
    status: ReviewStatus = "uncertain_review"
    error_type: ErrorType = "visual_ambiguity"
    time_span: Optional[List[float]] = None
    region_hint: Optional[str] = None
    objects: List[str] = Field(default_factory=list)
    evidence_gap: str = ""
    proposed_fix: List[InterventionType] = Field(default_factory=list)
    expected_change: str = ""


class ReflectionOutput(BaseModel):
    overall_self_confidence: float = 0.5
    claim_reviews: List[ClaimReview] = Field(default_factory=list)
    global_risk: RiskLevel = "medium"


# ---------------------------------------------------------------------------
# Evidence bundle
# ---------------------------------------------------------------------------

class EvidenceBundle(BaseModel):
    frame_meta: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tracks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    detection_results: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    ocr_spans: List[Dict[str, Any]] = Field(default_factory=list)
    event_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    support_trace: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audit & delta
# ---------------------------------------------------------------------------

class AuditResult(BaseModel):
    groundedness: float = 0.0
    attribution_validity: float = 0.0
    fix_validity: float = 0.0
    meta_trust: float = 0.0
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_intervention: Optional[Dict[str, Any]] = None


class Delta(BaseModel):
    answer_changed: bool = False
    claim_conf_delta: float = 0.0
    expected_change_matched: bool = False
    reflection_became_more_specific: bool = False


# ---------------------------------------------------------------------------
# Final MRA result
# ---------------------------------------------------------------------------

class MRAResult(BaseModel):
    status: MRAStatus
    final_output: BaseOutput
    base_output: BaseOutput
    reflection: Optional[ReflectionOutput] = None
    new_reflection: Optional[ReflectionOutput] = None
    audit: Optional[AuditResult] = None
    delta: Optional[Delta] = None
    intervention_type: Optional[str] = None
    audit_log: Dict[str, Any] = Field(default_factory=dict)
    raw_workflow_result: Dict[str, Any] = Field(default_factory=dict)
