"""Core MRA rule-based scoring: groundedness, attribution validity, fix validity."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.extensions.mra.schemas import (
    AuditResult, BaseOutput, ClaimReview, EvidenceBundle, ReflectionOutput,
)
from agent.extensions.mra.hypothesis import build_simple_hypotheses

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility matrices
# ---------------------------------------------------------------------------

# claim_type -> error_type -> base compatibility score
_CLAIM_ERROR_COMPAT: Dict[str, Dict[str, float]] = {
    "event": {
        "visual_ambiguity": 0.8, "temporal_boundary_error": 0.9,
        "tracking_failure": 0.85, "ocr_ambiguity": 0.2, "language_prior_bias": 0.7,
    },
    "qa_answer": {
        "visual_ambiguity": 0.8, "temporal_boundary_error": 0.7,
        "tracking_failure": 0.6, "ocr_ambiguity": 0.5, "language_prior_bias": 0.85,
    },
    "summary": {
        "visual_ambiguity": 0.7, "temporal_boundary_error": 0.6,
        "tracking_failure": 0.5, "ocr_ambiguity": 0.4, "language_prior_bias": 0.8,
    },
}

# error_type -> fix_type -> match score
_FIX_MATCH: Dict[str, Dict[str, float]] = {
    "visual_ambiguity": {
        "zoom_region": 0.9, "rerun_tracker_or_detector": 0.7,
        "dense_frame_resample": 0.5, "evidence_only_rereason": 0.3,
    },
    "temporal_boundary_error": {
        "dense_frame_resample": 0.9, "zoom_region": 0.4,
        "rerun_tracker_or_detector": 0.6, "evidence_only_rereason": 0.5,
    },
    "tracking_failure": {
        "rerun_tracker_or_detector": 0.9, "zoom_region": 0.6,
        "dense_frame_resample": 0.7, "evidence_only_rereason": 0.3,
    },
    "ocr_ambiguity": {
        "zoom_region": 0.85, "rerun_tracker_or_detector": 0.7,
        "dense_frame_resample": 0.5, "evidence_only_rereason": 0.4,
    },
    "language_prior_bias": {
        "evidence_only_rereason": 0.9, "zoom_region": 0.3,
        "dense_frame_resample": 0.3, "rerun_tracker_or_detector": 0.3,
    },
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_groundedness(reflection: ReflectionOutput,
                       evidence: EvidenceBundle) -> float:
    """Check whether the reflection's claims are supported by observable evidence."""
    if not reflection.claim_reviews:
        return 0.5  # no claims to audit

    scores: list[float] = []
    for review in reflection.claim_reviews:
        s = _score_review_groundedness(review, evidence)
        scores.append(_clamp(s))

    return sum(scores) / len(scores)


def _score_review_groundedness(review: ClaimReview,
                               evidence: EvidenceBundle) -> float:
    et = review.error_type
    span = review.time_span

    if et == "visual_ambiguity":
        # Use real frame quality metrics when available, else fall back to
        # detection confidence as proxy.
        blur_score = _avg_blur_in_span(evidence, span)
        det_score = _avg_detection_conf_in_span(evidence, span)
        if blur_score is not None:
            # blur_score is Laplacian variance; <100 is blurry
            # Normalise: 0 → 1.0 (very blurry), 500+ → 0.0 (sharp)
            norm_blur = _clamp(1.0 - blur_score / 500.0)
            brightness_penalty = _brightness_penalty_in_span(evidence, span)
            det_part = (1.0 - det_score) if det_score > 0 else 0.3
            return 0.4 * norm_blur + 0.3 * det_part + 0.3 * brightness_penalty
        else:
            return (1.0 - det_score) if det_score > 0 else 0.4

    elif et == "temporal_boundary_error":
        # Check: is the span near frame boundaries? Are frames sparse?
        return _temporal_boundary_uncertainty(evidence, span)

    elif et == "tracking_failure":
        # Check: do target objects have inconsistent confidence?
        return _track_inconsistency(evidence, review.objects)

    elif et == "ocr_ambiguity":
        # Check: are OCR spans present and potentially ambiguous?
        return _ocr_uncertainty(evidence, span)

    elif et == "language_prior_bias":
        # Check: is the support trace thin?
        return _weak_support_score(evidence, review.claim_id)

    return 0.3


def score_attribution_validity(reflection: ReflectionOutput,
                                base_output: BaseOutput,
                                evidence: EvidenceBundle) -> float:
    """Check if the error attribution is compatible with the claim type."""
    if not reflection.claim_reviews:
        return 0.5

    claim_map = {c.claim_id: c for c in base_output.claims}
    scores: list[float] = []

    for review in reflection.claim_reviews:
        claim = claim_map.get(review.claim_id)
        ctype = claim.type if claim else "general"
        et = review.error_type

        compat = _CLAIM_ERROR_COMPAT.get(ctype, _CLAIM_ERROR_COMPAT.get("summary", {}))
        s = compat.get(et, 0.4)

        # Bonus: object overlap
        if claim and review.objects:
            overlap = set(review.objects) & set(claim.objects)
            if overlap:
                s += 0.1

        # Bonus: time span overlap
        if claim and claim.span and review.time_span:
            if _spans_overlap(claim.span, review.time_span):
                s += 0.1

        scores.append(_clamp(s))

    return sum(scores) / len(scores)


def score_fix_validity(reflection: ReflectionOutput,
                       evidence: EvidenceBundle) -> float:
    """Check if proposed fixes match the error types."""
    if not reflection.claim_reviews:
        return 0.5

    scores: list[float] = []
    for review in reflection.claim_reviews:
        et = review.error_type
        fixes = review.proposed_fix
        if not fixes:
            scores.append(0.2)
            continue

        fix_scores = []
        match_table = _FIX_MATCH.get(et, {})
        for f in fixes:
            fix_scores.append(match_table.get(f, 0.3))
        scores.append(sum(fix_scores) / len(fix_scores))

    return sum(scores) / len(scores)


def run_meta_audit(base_output: BaseOutput,
                   reflection: ReflectionOutput,
                   evidence: EvidenceBundle,
                   config: dict) -> AuditResult:
    """Run the full meta-audit and produce an AuditResult."""

    g = score_groundedness(reflection, evidence)
    a = score_attribution_validity(reflection, base_output, evidence)
    f = score_fix_validity(reflection, evidence)

    meta_trust = 0.4 * g + 0.3 * a + 0.3 * f

    hypotheses = build_simple_hypotheses(reflection, base_output, evidence)
    recommended = _recommend_intervention(reflection, hypotheses, evidence, config)

    logger.info("Meta-audit: groundedness=%.2f, attribution=%.2f, fix=%.2f, trust=%.2f",
                g, a, f, meta_trust)

    return AuditResult(
        groundedness=g,
        attribution_validity=a,
        fix_validity=f,
        meta_trust=meta_trust,
        hypotheses=hypotheses,
        recommended_intervention=recommended,
    )


def should_accept_without_intervention(audit: AuditResult, config: dict) -> bool:
    threshold = config.get("meta_trust_accept", 0.75)
    return audit.meta_trust >= threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _avg_detection_conf_in_span(evidence: EvidenceBundle,
                                span: list[float] | None) -> float:
    """Average detection confidence for frames within a time span."""
    confs = []
    for fid, meta in evidence.frame_meta.items():
        ts = meta.get("ts", 0)
        if span and not (span[0] <= ts <= span[1]):
            continue
        if "avg_det_conf" in meta:
            confs.append(meta["avg_det_conf"])
    return sum(confs) / len(confs) if confs else 0.0


def _avg_blur_in_span(evidence: EvidenceBundle,
                       span: list[float] | None) -> float | None:
    """Average blur (Laplacian variance) for frames in a span.

    Returns None if no blur data is available (cv2 was not installed when
    evidence was collected).
    """
    values = []
    for fid, meta in evidence.frame_meta.items():
        ts = meta.get("ts", 0)
        if span and not (span[0] <= ts <= span[1]):
            continue
        if "blur" in meta:
            values.append(meta["blur"])
    return sum(values) / len(values) if values else None


def _brightness_penalty_in_span(evidence: EvidenceBundle,
                                 span: list[float] | None) -> float:
    """Score how problematic brightness is in a span.

    Returns 0.0 (fine) to 1.0 (very dark or overexposed).
    """
    values = []
    for fid, meta in evidence.frame_meta.items():
        ts = meta.get("ts", 0)
        if span and not (span[0] <= ts <= span[1]):
            continue
        if "brightness" in meta:
            values.append(meta["brightness"])
    if not values:
        return 0.3  # unknown → mild penalty

    avg_b = sum(values) / len(values)
    # Ideal brightness around 100-160; penalise extremes
    if avg_b < 40:
        return 0.9  # very dark
    elif avg_b < 70:
        return 0.5
    elif avg_b > 220:
        return 0.7  # overexposed
    elif avg_b > 190:
        return 0.3
    return 0.1  # good range


def _temporal_boundary_uncertainty(evidence: EvidenceBundle,
                                   span: list[float] | None) -> float:
    """Estimate how sparse frame coverage is around a span boundary."""
    if not span or not evidence.frame_meta:
        return 0.5

    timestamps = sorted(m.get("ts", 0) for m in evidence.frame_meta.values())
    if len(timestamps) < 2:
        return 0.7  # very sparse

    # Count frames within the span
    in_span = [t for t in timestamps if span[0] <= t <= span[1]]
    span_duration = max(span[1] - span[0], 0.1)
    density = len(in_span) / span_duration

    # Low density = high uncertainty = more grounded claim
    if density < 0.5:
        return 0.8
    elif density < 1.0:
        return 0.6
    else:
        return 0.3


def _track_inconsistency(evidence: EvidenceBundle,
                          objects: list[str]) -> float:
    """Check if object tracks show confidence inconsistency."""
    if not objects or not evidence.tracks:
        return 0.5

    scores = []
    for obj in objects:
        # Try exact match or partial match
        track = evidence.tracks.get(obj)
        if not track:
            # Try matching by class name
            for tid, tinfo in evidence.tracks.items():
                if obj.split("#")[0] in tid.lower():
                    track = tinfo
                    break

        if track:
            confs = track.get("conf_values", [])
            if len(confs) >= 2:
                variance = sum((c - sum(confs)/len(confs))**2 for c in confs) / len(confs)
                # High variance = more grounded tracking failure claim
                scores.append(min(1.0, variance * 10))
            else:
                scores.append(0.4)
        else:
            # Object not found in tracks at all — supports tracking failure claim
            scores.append(0.7)

    return sum(scores) / len(scores) if scores else 0.5


def _ocr_uncertainty(evidence: EvidenceBundle,
                     span: list[float] | None) -> float:
    """Check OCR coverage and ambiguity."""
    if not evidence.ocr_spans:
        return 0.6  # no OCR data = moderately supports OCR ambiguity claim
    # If OCR spans exist, less support for ambiguity
    return 0.4


def _weak_support_score(evidence: EvidenceBundle, claim_id: str) -> float:
    """Check if the claim has thin evidence support."""
    trace = evidence.support_trace.get(claim_id, {})
    frames = trace.get("frames", [])
    objects = trace.get("objects", [])

    score = 0.5
    if len(frames) <= 2:
        score += 0.2
    if not objects:
        score += 0.15
    return _clamp(score)


def _spans_overlap(s1: list[float], s2: list[float]) -> bool:
    if len(s1) < 2 or len(s2) < 2:
        return False
    return s1[0] <= s2[1] and s2[0] <= s1[1]


def _recommend_intervention(reflection: ReflectionOutput,
                             hypotheses: list[dict],
                             evidence: EvidenceBundle,
                             config: dict) -> dict | None:
    """Select the best intervention based on rule-based priority scoring."""
    candidates = []

    for review in reflection.claim_reviews:
        et = review.error_type
        for fix in review.proposed_fix:
            match_table = _FIX_MATCH.get(et, {})
            score = match_table.get(fix, 0.3)
            candidates.append({
                "type": fix,
                "score": score,
                "review": review.model_dump(),
            })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]
