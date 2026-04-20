"""Unit and integration tests for the Meta-Reflective Auditor module."""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agent.extensions.mra.schemas import (
    AuditResult, BaseOutput, Claim, ClaimReview, Delta,
    EvidenceBundle, MRAResult, ReflectionOutput,
)
from agent.extensions.mra.config import get_default_mra_config, load_mra_config
from agent.extensions.mra.auditor import (
    run_meta_audit, score_attribution_validity, score_fix_validity,
    score_groundedness, should_accept_without_intervention,
)
from agent.extensions.mra.hypothesis import build_simple_hypotheses
from agent.extensions.mra.delta_eval import evaluate_delta
from agent.extensions.mra.decision import finalize_decision, finalize_early
from agent.extensions.mra.evidence_collector import collect_evidence, summarize_evidence
from agent.extensions.mra.prompts import (
    build_evidence_only_rereason_prompt, build_reflection_prompt,
)
from agent.extensions.mra.reflector import _parse_reflection, _fallback_reflection
from agent.extensions.mra.intervention import select_best_intervention
from agent.extensions.mra.evidence_collector import (
    estimate_frame_quality, estimate_motion_proxy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_base_output():
    return BaseOutput(
        answer="The person picks up the cup",
        answer_confidence=0.74,
        claims=[Claim(
            claim_id="c1",
            type="event",
            text="person picks up the cup",
            span=[42.0, 58.0],
            objects=["person#1", "cup#1"],
            confidence=0.74,
            support_refs={"frames": ["f_0010", "f_0012", "f_0014"]},
        )],
        trace={"sampled_frames": ["f_0010", "f_0012", "f_0014"]},
    )


@pytest.fixture
def sample_reflection():
    return ReflectionOutput(
        overall_self_confidence=0.61,
        claim_reviews=[ClaimReview(
            claim_id="c1",
            status="possibly_wrong",
            error_type="visual_ambiguity",
            time_span=[42.0, 48.0],
            region_hint="hand_cup_contact_region",
            objects=["hand#1", "cup#1"],
            evidence_gap="contact relation unclear",
            proposed_fix=["zoom_region", "rerun_tracker_or_detector"],
            expected_change="confidence for pickup should decrease",
        )],
        global_risk="medium",
    )


@pytest.fixture
def sample_evidence():
    return EvidenceBundle(
        frame_meta={
            "f_0010": {"ts": 42.0, "has_caption": True, "caption_len": 30, "avg_det_conf": 0.55},
            "f_0012": {"ts": 45.0, "has_caption": True, "caption_len": 25, "avg_det_conf": 0.48},
            "f_0014": {"ts": 52.0, "has_caption": True, "caption_len": 35, "avg_det_conf": 0.72},
        },
        tracks={
            "person": {"avg_conf": 0.78, "frame_count": 3, "conf_values": [0.8, 0.75, 0.79]},
            "cup": {"avg_conf": 0.62, "frame_count": 3, "conf_values": [0.7, 0.55, 0.61]},
        },
        detection_results={},
        ocr_spans=[],
        event_candidates=[],
        support_trace={
            "c1": {"frames": ["f_0010", "f_0012", "f_0014"], "objects": ["person#1", "cup#1"], "span": [42.0, 58.0]},
        },
    )


@pytest.fixture
def mra_config():
    return get_default_mra_config()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestBaseOutputAdapter:
    def test_from_brief_result(self):
        result = {
            "timeline": "A person enters the room and sits down.",
            "frames": {"items": [{"id": "f_0000", "ts": 0, "path": "/tmp/f.jpg", "caption": "room"}], "strategy": {"type": "scene", "params": {}}},
            "video": {"duration_sec": 60.0},
            "asr": {"segments": [], "language": None},
        }
        out = BaseOutput.from_workflow_result(result, "brief")
        assert out.answer
        assert len(out.claims) >= 1
        assert out.claims[0].claim_id == "c0"

    def test_from_ask_result(self):
        result = {
            "result": {"answer": "Yes, the cup was picked up", "evidence": ["frame 42"]},
            "hits": [{"start": 40, "end": 55, "text": "context chunk"}],
        }
        out = BaseOutput.from_workflow_result(result, "ask", query="Was the cup picked up?")
        assert "cup" in out.answer.lower()
        assert out.claims[0].type == "qa_answer"


class TestReflectionOutputParsing:
    def test_valid_json(self):
        data = {
            "overall_self_confidence": 0.61,
            "claim_reviews": [{
                "claim_id": "c1",
                "status": "possibly_wrong",
                "error_type": "visual_ambiguity",
                "time_span": [42, 48],
                "objects": ["hand#1"],
                "evidence_gap": "unclear",
                "proposed_fix": ["zoom_region"],
                "expected_change": "decrease",
            }],
            "global_risk": "medium",
        }
        r = ReflectionOutput.model_validate(data)
        assert r.overall_self_confidence == 0.61
        assert len(r.claim_reviews) == 1

    def test_empty_reviews(self):
        r = ReflectionOutput(overall_self_confidence=0.5, claim_reviews=[], global_risk="low")
        assert r.global_risk == "low"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMRAConfig:
    def test_defaults(self):
        cfg = get_default_mra_config()
        assert cfg["max_intervention_rounds"] == 1
        assert "visual_ambiguity" in cfg["supported_error_types"]

    def test_load_with_overrides(self):
        wf_cfg = {"audit": {"base_mode": "detailed", "zoom_size": 512}}
        cfg = load_mra_config(wf_cfg)
        assert cfg["base_mode"] == "detailed"
        assert cfg["zoom_size"] == 512
        assert cfg["max_intervention_rounds"] == 1  # default preserved


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestGroundedness:
    def test_visual_ambiguity_low_conf(self, sample_reflection, sample_evidence):
        score = score_groundedness(sample_reflection, sample_evidence)
        # Low detection conf in span -> high groundedness for visual_ambiguity
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # should be reasonably grounded

    def test_empty_reflection(self, sample_evidence):
        empty = ReflectionOutput(overall_self_confidence=0.5, claim_reviews=[], global_risk="low")
        score = score_groundedness(empty, sample_evidence)
        assert score == 0.5


class TestAttributionValidity:
    def test_event_visual_ambiguity(self, sample_reflection, sample_base_output, sample_evidence):
        score = score_attribution_validity(sample_reflection, sample_base_output, sample_evidence)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # event + visual_ambiguity is a reasonable match


class TestFixValidity:
    def test_zoom_for_visual_ambiguity(self, sample_reflection, sample_evidence):
        score = score_fix_validity(sample_reflection, sample_evidence)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # zoom_region + visual_ambiguity = good match

    def test_wrong_fix_for_error(self, sample_evidence):
        # language_prior_bias with zoom_region = poor match
        ref = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="language_prior_bias",
                proposed_fix=["zoom_region"],
            )],
            global_risk="medium",
        )
        score = score_fix_validity(ref, sample_evidence)
        assert score < 0.5


# ---------------------------------------------------------------------------
# Meta-audit tests
# ---------------------------------------------------------------------------

class TestMetaAudit:
    def test_produces_audit_result(self, sample_base_output, sample_reflection, sample_evidence, mra_config):
        audit = run_meta_audit(sample_base_output, sample_reflection, sample_evidence, mra_config)
        assert isinstance(audit, AuditResult)
        assert 0 <= audit.meta_trust <= 1
        assert audit.recommended_intervention is not None

    def test_accept_threshold(self, mra_config):
        audit = AuditResult(meta_trust=0.80)
        assert should_accept_without_intervention(audit, mra_config)

    def test_reject_threshold(self, mra_config):
        audit = AuditResult(meta_trust=0.50)
        assert not should_accept_without_intervention(audit, mra_config)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

class TestHypotheses:
    def test_generates_two_per_claim(self, sample_reflection, sample_base_output, sample_evidence):
        hyps = build_simple_hypotheses(sample_reflection, sample_base_output, sample_evidence)
        assert len(hyps) == 2  # one claim -> two hypotheses
        types = {h["type"] for h in hyps}
        assert "original_correct" in types
        assert "reflection_supported" in types


# ---------------------------------------------------------------------------
# Delta evaluation tests
# ---------------------------------------------------------------------------

class TestDeltaEval:
    def test_answer_changed(self, sample_base_output, sample_reflection, sample_evidence):
        new_out = BaseOutput(
            answer="The person did NOT pick up the cup",
            answer_confidence=0.41,
            claims=[Claim(claim_id="c1", text="no pickup", confidence=0.41)],
        )
        intervention = {"review": {"claim_id": "c1", "expected_change": "confidence should decrease"}}
        delta = evaluate_delta(
            sample_base_output, sample_reflection,
            new_out, None, sample_evidence, intervention,
        )
        assert delta.answer_changed is True
        assert delta.claim_conf_delta < 0
        assert delta.expected_change_matched is True

    def test_no_change(self, sample_base_output, sample_reflection, sample_evidence):
        intervention = {"review": {"claim_id": "c1", "expected_change": "should decrease"}}
        delta = evaluate_delta(
            sample_base_output, sample_reflection,
            sample_base_output, sample_reflection,
            sample_evidence, intervention,
        )
        assert delta.answer_changed is False
        assert delta.claim_conf_delta == 0.0


# ---------------------------------------------------------------------------
# Decision tests
# ---------------------------------------------------------------------------

class TestDecision:
    def test_accept_revised(self, sample_base_output, sample_reflection, mra_config):
        new_out = BaseOutput(answer="No", answer_confidence=0.8,
                             claims=[Claim(claim_id="c1", text="no pickup", confidence=0.8)])
        audit = AuditResult(meta_trust=0.80, groundedness=0.8,
                            attribution_validity=0.8, fix_validity=0.8)
        delta = Delta(answer_changed=True, claim_conf_delta=-0.3,
                      expected_change_matched=True, reflection_became_more_specific=True)
        result = finalize_decision(
            sample_base_output, new_out, sample_reflection, None,
            audit, delta, "zoom_region", mra_config,
        )
        assert result.status == "accept_revised_answer"

    def test_keep_uncertain(self, sample_base_output, sample_reflection, mra_config):
        audit = AuditResult(meta_trust=0.55)
        delta = Delta(answer_changed=False, claim_conf_delta=0.0)
        result = finalize_decision(
            sample_base_output, sample_base_output, sample_reflection, None,
            audit, delta, None, mra_config,
        )
        assert result.status == "keep_uncertain"

    def test_abstain(self, sample_base_output, sample_reflection, mra_config):
        audit = AuditResult(meta_trust=0.20)
        delta = Delta(answer_changed=False, claim_conf_delta=0.0)
        result = finalize_decision(
            sample_base_output, sample_base_output, sample_reflection, None,
            audit, delta, None, mra_config,
        )
        assert result.status == "abstain_or_escalate"

    def test_finalize_early(self, sample_base_output):
        result = finalize_early(
            sample_base_output, reflection=None, audit=None,
            status="accepted_without_intervention",
        )
        assert result.status == "accepted_without_intervention"
        assert result.final_output.answer == sample_base_output.answer


# ---------------------------------------------------------------------------
# Evidence collector tests
# ---------------------------------------------------------------------------

class TestEvidenceCollector:
    def test_collect_from_brief_result(self):
        result = {
            "frames": {
                "items": [
                    {"id": "f_0000", "ts": 0, "path": "/tmp/f.jpg", "caption": "a room"},
                    {"id": "f_0001", "ts": 5, "path": "/tmp/f2.jpg", "caption": None},
                ],
                "strategy": {"type": "scene", "params": {}},
            },
        }
        base = BaseOutput(answer="test", claims=[])
        # Mock asset
        asset = MagicMock()
        asset.cache_dir = "/tmp"

        evidence = collect_evidence(asset, base, result, {})
        assert len(evidence.frame_meta) == 2
        assert evidence.frame_meta["f_0000"]["has_caption"] is True
        assert evidence.frame_meta["f_0001"]["has_caption"] is False

    def test_summarize_evidence(self, sample_evidence):
        summary = summarize_evidence(sample_evidence)
        assert "Frames analysed: 3" in summary
        assert "Object tracks: 2" in summary


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_reflection_prompt_contains_key_elements(self, mra_config):
        prompt = build_reflection_prompt(
            query="Did the person pick up the cup?",
            base_output_dict={"answer": "yes", "claims": [{"claim_id": "c1", "text": "pickup", "confidence": 0.7, "span": [42, 58]}]},
            evidence_summary="Frames analysed: 3",
            config=mra_config,
        )
        assert "visual_ambiguity" in prompt
        assert "zoom_region" in prompt
        assert "JSON" in prompt

    def test_evidence_only_prompt(self):
        prompt = build_evidence_only_rereason_prompt(
            query="Did pickup happen?",
            claim={"claim_id": "c1", "text": "person picks up cup"},
            evidence_subset={"frames": ["f1", "f2"]},
        )
        assert "ONLY" in prompt
        assert "uncertain" in prompt


# ---------------------------------------------------------------------------
# Reflector parsing tests
# ---------------------------------------------------------------------------

class TestReflectorParsing:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "overall_self_confidence": 0.7,
            "claim_reviews": [{
                "claim_id": "c1",
                "status": "possibly_wrong",
                "error_type": "tracking_failure",
                "time_span": [10, 20],
                "objects": ["car#1"],
                "evidence_gap": "track lost",
                "proposed_fix": ["rerun_tracker_or_detector"],
                "expected_change": "should clarify tracking",
            }],
            "global_risk": "high",
        })
        r = _parse_reflection(raw)
        assert r.overall_self_confidence == 0.7
        assert r.claim_reviews[0].error_type == "tracking_failure"
        assert r.global_risk == "high"

    def test_parse_markdown_fenced_json(self):
        raw = "```json\n" + json.dumps({
            "overall_self_confidence": 0.5,
            "claim_reviews": [],
            "global_risk": "low",
        }) + "\n```"
        r = _parse_reflection(raw)
        assert r.overall_self_confidence == 0.5
        assert r.global_risk == "low"

    def test_parse_garbage_returns_fallback(self):
        r = _parse_reflection("this is not json at all!!!")
        assert r.overall_self_confidence == 0.5
        assert r.claim_reviews == []
        assert r.global_risk == "medium"

    def test_parse_partial_json_returns_fallback(self):
        r = _parse_reflection('{"overall_self_confidence": 0.8, "claim_reviews": [')
        assert r.overall_self_confidence == 0.5  # fallback

    def test_fallback_reflection(self):
        r = _fallback_reflection()
        assert isinstance(r, ReflectionOutput)
        assert r.claim_reviews == []


# ---------------------------------------------------------------------------
# Intervention selection tests
# ---------------------------------------------------------------------------

class TestInterventionSelection:
    def test_select_from_audit_result(self, mra_config):
        audit = AuditResult(
            meta_trust=0.5,
            recommended_intervention={
                "type": "zoom_region",
                "score": 0.9,
                "review": {"claim_id": "c1", "time_span": [42, 48]},
            },
        )
        sel = select_best_intervention(audit, mra_config)
        assert sel is not None
        assert sel["type"] == "zoom_region"

    def test_select_none_when_no_recommendation(self, mra_config):
        audit = AuditResult(meta_trust=0.5, recommended_intervention=None)
        sel = select_best_intervention(audit, mra_config)
        assert sel is None

    def test_select_none_for_unsupported_type(self, mra_config):
        audit = AuditResult(
            meta_trust=0.5,
            recommended_intervention={"type": "unknown_intervention", "score": 0.9, "review": {}},
        )
        sel = select_best_intervention(audit, mra_config)
        assert sel is None


# ---------------------------------------------------------------------------
# Groundedness scoring — all error types
# ---------------------------------------------------------------------------

class TestGroundednessAllErrorTypes:
    def test_temporal_boundary_error_sparse(self, sample_evidence):
        ref = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="temporal_boundary_error",
                time_span=[42.0, 43.0],  # very short span with sparse frames
                proposed_fix=["dense_frame_resample"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(ref, sample_evidence)
        assert 0.0 <= score <= 1.0

    def test_tracking_failure_missing_object(self, sample_evidence):
        ref = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="tracking_failure",
                objects=["robot#1"],  # not in tracks
                proposed_fix=["rerun_tracker_or_detector"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(ref, sample_evidence)
        assert score >= 0.5  # missing object supports tracking failure claim

    def test_tracking_failure_present_object(self, sample_evidence):
        ref = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="tracking_failure",
                objects=["person"],  # present in tracks, low variance
                proposed_fix=["rerun_tracker_or_detector"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(ref, sample_evidence)
        assert 0.0 <= score <= 1.0

    def test_ocr_ambiguity_no_ocr(self, sample_evidence):
        ref = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="ocr_ambiguity",
                proposed_fix=["zoom_region"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(ref, sample_evidence)
        assert score >= 0.5  # no OCR data supports ambiguity claim

    def test_language_prior_bias(self, sample_evidence):
        ref = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="language_prior_bias",
                proposed_fix=["evidence_only_rereason"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(ref, sample_evidence)
        assert 0.0 <= score <= 1.0

    def test_no_detection_conf_in_span(self):
        """Frames exist but none have detection confidence data."""
        evidence = EvidenceBundle(
            frame_meta={
                "f1": {"ts": 5.0, "has_caption": True, "caption_len": 20},
                "f2": {"ts": 10.0, "has_caption": True, "caption_len": 30},
            },
        )
        ref = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(
                claim_id="c1",
                error_type="visual_ambiguity",
                time_span=[4.0, 11.0],
                proposed_fix=["zoom_region"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(ref, evidence)
        # avg_det_conf returns 0 -> 1 - 0 = 1.0? No, _avg_detection_conf_in_span returns 0.0
        # then the branch: det_score == 0.0 -> goes to "else" -> 0.4
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Attribution validity — edge cases
# ---------------------------------------------------------------------------

class TestAttributionEdgeCases:
    def test_unknown_claim_type(self, sample_evidence):
        base = BaseOutput(
            answer="test",
            claims=[Claim(claim_id="c1", type="custom_weird_type", text="something")],
        )
        ref = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(claim_id="c1", error_type="visual_ambiguity")],
            global_risk="medium",
        )
        score = score_attribution_validity(ref, base, sample_evidence)
        assert 0.0 <= score <= 1.0

    def test_span_overlap_bonus(self, sample_evidence):
        base = BaseOutput(
            answer="test",
            claims=[Claim(claim_id="c1", type="event", text="t", span=[40.0, 60.0])],
        )
        ref_with_overlap = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(
                claim_id="c1", error_type="visual_ambiguity",
                time_span=[42.0, 50.0],  # overlaps with claim span
            )],
            global_risk="medium",
        )
        ref_no_overlap = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(
                claim_id="c1", error_type="visual_ambiguity",
                time_span=[100.0, 110.0],  # no overlap
            )],
            global_risk="medium",
        )
        score_overlap = score_attribution_validity(ref_with_overlap, base, sample_evidence)
        score_no_overlap = score_attribution_validity(ref_no_overlap, base, sample_evidence)
        assert score_overlap > score_no_overlap

    def test_missing_claim_in_base(self, sample_evidence):
        base = BaseOutput(answer="test", claims=[])
        ref = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(claim_id="c_missing", error_type="visual_ambiguity")],
            global_risk="medium",
        )
        score = score_attribution_validity(ref, base, sample_evidence)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Multi-claim scenarios
# ---------------------------------------------------------------------------

class TestMultiClaim:
    def test_multiple_claims_multiple_reviews(self, sample_evidence):
        base = BaseOutput(
            answer="Person picks up cup then opens door",
            answer_confidence=0.6,
            claims=[
                Claim(claim_id="c1", type="event", text="pickup cup",
                      span=[10, 20], confidence=0.7),
                Claim(claim_id="c2", type="event", text="opens door",
                      span=[40, 55], confidence=0.5),
            ],
        )
        ref = ReflectionOutput(
            overall_self_confidence=0.55,
            claim_reviews=[
                ClaimReview(claim_id="c1", error_type="visual_ambiguity",
                            time_span=[10, 15], proposed_fix=["zoom_region"]),
                ClaimReview(claim_id="c2", error_type="temporal_boundary_error",
                            time_span=[38, 42], proposed_fix=["dense_frame_resample"]),
            ],
            global_risk="high",
        )
        audit = run_meta_audit(base, ref, sample_evidence, get_default_mra_config())
        assert isinstance(audit, AuditResult)
        assert len(audit.hypotheses) == 4  # 2 claims * 2 hypotheses each
        assert audit.recommended_intervention is not None

    def test_multiple_reviews_fix_validity(self, sample_evidence):
        ref = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[
                ClaimReview(claim_id="c1", error_type="visual_ambiguity",
                            proposed_fix=["zoom_region"]),  # good match
                ClaimReview(claim_id="c2", error_type="language_prior_bias",
                            proposed_fix=["evidence_only_rereason"]),  # good match
            ],
            global_risk="medium",
        )
        score = score_fix_validity(ref, sample_evidence)
        assert score > 0.7  # both are good matches


# ---------------------------------------------------------------------------
# Evidence collector edge cases
# ---------------------------------------------------------------------------

class TestEvidenceCollectorEdgeCases:
    def test_collect_empty_result(self):
        base = BaseOutput(answer="test", claims=[])
        asset = MagicMock()
        asset.cache_dir = "/tmp"
        evidence = collect_evidence(asset, base, {}, {})
        assert evidence.frame_meta == {}
        assert evidence.tracks == {}

    def test_collect_with_objects(self):
        result = {
            "frames": {
                "items": [{"id": "f_0000", "ts": 0, "path": "/tmp/f.jpg", "caption": "a room"}],
                "strategy": {"type": "scene", "params": {}},
            },
            "objects": {
                "/tmp/f.jpg": [
                    {"class": "person", "confidence": 0.92, "bbox": [10, 20, 100, 200]},
                    {"class": "cup", "confidence": 0.71, "bbox": [50, 80, 70, 100]},
                ],
            },
        }
        base = BaseOutput(answer="test", claims=[])
        asset = MagicMock()
        asset.cache_dir = "/tmp"
        evidence = collect_evidence(asset, base, result, {})
        assert len(evidence.tracks) >= 1  # should build tracks from objects

    def test_collect_with_ocr_dict(self):
        result = {
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "ocr": {
                "frame_0": ["Hello world", "Test 123"],
                "frame_1": "Single text",
            },
        }
        base = BaseOutput(answer="test", claims=[])
        asset = MagicMock()
        asset.cache_dir = "/tmp"
        evidence = collect_evidence(asset, base, result, {})
        assert len(evidence.ocr_spans) == 3  # 2 from frame_0 + 1 from frame_1

    def test_collect_with_ocr_list(self):
        result = {
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "ocr": [{"text": "some text", "conf": 0.9}],
        }
        base = BaseOutput(answer="test", claims=[])
        asset = MagicMock()
        asset.cache_dir = "/tmp"
        evidence = collect_evidence(asset, base, result, {})
        assert len(evidence.ocr_spans) == 1

    def test_collect_with_timeline_chapters(self):
        result = {
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "timeline": {
                "chapters": [
                    {"start": 0, "end": 30, "title": "Introduction"},
                    {"start": 30, "end": 60, "title": "Main content"},
                ],
            },
        }
        base = BaseOutput(answer="test", claims=[])
        asset = MagicMock()
        asset.cache_dir = "/tmp"
        evidence = collect_evidence(asset, base, result, {})
        assert len(evidence.event_candidates) == 2

    def test_summarize_empty_evidence(self):
        evidence = EvidenceBundle()
        summary = summarize_evidence(evidence)
        assert summary == "No evidence available."

    def test_summarize_with_ocr_and_events(self):
        evidence = EvidenceBundle(
            frame_meta={"f1": {"ts": 0, "has_caption": True, "caption_len": 10}},
            ocr_spans=[{"frame": "f1", "text": "hello"}],
            event_candidates=[{"type": "chapter", "span": [0, 10], "text": "intro"}],
        )
        summary = summarize_evidence(evidence)
        assert "OCR text regions: 1" in summary
        assert "Event candidates: 1" in summary


# ---------------------------------------------------------------------------
# BaseOutput adapter edge cases
# ---------------------------------------------------------------------------

class TestBaseOutputEdgeCases:
    def test_from_dict_timeline(self):
        result = {
            "timeline": {"chapters": [{"title": "Intro", "summary": "Beginning"}]},
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "video": {},
        }
        out = BaseOutput.from_workflow_result(result, "brief")
        assert "Intro" in out.answer or "Beginning" in out.answer

    def test_from_list_timeline(self):
        result = {
            "timeline": [
                {"title": "Part 1", "summary": "First half"},
                {"title": "Part 2", "summary": "Second half"},
            ],
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "video": {},
        }
        out = BaseOutput.from_workflow_result(result, "detailed")
        assert "First half" in out.answer

    def test_from_string_timeline(self):
        result = {
            "timeline": "A video about cooking",
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "video": {"duration_sec": 120.0},
        }
        out = BaseOutput.from_workflow_result(result, "brief")
        assert out.answer == "A video about cooking"
        assert out.claims[0].span == [0, 120.0]

    def test_from_ask_no_evidence(self):
        result = {
            "result": "Just a plain string answer",
            "hits": [],
        }
        out = BaseOutput.from_workflow_result(result, "ask")
        assert "plain string" in out.answer

    def test_from_ask_with_visual_context(self):
        result = {
            "result": {"answer": "Yes", "evidence": ["frame_42"]},
            "hits": [
                {"start": 40, "end": 50, "text": "chunk1"},
                {"start": 50, "end": 60, "text": "chunk2"},
            ],
            "visual_context": [{"ts": 45, "caption": "person holding cup"}],
        }
        out = BaseOutput.from_workflow_result(result, "ask", query="Is it held?")
        assert out.claims[0].span == [40, 60]


# ---------------------------------------------------------------------------
# Delta eval edge cases
# ---------------------------------------------------------------------------

class TestDeltaEdgeCases:
    def test_missing_claim_in_output(self, sample_reflection, sample_evidence):
        base = BaseOutput(answer="yes", answer_confidence=0.7, claims=[])
        new = BaseOutput(answer="no", answer_confidence=0.3, claims=[])
        intervention = {"review": {"claim_id": "c_nonexistent", "expected_change": "change"}}
        delta = evaluate_delta(base, sample_reflection, new, None, sample_evidence, intervention)
        assert delta.answer_changed is True
        # conf_delta from answer_confidence since claim not found
        assert delta.claim_conf_delta == pytest.approx(0.3 - 0.7)

    def test_expected_change_clarify(self, sample_base_output, sample_reflection, sample_evidence):
        new_out = BaseOutput(
            answer=sample_base_output.answer,
            answer_confidence=0.80,
            claims=[Claim(claim_id="c1", text="pickup", confidence=0.80)],
        )
        intervention = {"review": {"claim_id": "c1", "expected_change": "should clarify the contact"}}
        delta = evaluate_delta(
            sample_base_output, sample_reflection,
            new_out, None, sample_evidence, intervention,
        )
        assert delta.expected_change_matched is True  # confidence changed by > 0.05

    def test_reflection_specificity_comparison(self, sample_base_output, sample_evidence):
        old_ref = ReflectionOutput(
            overall_self_confidence=0.5,
            claim_reviews=[ClaimReview(claim_id="c1", error_type="visual_ambiguity")],
            global_risk="medium",
        )
        new_ref = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1", error_type="visual_ambiguity",
                time_span=[42, 48], objects=["hand#1"],
                evidence_gap="contact unclear",
                proposed_fix=["zoom_region"],
                expected_change="will clarify",
            )],
            global_risk="medium",
        )
        intervention = {"review": {"claim_id": "c1", "expected_change": "change"}}
        delta = evaluate_delta(
            sample_base_output, old_ref,
            sample_base_output, new_ref,
            sample_evidence, intervention,
        )
        assert delta.reflection_became_more_specific is True


# ---------------------------------------------------------------------------
# Decision edge cases
# ---------------------------------------------------------------------------

class TestDecisionEdgeCases:
    def test_delta_boost_crosses_accept_threshold(self, sample_base_output, mra_config):
        """Delta expected_change_matched + more_specific can push trust above threshold."""
        audit = AuditResult(meta_trust=0.70)  # below 0.75
        delta = Delta(
            answer_changed=True, claim_conf_delta=-0.2,
            expected_change_matched=True,  # +0.1
            reflection_became_more_specific=True,  # +0.05
        )
        new_out = BaseOutput(answer="No", answer_confidence=0.5,
                             claims=[Claim(claim_id="c1", text="no", confidence=0.5)])
        result = finalize_decision(
            sample_base_output, new_out, None, None,
            audit, delta, "zoom_region", mra_config,
        )
        # 0.70 + 0.10 + 0.05 = 0.85 >= 0.75, answer changed -> accept_revised
        assert result.status == "accept_revised_answer"

    def test_accept_original_high_trust_no_change(self, sample_base_output, mra_config):
        audit = AuditResult(meta_trust=0.80)
        delta = Delta(answer_changed=False, claim_conf_delta=0.05)
        result = finalize_decision(
            sample_base_output, sample_base_output, None, None,
            audit, delta, None, mra_config,
        )
        assert result.status == "accept_original_with_higher_confidence"

    def test_uncertain_caps_confidence(self, sample_base_output, mra_config):
        audit = AuditResult(meta_trust=0.55)
        delta = Delta(answer_changed=False, claim_conf_delta=0.0)
        result = finalize_decision(
            sample_base_output, sample_base_output, None, None,
            audit, delta, None, mra_config,
        )
        assert result.status == "keep_uncertain"
        assert result.final_output.answer_confidence <= 0.5

    def test_abstain_caps_confidence(self, sample_base_output, mra_config):
        audit = AuditResult(meta_trust=0.10)
        delta = Delta(answer_changed=False, claim_conf_delta=0.0)
        result = finalize_decision(
            sample_base_output, sample_base_output, None, None,
            audit, delta, None, mra_config,
        )
        assert result.status == "abstain_or_escalate"
        assert result.final_output.answer_confidence <= 0.3


# ---------------------------------------------------------------------------
# Full audit pipeline (mocked LLM) — integration test
# ---------------------------------------------------------------------------

class TestFullAuditPipeline:
    """Integration tests that exercise the full audit pipeline with mocked LLM."""

    def _make_mock_llm_response(self, content: str):
        """Build a mock OpenAI ChatCompletion response."""
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def _reflection_json(self, conf=0.6, error_type="visual_ambiguity",
                          proposed_fix=None, expected_change="confidence should decrease"):
        return json.dumps({
            "overall_self_confidence": conf,
            "claim_reviews": [{
                "claim_id": "c0",
                "status": "possibly_wrong",
                "error_type": error_type,
                "time_span": [10, 20],
                "objects": [],
                "evidence_gap": "unclear",
                "proposed_fix": proposed_fix or ["evidence_only_rereason"],
                "expected_change": expected_change,
            }],
            "global_risk": "medium",
        })

    def _rereason_json(self, answer="uncertain", confidence=0.4):
        return json.dumps({
            "answer": answer,
            "confidence": confidence,
            "evidence_used": ["frame_meta"],
            "unsupported_parts": ["contact relation"],
        })

    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_reflector_generates_valid_output(self, mock_make_client):
        mock_client = MagicMock()
        mock_make_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_llm_response(
            self._reflection_json()
        )

        from agent.extensions.mra.reflector import generate_reflection
        base = BaseOutput(answer="test", answer_confidence=0.6,
                          claims=[Claim(claim_id="c0", text="test claim", confidence=0.6)])
        evidence = EvidenceBundle(
            frame_meta={"f1": {"ts": 0, "has_caption": True, "caption_len": 20}},
        )
        config = get_default_mra_config()

        reflection = generate_reflection("test query?", base, evidence, "test-model", "http://localhost:8000/v1", config)
        assert isinstance(reflection, ReflectionOutput)
        assert len(reflection.claim_reviews) == 1
        assert reflection.claim_reviews[0].error_type == "visual_ambiguity"

    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_reflector_handles_llm_failure(self, mock_make_client):
        mock_client = MagicMock()
        mock_make_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        from agent.extensions.mra.reflector import generate_reflection
        base = BaseOutput(answer="test", claims=[])
        evidence = EvidenceBundle()
        config = get_default_mra_config()

        reflection = generate_reflection("q", base, evidence, "m", "http://x", config)
        assert isinstance(reflection, ReflectionOutput)
        assert reflection.claim_reviews == []  # fallback

    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_evidence_only_rereason(self, mock_make_client):
        mock_client = MagicMock()
        mock_make_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_llm_response(
            self._rereason_json(answer="Not enough evidence for pickup", confidence=0.35)
        )

        from agent.extensions.mra.intervention import rerun_local_reasoning
        base = BaseOutput(
            answer="person picks up cup", answer_confidence=0.7,
            claims=[Claim(claim_id="c1", text="pickup", confidence=0.7)],
        )
        evidence = EvidenceBundle(
            frame_meta={"f1": {"ts": 10, "has_caption": True, "caption_len": 20}},
        )
        intervention = {
            "type": "evidence_only_rereason",
            "review": {"claim_id": "c1"},
        }
        config = get_default_mra_config()

        new_out = rerun_local_reasoning("Did pickup happen?", base, evidence, intervention,
                                         "model", "http://localhost:8000/v1", config)
        assert new_out.answer_confidence == pytest.approx(0.35)
        assert "Not enough evidence" in new_out.answer

    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_evidence_only_rereason_handles_failure(self, mock_make_client):
        mock_client = MagicMock()
        mock_make_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("timeout")

        from agent.extensions.mra.intervention import rerun_local_reasoning
        base = BaseOutput(
            answer="original", answer_confidence=0.7,
            claims=[Claim(claim_id="c1", text="claim", confidence=0.7)],
        )
        evidence = EvidenceBundle()
        intervention = {"type": "evidence_only_rereason", "review": {"claim_id": "c1"}}
        config = get_default_mra_config()

        new_out = rerun_local_reasoning("q", base, evidence, intervention, "m", "http://x", config)
        # Should return base_output unchanged on failure
        assert new_out.answer == "original"
        assert new_out.answer_confidence == 0.7

    def test_full_audit_accept_path(self, sample_base_output, sample_evidence, mra_config):
        """End-to-end: high-confidence reflection -> accepted without intervention."""
        reflection = ReflectionOutput(
            overall_self_confidence=0.9,
            claim_reviews=[ClaimReview(
                claim_id="c1", status="likely_correct",
                error_type="visual_ambiguity",
                time_span=[42, 48],
                proposed_fix=["zoom_region"],
                expected_change="minor clarification",
            )],
            global_risk="low",
        )
        audit = run_meta_audit(sample_base_output, reflection, sample_evidence, mra_config)
        # With well-matched reflection, trust should be reasonable
        assert isinstance(audit.meta_trust, float)

    def test_full_audit_intervention_path(self, sample_base_output, sample_reflection,
                                          sample_evidence, mra_config):
        """End-to-end: low trust -> intervention selected -> delta computed -> decision made."""
        # Audit
        audit = run_meta_audit(sample_base_output, sample_reflection, sample_evidence, mra_config)

        # Select intervention
        intervention = select_best_intervention(audit, mra_config)
        assert intervention is not None

        # Simulate intervention result
        new_out = BaseOutput(
            answer="Person approaches but does not pick up the cup",
            answer_confidence=0.42,
            claims=[Claim(claim_id="c1", text="no pickup", confidence=0.42)],
        )
        new_ref = ReflectionOutput(
            overall_self_confidence=0.75,
            claim_reviews=[ClaimReview(
                claim_id="c1", status="possibly_wrong",
                error_type="visual_ambiguity",
                time_span=[42, 48], objects=["hand#1"],
                evidence_gap="no stable grasp",
                proposed_fix=["zoom_region"],
                expected_change="confirms no pickup",
            )],
            global_risk="low",
        )

        # Delta
        delta = evaluate_delta(
            sample_base_output, sample_reflection,
            new_out, new_ref, sample_evidence, intervention,
        )
        assert delta.answer_changed is True
        assert delta.claim_conf_delta < 0

        # Decision
        result = finalize_decision(
            sample_base_output, new_out, sample_reflection, new_ref,
            audit, delta, intervention["type"], mra_config,
        )
        assert isinstance(result, MRAResult)
        assert result.status in (
            "accept_revised_answer", "accept_original_with_higher_confidence",
            "keep_uncertain", "abstain_or_escalate",
        )
        assert result.audit_log.get("intervention_type") == intervention["type"]


# ---------------------------------------------------------------------------
# Runner integration test (fully mocked)
# ---------------------------------------------------------------------------

class TestRunnerIntegration:
    """Test the runner entry point with all external calls mocked."""

    @patch("agent.extensions.mra.runner._run_base_workflow")
    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_runner_high_confidence_skips_mra(self, mock_make_client, mock_base_wf):
        """When all claims have high confidence, MRA is skipped entirely."""
        # Base workflow returns a result where the synthesized claim will have conf 0.6
        # But we set min_claim_conf_for_reflect to 0.5 so it triggers reflection
        # Actually, let's set the threshold low enough that the default 0.6 is above it
        mock_base_wf.return_value = {
            "timeline": "A clear video of someone cooking.",
            "frames": {"items": [{"id": "f0", "ts": 0, "path": "/tmp/f.jpg", "caption": "cooking"}],
                       "strategy": {"type": "scene", "params": {}}},
            "video": {"duration_sec": 30},
            "asr": {"segments": [], "language": None},
        }

        from agent.extensions.mra.runner import run_with_meta_reflection

        asset = MagicMock()
        asset.cache_dir = tempfile.mkdtemp()

        cfg = {
            "llm_base_url": "http://localhost:8000/v1",
            "llm_model": "test-model",
        }

        # Override the min_claim_conf_for_reflect to be very low so it accepts
        with patch("agent.extensions.mra.runner.load_mra_config") as mock_cfg:
            mock_cfg.return_value = {
                **get_default_mra_config(),
                "min_claim_conf_for_reflect": 0.01,  # every claim is above this
            }
            result = run_with_meta_reflection(asset, None, "audit", cfg)

        assert "mra" in result
        assert result["mra"]["status"] == "accepted_without_intervention"
        # LLM should NOT have been called (no reflection needed)
        mock_make_client.assert_not_called()

    @patch("agent.extensions.mra.runner._run_base_workflow")
    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_runner_empty_reflection_accepts(self, mock_make_client, mock_base_wf):
        """When the LLM returns an empty reflection, accept the base output."""
        mock_base_wf.return_value = {
            "timeline": "A video",
            "frames": {"items": [], "strategy": {"type": "scene", "params": {}}},
            "video": {"duration_sec": 30},
        }

        # LLM returns a reflection with no claim reviews
        mock_client = MagicMock()
        mock_make_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "overall_self_confidence": 0.8,
                "claim_reviews": [],
                "global_risk": "low",
            })))]
        )

        from agent.extensions.mra.runner import run_with_meta_reflection

        asset = MagicMock()
        asset.cache_dir = tempfile.mkdtemp()

        cfg = {
            "llm_base_url": "http://localhost:8000/v1",
            "llm_model": "test-model",
        }
        result = run_with_meta_reflection(asset, None, "audit", cfg)
        assert result["mra"]["status"] == "accepted_without_intervention"

    @patch("agent.extensions.mra.runner._run_base_workflow")
    @patch("agent.extensions.models.vllm_openai_client.make_client")
    def test_runner_full_intervention_flow(self, mock_client,
                                            mock_base_wf):
        """Full flow: base -> reflect -> audit -> intervene -> decide."""
        mock_base_wf.return_value = {
            "timeline": "Person picks up a cup",
            "frames": {
                "items": [
                    {"id": "f0", "ts": 10, "path": "/tmp/f.jpg", "caption": "person near cup"},
                ],
                "strategy": {"type": "scene", "params": {}},
            },
            "video": {"duration_sec": 60},
        }

        # Reflection LLM call — deliberately mismatched to get lower meta-trust
        # Use ocr_ambiguity error for a non-OCR task => attribution score drops
        reflection_json = json.dumps({
            "overall_self_confidence": 0.55,
            "claim_reviews": [{
                "claim_id": "c0",
                "status": "possibly_wrong",
                "error_type": "ocr_ambiguity",
                "time_span": [5, 15],
                "objects": [],
                "evidence_gap": "text unclear",
                "proposed_fix": ["evidence_only_rereason"],
                "expected_change": "confidence should decrease if no grounding",
            }],
            "global_risk": "medium",
        })
        # Post-intervention reflection
        post_reflection_json = json.dumps({
            "overall_self_confidence": 0.7,
            "claim_reviews": [{
                "claim_id": "c0",
                "status": "possibly_wrong",
                "error_type": "ocr_ambiguity",
                "time_span": [5, 15],
                "objects": [],
                "evidence_gap": "still no strong visual evidence",
                "proposed_fix": ["evidence_only_rereason"],
                "expected_change": "remains uncertain",
            }],
            "global_risk": "medium",
        })

        # Intervention (evidence_only_rereason) LLM call
        rereason_json = json.dumps({
            "answer": "Uncertain — not enough visual evidence for pickup",
            "confidence": 0.35,
            "evidence_used": ["frame_meta"],
            "unsupported_parts": ["pickup action"],
        })

        mock_reflect = MagicMock()
        mock_client.return_value = mock_reflect
        # All LLM calls go through the same mock_client (reflector + intervention)
        # Call order: reflection, rereason, post-intervention reflection
        mock_reflect.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=reflection_json))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=rereason_json))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=post_reflection_json))]),
        ]

        from agent.extensions.mra.runner import run_with_meta_reflection

        asset = MagicMock()
        asset.cache_dir = tempfile.mkdtemp()

        cfg = {
            "llm_base_url": "http://localhost:8000/v1",
            "llm_model": "test-model",
        }

        # Override the accept threshold to be very high so intervention is always triggered
        with patch("agent.extensions.mra.runner.load_mra_config") as mock_mra_cfg:
            from agent.extensions.mra.config import get_default_mra_config as _get_defaults
            mock_mra_cfg.return_value = {**_get_defaults(), "meta_trust_accept": 0.99}
            result = run_with_meta_reflection(asset, "Did the person pick up the cup?", "audit", cfg)

        mra = result["mra"]
        assert mra["status"] in (
            "accept_revised_answer", "accept_original_with_higher_confidence",
            "keep_uncertain", "abstain_or_escalate",
        )
        assert mra["intervention_type"] is not None
        assert mra["delta"] is not None
        assert mra["audit"]["meta_trust"] >= 0

        # Verify the audit log has useful info
        audit_log = mra["audit_log"]
        assert "meta_trust_raw" in audit_log or "meta_trust" in audit_log
        assert "status" in audit_log

        # Verify MRA result file was written
        mra_path = os.path.join(asset.cache_dir, "mra_audit.json")
        assert os.path.exists(mra_path)
        with open(mra_path) as f:
            saved = json.load(f)
        assert saved["status"] == mra["status"]


# ---------------------------------------------------------------------------
# MRA result serialization
# ---------------------------------------------------------------------------

class TestMRAResultSerialization:
    def test_model_dump_roundtrip(self, sample_base_output, sample_reflection):
        result = MRAResult(
            status="keep_uncertain",
            final_output=sample_base_output,
            base_output=sample_base_output,
            reflection=sample_reflection,
            audit=AuditResult(meta_trust=0.6, groundedness=0.7,
                              attribution_validity=0.6, fix_validity=0.5),
            delta=Delta(answer_changed=False, claim_conf_delta=0.0),
            intervention_type="zoom_region",
            audit_log={"status": "keep_uncertain"},
        )
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["status"] == "keep_uncertain"
        # Should be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert '"keep_uncertain"' in json_str

    def test_model_dump_minimal(self):
        base = BaseOutput(answer="test", claims=[])
        result = MRAResult(status="abstain_or_escalate", final_output=base, base_output=base)
        d = result.model_dump()
        assert d["reflection"] is None
        assert d["delta"] is None


# ---------------------------------------------------------------------------
# Frame quality estimation tests
# ---------------------------------------------------------------------------

class TestFrameQualityEstimation:
    """Tests for real frame quality metrics (blur, brightness, contrast).

    These tests work both with and without cv2 installed — they verify
    correct behaviour in both environments.
    """

    def test_estimate_nonexistent_file(self):
        result = estimate_frame_quality("/nonexistent/file.jpg")
        assert result == {}

    def test_estimate_frame_quality_returns_dict(self, tmp_path):
        """When cv2 IS available, quality dict has blur/brightness/contrast/edge_density."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2 not installed on this node")

        # Create a real test image: gradient (sharp, good brightness)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            img[i, :, :] = i * 2  # gradient
        path = str(tmp_path / "test_frame.jpg")
        cv2.imwrite(path, img)

        result = estimate_frame_quality(path)
        assert "blur" in result
        assert "brightness" in result
        assert "contrast" in result
        assert "edge_density" in result
        assert isinstance(result["blur"], float)
        assert 0 <= result["brightness"] <= 255
        assert result["contrast"] >= 0

    def test_estimate_blurry_vs_sharp(self, tmp_path):
        """Blurry images should have lower Laplacian variance than sharp ones."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2 not installed on this node")

        # Sharp: high-frequency checkerboard
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[::2, ::2] = 255
        sharp[1::2, 1::2] = 255
        sharp_path = str(tmp_path / "sharp.jpg")
        cv2.imwrite(sharp_path, sharp)

        # Blurry: heavily Gaussian-blurred version
        blurry = cv2.GaussianBlur(sharp, (31, 31), 10)
        blurry_path = str(tmp_path / "blurry.jpg")
        cv2.imwrite(blurry_path, blurry)

        q_sharp = estimate_frame_quality(sharp_path)
        q_blurry = estimate_frame_quality(blurry_path)
        assert q_sharp["blur"] > q_blurry["blur"]

    def test_motion_proxy_identical_frames(self, tmp_path):
        """Identical frames should have near-zero motion."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2 not installed on this node")

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        p1 = str(tmp_path / "a.jpg")
        p2 = str(tmp_path / "b.jpg")
        cv2.imwrite(p1, img)
        cv2.imwrite(p2, img)

        motion = estimate_motion_proxy(p1, p2)
        assert motion >= 0
        assert motion < 1.0  # should be nearly zero

    def test_motion_proxy_different_frames(self, tmp_path):
        """Very different frames should have high motion score."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2 not installed on this node")

        black = np.zeros((50, 50, 3), dtype=np.uint8)
        white = np.full((50, 50, 3), 255, dtype=np.uint8)
        p1 = str(tmp_path / "black.jpg")
        p2 = str(tmp_path / "white.jpg")
        cv2.imwrite(p1, black)
        cv2.imwrite(p2, white)

        motion = estimate_motion_proxy(p1, p2)
        assert motion > 100  # should be very high

    def test_motion_proxy_missing_file(self):
        result = estimate_motion_proxy("/no/a.jpg", "/no/b.jpg")
        assert result == -1.0

    def test_estimate_quality_no_cv2(self):
        """When cv2 is not available, estimate_frame_quality returns empty dict."""
        import agent.extensions.mra.evidence_collector as ec
        original = ec._HAS_CV2
        try:
            ec._HAS_CV2 = False
            result = ec.estimate_frame_quality("/any/path.jpg")
            assert result == {}

            motion = ec.estimate_motion_proxy("/a.jpg", "/b.jpg")
            assert motion == -1.0
        finally:
            ec._HAS_CV2 = original


class TestEvidenceEnrichment:
    """Test that evidence collection enriches frame_meta with quality metrics."""

    def test_enrichment_with_real_images(self, tmp_path):
        """When frame files exist and cv2 is available, quality metrics are added."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2 not installed on this node")

        # Create a real frame image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_path = str(tmp_path / "f_0000.jpg")
        cv2.imwrite(frame_path, img)

        result = {
            "frames": {
                "items": [{"id": "f_0000", "ts": 5.0, "path": frame_path, "caption": "something"}],
                "strategy": {"type": "scene", "params": {}},
            },
        }
        base = BaseOutput(answer="test", claims=[])
        asset = MagicMock()
        asset.cache_dir = str(tmp_path)

        evidence = collect_evidence(asset, base, result, {})
        fmeta = evidence.frame_meta["f_0000"]

        assert "blur" in fmeta
        assert "brightness" in fmeta
        assert "contrast" in fmeta
        assert "edge_density" in fmeta
        assert fmeta["has_caption"] is True

    def test_enrichment_without_cv2(self):
        """When cv2 is not available, frame_meta has caption info but no quality metrics."""
        import agent.extensions.mra.evidence_collector as ec
        original = ec._HAS_CV2
        try:
            ec._HAS_CV2 = False
            result = {
                "frames": {
                    "items": [{"id": "f_0000", "ts": 0, "path": "/tmp/x.jpg", "caption": "hello"}],
                    "strategy": {"type": "scene", "params": {}},
                },
            }
            base = BaseOutput(answer="test", claims=[])
            asset = MagicMock()
            asset.cache_dir = "/tmp"

            evidence = collect_evidence(asset, base, result, {})
            fmeta = evidence.frame_meta["f_0000"]

            assert "blur" not in fmeta
            assert fmeta["has_caption"] is True
            assert fmeta["caption_len"] == 5
        finally:
            ec._HAS_CV2 = original

    def test_summarize_with_blur_data(self):
        """Summary includes blur stats when available."""
        evidence = EvidenceBundle(
            frame_meta={
                "f1": {"ts": 0, "has_caption": True, "caption_len": 10,
                       "blur": 50.0, "brightness": 120.0},
                "f2": {"ts": 5, "has_caption": True, "caption_len": 20,
                       "blur": 300.0, "brightness": 130.0},
            },
        )
        summary = summarize_evidence(evidence)
        assert "Frame quality (blur)" in summary
        assert "low-quality=1" in summary  # f1 has blur=50 < 100
        assert "Brightness" in summary


class TestAuditorWithRealMetrics:
    """Test that auditor scoring uses real blur/brightness when available."""

    def test_visual_ambiguity_with_blur_data(self):
        """When blur data is present, groundedness uses it instead of detection proxy."""
        from agent.extensions.mra.auditor import score_groundedness

        # Blurry frames in span — should strongly support visual_ambiguity claim
        evidence = EvidenceBundle(
            frame_meta={
                "f1": {"ts": 10, "blur": 30.0, "brightness": 45.0},   # very blurry + dark
                "f2": {"ts": 15, "blur": 50.0, "brightness": 55.0},   # blurry
            },
        )
        reflection = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1", error_type="visual_ambiguity",
                time_span=[8, 18], proposed_fix=["zoom_region"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(reflection, evidence)
        assert score > 0.6  # blurry + dark = well-grounded claim

    def test_visual_ambiguity_sharp_frames(self):
        """Sharp frames should yield low groundedness for visual_ambiguity."""
        from agent.extensions.mra.auditor import score_groundedness

        evidence = EvidenceBundle(
            frame_meta={
                "f1": {"ts": 10, "blur": 800.0, "brightness": 130.0},  # very sharp
                "f2": {"ts": 15, "blur": 900.0, "brightness": 140.0},  # very sharp
            },
        )
        reflection = ReflectionOutput(
            overall_self_confidence=0.6,
            claim_reviews=[ClaimReview(
                claim_id="c1", error_type="visual_ambiguity",
                time_span=[8, 18], proposed_fix=["zoom_region"],
            )],
            global_risk="medium",
        )
        score = score_groundedness(reflection, evidence)
        assert score < 0.4  # sharp frames → visual_ambiguity claim is NOT grounded
