"""Main MRA entry point: run_with_meta_reflection."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from agent.config import load_workflows_config
from agent.core.events import event_bus
from agent.extensions.mra.config import load_mra_config
from agent.extensions.mra.schemas import BaseOutput, MRAResult

logger = logging.getLogger(__name__)


def run_with_meta_reflection(asset: Any,
                             query: str | None,
                             mode: str,
                             cfg: dict) -> dict:
    """Run a base workflow, then apply the Meta-Reflective Auditor loop.

    Returns a dict containing both the raw workflow result and the MRA audit.
    """
    from agent.extensions.mra.auditor import (
        run_meta_audit, should_accept_without_intervention,
    )
    from agent.extensions.mra.decision import finalize_decision, finalize_early
    from agent.extensions.mra.delta_eval import evaluate_delta
    from agent.extensions.mra.evidence_collector import collect_evidence
    from agent.extensions.mra.intervention import (
        execute_intervention, rerun_local_reasoning, select_best_intervention,
    )
    from agent.extensions.mra.reflector import generate_reflection

    # ── Load MRA config ─────────────────────────────────────────────
    workflows_config = load_workflows_config()
    mra_cfg = load_mra_config(workflows_config)

    llm_base_url = cfg.get("llm_base_url", "http://localhost:8000/v1")
    # When running with pooled vLLM endpoints, the URL may be comma-separated;
    # MRA only needs a single endpoint for its LLM calls.
    if isinstance(llm_base_url, str) and "," in llm_base_url:
        llm_base_url = llm_base_url.split(",")[0].strip()
    llm_model = cfg.get("llm_model", "qwen3.5-9b")

    # ── Step 1: Run base workflow ────────────────────────────────────
    base_mode = mra_cfg.get("base_mode", "brief")
    event_bus.emit_progress("MRA: running base analysis", 5)

    result = _run_base_workflow(asset, base_mode, query, cfg)

    event_bus.emit_progress("MRA: base analysis complete", 30)

    # ── Step 2: Standardize output ───────────────────────────────────
    effective_mode = "ask" if query and "result" in result else base_mode
    base_output = BaseOutput.from_workflow_result(result, effective_mode, query)

    # ── Step 3: Check if reflection is needed ────────────────────────
    min_conf = mra_cfg.get("min_claim_conf_for_reflect", 0.80)
    if all(c.confidence >= min_conf for c in base_output.claims) and base_output.claims:
        logger.info("All claims above confidence threshold (%.2f), skipping MRA", min_conf)
        mra_result = finalize_early(
            base_output, reflection=None, audit=None,
            status="accepted_without_intervention",
            raw_workflow_result=result,
        )
        _save_mra_result(asset.cache_dir, mra_result)
        return _merge_result(result, mra_result)

    # ── Step 4: Collect evidence ─────────────────────────────────────
    event_bus.emit_skill_start("MRA Evidence Collection", progress_pct=35)
    evidence = collect_evidence(asset, base_output, result, mra_cfg)
    event_bus.emit_skill_complete("MRA Evidence Collection", progress_pct=40)

    # ── Step 5: Generate reflection ──────────────────────────────────
    event_bus.emit_skill_start("MRA Reflection", progress_pct=40)
    reflection = generate_reflection(
        query, base_output, evidence, llm_model, llm_base_url, mra_cfg,
    )
    event_bus.emit_skill_complete("MRA Reflection", progress_pct=50)

    if not reflection.claim_reviews:
        logger.info("Reflection produced no claim reviews, accepting base output")
        mra_result = finalize_early(
            base_output, reflection=reflection, audit=None,
            status="accepted_without_intervention",
            raw_workflow_result=result,
        )
        _save_mra_result(asset.cache_dir, mra_result)
        return _merge_result(result, mra_result)

    # ── Step 6: Meta-audit ───────────────────────────────────────────
    event_bus.emit_skill_start("MRA Meta-Audit", progress_pct=50)
    audit = run_meta_audit(base_output, reflection, evidence, mra_cfg)
    event_bus.emit_skill_complete("MRA Meta-Audit", progress_pct=60)

    # ── Step 7: Accept or intervene? ─────────────────────────────────
    if should_accept_without_intervention(audit, mra_cfg):
        logger.info("Meta-trust %.2f >= threshold, accepting without intervention",
                     audit.meta_trust)
        mra_result = finalize_early(
            base_output, reflection=reflection, audit=audit,
            status="accepted_without_intervention",
            raw_workflow_result=result,
        )
        _save_mra_result(asset.cache_dir, mra_result)
        return _merge_result(result, mra_result)

    # ── Step 8: Select intervention ──────────────────────────────────
    intervention = select_best_intervention(audit, mra_cfg)
    if intervention is None:
        logger.info("No suitable intervention found")
        mra_result = finalize_early(
            base_output, reflection=reflection, audit=audit,
            status="no_intervention_possible",
            raw_workflow_result=result,
        )
        _save_mra_result(asset.cache_dir, mra_result)
        return _merge_result(result, mra_result)

    itype = intervention.get("type", "unknown")
    logger.info("Selected intervention: %s (score=%.2f)", itype, intervention.get("score", 0))

    # ── Step 9: Execute intervention ─────────────────────────────────
    event_bus.emit_skill_start(f"MRA Intervention: {itype}", progress_pct=65)
    updated_evidence = execute_intervention(
        asset, query, base_output, evidence, intervention,
        llm_model, llm_base_url, mra_cfg,
    )
    event_bus.emit_skill_complete(f"MRA Intervention: {itype}", progress_pct=75)

    # ── Step 10: Local re-reasoning ──────────────────────────────────
    event_bus.emit_skill_start("MRA Re-reasoning", progress_pct=75)
    new_output = rerun_local_reasoning(
        query, base_output, updated_evidence, intervention,
        llm_model, llm_base_url, mra_cfg,
    )
    event_bus.emit_skill_complete("MRA Re-reasoning", progress_pct=80)

    # ── Step 11: New reflection ──────────────────────────────────────
    event_bus.emit_skill_start("MRA Post-Intervention Reflection", progress_pct=80)
    new_reflection = generate_reflection(
        query, new_output, updated_evidence, llm_model, llm_base_url, mra_cfg,
    )
    event_bus.emit_skill_complete("MRA Post-Intervention Reflection", progress_pct=85)

    # ── Step 12: Delta evaluation ────────────────────────────────────
    delta = evaluate_delta(
        base_output, reflection, new_output, new_reflection,
        updated_evidence, intervention,
    )

    # ── Step 13: Final decision ──────────────────────────────────────
    event_bus.emit_skill_start("MRA Final Decision", progress_pct=90)
    mra_result = finalize_decision(
        base_output=base_output,
        new_output=new_output,
        reflection=reflection,
        new_reflection=new_reflection,
        audit=audit,
        delta=delta,
        intervention_type=itype,
        config=mra_cfg,
        raw_workflow_result=result,
    )
    event_bus.emit_skill_complete("MRA Final Decision", progress_pct=95)

    _save_mra_result(asset.cache_dir, mra_result)
    event_bus.emit_progress("MRA: complete", 100)

    return _merge_result(result, mra_result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_base_workflow(asset: Any, base_mode: str,
                       query: str | None, cfg: dict) -> dict:
    """Run the underlying base workflow (brief/detailed/ask)."""
    if query and base_mode in ("brief", "detailed"):
        # If there is a query, also attempt ask workflow, but need an index first.
        # For MVP, just run base analysis — ask requires index which requires analysis.
        pass

    if base_mode in ("brief", "detailed"):
        from agent.extensions.workflows.analyze import wf_analyze
        return wf_analyze(
            asset, base_mode,
            llm_base_url=cfg.get("llm_base_url", "http://localhost:8000/v1"),
            llm_model=cfg.get("llm_model", "qwen3.5-9b"),
            max_frames=cfg.get("max_frames", 128),
            whisper_model=cfg.get("whisper_model"),
            direct_model=cfg.get("direct_model", False),
            model_path=cfg.get("model_path"),
            tokenizer_path=cfg.get("tokenizer_path"),
            include_web_search=cfg.get("include_web_search", False),
            google_api_key=cfg.get("google_api_key"),
            google_search_engine_id=cfg.get("google_search_engine_id"),
            frame_strategy=cfg.get("frame_strategy"),
            frame_fps=cfg.get("frame_fps"),
            force_visual=cfg.get("force_visual"),
        )

    raise ValueError(f"Unsupported MRA base mode: {base_mode}")


def _save_mra_result(cache_dir: str, mra_result: MRAResult) -> None:
    """Persist the MRA audit result to disk."""
    try:
        path = os.path.join(cache_dir, "mra_audit.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mra_result.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        logger.info("MRA result saved to %s", path)
    except Exception as exc:
        logger.warning("Failed to save MRA result: %s", exc)


def _merge_result(workflow_result: dict, mra_result: MRAResult) -> dict:
    """Merge MRA result into the workflow result for backward compatibility."""
    merged = dict(workflow_result)
    merged["mra"] = mra_result.model_dump()
    return merged
