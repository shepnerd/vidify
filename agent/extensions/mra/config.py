"""MRA-specific configuration defaults and loader."""

from __future__ import annotations

from typing import Any, Dict


def get_default_mra_config() -> Dict[str, Any]:
    return {
        "frame_stride": 4,
        "max_intervention_rounds": 1,
        "zoom_size": 336,
        "meta_trust_accept": 0.75,
        "meta_trust_uncertain": 0.45,
        "min_claim_conf_for_reflect": 0.80,
        "base_mode": "brief",
        "supported_error_types": [
            "visual_ambiguity",
            "temporal_boundary_error",
            "tracking_failure",
            "ocr_ambiguity",
            "language_prior_bias",
        ],
        "supported_interventions": [
            "dense_frame_resample",
            "zoom_region",
            "rerun_tracker_or_detector",
            "evidence_only_rereason",
        ],
    }


def load_mra_config(workflows_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    defaults = get_default_mra_config()
    if workflows_config:
        overrides = workflows_config.get("audit", {})
        if isinstance(overrides, dict):
            defaults.update(overrides)
    return defaults
