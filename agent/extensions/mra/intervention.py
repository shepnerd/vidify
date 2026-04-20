"""Intervention selection, execution, and local re-reasoning."""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict

from agent.extensions.mra.schemas import (
    BaseOutput, Claim, ClaimReview, EvidenceBundle,
)

logger = logging.getLogger(__name__)


def select_best_intervention(audit_result: Any,
                             config: dict) -> dict | None:
    """Pick the top-1 intervention from the audit recommendation."""
    rec = audit_result.recommended_intervention
    if rec is None:
        return None

    itype = rec.get("type", "")
    supported = config.get("supported_interventions", [])
    if itype not in supported:
        return None

    return rec


def execute_intervention(asset: Any,
                         query: str | None,
                         base_output: BaseOutput,
                         evidence: EvidenceBundle,
                         intervention: dict,
                         llm_model: str,
                         llm_base_url: str,
                         config: dict) -> EvidenceBundle:
    """Execute the selected intervention and return an updated evidence bundle."""

    itype = intervention.get("type", "")
    review = intervention.get("review", {})
    span = review.get("time_span")
    objects = review.get("objects", [])

    updated = EvidenceBundle(**evidence.model_dump())

    if itype == "dense_frame_resample":
        _do_dense_frame_resample(asset, span, updated, llm_model, llm_base_url, config)
    elif itype == "zoom_region":
        _do_zoom_region(asset, span, objects, evidence, updated, llm_model, llm_base_url, config)
    elif itype == "rerun_tracker_or_detector":
        _do_rerun_tracker_or_detector(asset, span, updated, config)
    elif itype == "evidence_only_rereason":
        # This intervention doesn't update evidence, it re-reasons
        pass
    else:
        logger.warning("Unknown intervention type: %s", itype)

    return updated


def rerun_local_reasoning(query: str | None,
                          base_output: BaseOutput,
                          updated_evidence: EvidenceBundle,
                          intervention: dict,
                          llm_model: str,
                          llm_base_url: str,
                          config: dict) -> BaseOutput:
    """Run localized re-reasoning on the target claim only."""

    review = intervention.get("review", {})
    target_claim_id = review.get("claim_id", "")
    itype = intervention.get("type", "")

    new_out = BaseOutput(**base_output.model_dump())

    if itype == "evidence_only_rereason":
        new_out = _evidence_only_rereason(
            query, base_output, target_claim_id, updated_evidence,
            llm_model, llm_base_url,
        )
    else:
        new_out = _visual_rereason(
            query, base_output, target_claim_id, updated_evidence,
            llm_model, llm_base_url,
        )

    return new_out


# ---------------------------------------------------------------------------
# Intervention implementations
# ---------------------------------------------------------------------------

def _do_dense_frame_resample(asset: Any, span: list | None,
                              updated: EvidenceBundle,
                              llm_model: str, llm_base_url: str,
                              config: dict) -> None:
    """Resample frames at higher density within the target span."""
    if not span or len(span) < 2:
        return

    try:
        from agent.extensions.skills.frame_sampler import sample_frames
        from agent.extensions.skills.vision_caption import caption_frames
        from agent.core.schemas import FrameStrategy

        frames_dir = os.path.join(asset.cache_dir, "mra_frames", "dense")
        os.makedirs(frames_dir, exist_ok=True)

        stride = config.get("frame_stride", 4)
        strategy = FrameStrategy(
            type="fps",
            params={"fps": stride, "max_frames": 32},
        )

        frames = sample_frames(
            asset, frames_dir, strategy,
            start_sec=span[0], end_sec=span[1],
        )
        frames = caption_frames(
            frames, llm_model, llm_base_url, batch_size=8,
        )

        # Merge into evidence
        for item in frames.items:
            fid = f"mra_dense_{item.id}"
            updated.frame_meta[fid] = {
                "ts": item.ts,
                "has_caption": item.caption is not None,
                "caption_len": len(item.caption) if item.caption else 0,
                "source": "dense_resample",
            }

        logger.info("Dense resample: added %d frames in span [%.1f, %.1f]",
                     len(frames.items), span[0], span[1])
    except Exception as exc:
        logger.warning("Dense frame resample failed: %s", exc)


def _do_zoom_region(asset: Any, span: list | None,
                     objects: list, evidence: EvidenceBundle,
                     updated: EvidenceBundle,
                     llm_model: str, llm_base_url: str,
                     config: dict) -> None:
    """Crop and zoom on the region of interest, then caption."""
    if not span or len(span) < 2:
        return

    try:
        import cv2
        from agent.extensions.skills.frame_sampler import sample_frames
        from agent.extensions.skills.vision_caption import caption_frames
        from agent.core.schemas import FrameStrategy, FrameItem, FrameSet

        frames_dir = os.path.join(asset.cache_dir, "mra_frames", "zoom")
        os.makedirs(frames_dir, exist_ok=True)

        # Sample a few frames in the span
        strategy = FrameStrategy(type="fps", params={"fps": 2, "max_frames": 8})
        raw_frames = sample_frames(
            asset, os.path.join(frames_dir, "raw"), strategy,
            start_sec=span[0], end_sec=span[1],
        )

        # Estimate crop region from detection bounding boxes
        bbox = _estimate_crop_region(evidence, objects, span, config)

        # Crop each frame
        zoomed_items = []
        zoom_size = config.get("zoom_size", 336)
        for item in raw_frames.items:
            if not os.path.exists(item.path):
                continue
            img = cv2.imread(item.path)
            if img is None:
                continue

            h, w = img.shape[:2]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (zoom_size, zoom_size))
            out_path = os.path.join(frames_dir, f"zoom_{item.id}.jpg")
            cv2.imwrite(out_path, crop)
            zoomed_items.append(FrameItem(
                id=f"zoom_{item.id}", ts=item.ts, path=out_path,
            ))

        if zoomed_items:
            zoomed_fs = FrameSet(
                items=zoomed_items,
                strategy=FrameStrategy(type="scene", params={"source": "zoom"}),
            )
            zoomed_fs = caption_frames(zoomed_fs, llm_model, llm_base_url, batch_size=4)

            for item in zoomed_fs.items:
                updated.frame_meta[item.id] = {
                    "ts": item.ts,
                    "has_caption": item.caption is not None,
                    "caption_len": len(item.caption) if item.caption else 0,
                    "source": "zoom_region",
                }

            logger.info("Zoom region: added %d cropped frames", len(zoomed_items))

    except ImportError:
        logger.warning("cv2 not available, skipping zoom_region intervention")
    except Exception as exc:
        logger.warning("Zoom region failed: %s", exc)


def _do_rerun_tracker_or_detector(asset: Any, span: list | None,
                                    updated: EvidenceBundle,
                                    config: dict) -> None:
    """Re-run object detection on frames in the target span."""
    if not span or len(span) < 2:
        return

    try:
        from agent.extensions.skills.frame_sampler import sample_frames
        from agent.extensions.skills.object_detection import detect_objects_in_video_frames
        from agent.core.schemas import FrameStrategy

        frames_dir = os.path.join(asset.cache_dir, "mra_frames", "redetect")
        os.makedirs(frames_dir, exist_ok=True)

        strategy = FrameStrategy(type="fps", params={"fps": 2, "max_frames": 16})
        frames = sample_frames(
            asset, frames_dir, strategy,
            start_sec=span[0], end_sec=span[1],
        )

        paths = [item.path for item in frames.items if os.path.exists(item.path)]
        if paths:
            dets = detect_objects_in_video_frames(paths)
            updated.detection_results.update(dets)

            # Update tracks
            for fpath, frame_dets in dets.items():
                for d in frame_dets:
                    cls = d.get("class", "unknown")
                    conf = d.get("confidence", 0)
                    if cls not in updated.tracks:
                        updated.tracks[cls] = {
                            "avg_conf": conf,
                            "frame_count": 1,
                            "conf_values": [conf],
                        }
                    else:
                        t = updated.tracks[cls]
                        t["conf_values"].append(conf)
                        t["frame_count"] = len(t["conf_values"])
                        t["avg_conf"] = sum(t["conf_values"]) / t["frame_count"]

            logger.info("Re-detection: processed %d frames in span [%.1f, %.1f]",
                         len(paths), span[0], span[1])

    except ImportError:
        logger.warning("Object detection not available, skipping rerun_tracker_or_detector")
    except Exception as exc:
        logger.warning("Re-detection failed: %s", exc)


# ---------------------------------------------------------------------------
# Re-reasoning helpers
# ---------------------------------------------------------------------------

def _evidence_only_rereason(query: str | None,
                             base_output: BaseOutput,
                             target_claim_id: str,
                             evidence: EvidenceBundle,
                             llm_model: str,
                             llm_base_url: str) -> BaseOutput:
    """Re-reason using only structured evidence, no free-form generation."""
    from agent.extensions.models.vllm_openai_client import make_client, _is_qwen35
    from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body
    from agent.extensions.mra.prompts import build_evidence_only_rereason_prompt

    claim = None
    for c in base_output.claims:
        if c.claim_id == target_claim_id:
            claim = c
            break

    if claim is None:
        return base_output

    # Build evidence subset for the target claim
    evidence_subset = {
        "frame_meta": {k: v for k, v in evidence.frame_meta.items()
                       if v.get("has_caption")},
        "tracks": dict(list(evidence.tracks.items())[:10]),
        "detection_summary": {k: len(v) for k, v in evidence.detection_results.items()},
        "ocr_spans": evidence.ocr_spans[:5],
    }

    prompt = build_evidence_only_rereason_prompt(query, claim.model_dump(), evidence_subset)

    client = make_client(llm_base_url)
    kwargs = {}
    if _is_qwen35(llm_model):
        kwargs["extra_body"] = make_no_thinking_extra_body()

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=500,
            **kwargs,
        )
        raw = strip_thinking(resp.choices[0].message.content.strip())
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("Evidence-only rereason failed: %s", exc)
        return base_output

    new_out = BaseOutput(**base_output.model_dump())
    new_conf = float(data.get("confidence", claim.confidence))
    new_answer = data.get("answer", base_output.answer)

    for c in new_out.claims:
        if c.claim_id == target_claim_id:
            c.confidence = new_conf
            c.text = new_answer[:200]
            break

    new_out.answer = new_answer[:500]
    new_out.answer_confidence = new_conf
    return new_out


def _visual_rereason(query: str | None,
                      base_output: BaseOutput,
                      target_claim_id: str,
                      evidence: EvidenceBundle,
                      llm_model: str,
                      llm_base_url: str) -> BaseOutput:
    """Re-reason using new visual evidence (frames from intervention)."""
    try:
        from agent.extensions.skills.mm_qa import video_frames_qa
        from agent.core.schemas import FrameItem

        # Collect intervention frames
        intervention_frames = []
        for fid, meta in evidence.frame_meta.items():
            source = meta.get("source", "")
            if source in ("dense_resample", "zoom_region"):
                # Need to find the actual path — stored during intervention
                # For now, use the frame id pattern
                intervention_frames.append(FrameItem(
                    id=fid, ts=meta.get("ts", 0),
                    path=meta.get("path", ""),
                    caption=None,
                ))

        if not intervention_frames:
            return base_output

        q = query or f"Is the following claim correct? {base_output.answer[:200]}"
        result_text = video_frames_qa(intervention_frames, q, llm_model, llm_base_url)

        new_out = BaseOutput(**base_output.model_dump())
        # Parse confidence from result if possible
        try:
            result_data = json.loads(result_text)
            new_conf = float(result_data.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            new_conf = 0.5

        new_out.answer = result_text[:500]
        new_out.answer_confidence = new_conf
        for c in new_out.claims:
            if c.claim_id == target_claim_id:
                c.confidence = new_conf
                break

        return new_out

    except Exception as exc:
        logger.warning("Visual rereason failed: %s", exc)
        return base_output


# ---------------------------------------------------------------------------
# Crop region estimation
# ---------------------------------------------------------------------------

def _estimate_crop_region(evidence: EvidenceBundle,
                           objects: list,
                           span: list,
                           config: dict) -> tuple:
    """Estimate a normalized crop region [x1, y1, x2, y2] from detections.

    Falls back to center crop if no detections match.
    """
    bboxes = []
    for fpath, dets in evidence.detection_results.items():
        for d in dets:
            cls = d.get("class", "")
            # Check if this detection matches any target object
            if objects and not any(obj.split("#")[0].lower() in cls.lower() for obj in objects):
                continue
            bbox = d.get("bbox", [])
            if len(bbox) == 4:
                bboxes.append(bbox)

    if not bboxes:
        # Center crop fallback (normalized coords)
        return (0.15, 0.15, 0.85, 0.85)

    # Union of all matching bboxes (assumes pixel coords, normalize later)
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)

    # Add padding (10%)
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, x1 - 0.1 * w)
    y1 = max(0, y1 - 0.1 * h)
    x2 = x2 + 0.1 * w
    y2 = y2 + 0.1 * h

    # These are pixel coords from YOLO, normalize assuming typical resolution
    # The actual normalization happens in the caller using img dimensions
    return (x1, y1, x2, y2)
