"""Assemble an EvidenceBundle from existing perception pipeline results."""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.extensions.mra.schemas import BaseOutput, EvidenceBundle

logger = logging.getLogger(__name__)


def collect_evidence(asset: Any,
                     base_output: BaseOutput,
                     workflow_result: dict,
                     config: dict) -> EvidenceBundle:
    """Extract and restructure evidence from an already-completed workflow result."""

    frame_meta = _build_frame_meta(workflow_result)
    detection_results = _build_detection_results(workflow_result)
    tracks = _build_tracks(detection_results)
    ocr_spans = _build_ocr_spans(workflow_result)
    event_candidates = _build_event_candidates(workflow_result)
    support_trace = _build_support_trace(base_output)

    return EvidenceBundle(
        frame_meta=frame_meta,
        tracks=tracks,
        detection_results=detection_results,
        ocr_spans=ocr_spans,
        event_candidates=event_candidates,
        support_trace=support_trace,
    )


def summarize_evidence(bundle: EvidenceBundle) -> str:
    """Produce a compact text summary of the evidence bundle for the reflection prompt."""
    lines = []

    n_frames = len(bundle.frame_meta)
    if n_frames > 0:
        lines.append(f"Frames analysed: {n_frames}")

    if bundle.frame_meta:
        captioned = sum(1 for m in bundle.frame_meta.values() if m.get("has_caption"))
        lines.append(f"Frames with captions: {captioned}/{n_frames}")

        confs = [m.get("avg_det_conf", 0) for m in bundle.frame_meta.values() if m.get("avg_det_conf")]
        if confs:
            lines.append(f"Detection confidence range: {min(confs):.2f} - {max(confs):.2f}")

    if bundle.detection_results:
        all_classes = set()
        for dets in bundle.detection_results.values():
            for d in dets:
                all_classes.add(d.get("class", "?"))
        lines.append(f"Detected object classes: {', '.join(sorted(all_classes)[:10])}")

    if bundle.tracks:
        lines.append(f"Object tracks: {len(bundle.tracks)}")
        for tid, tinfo in list(bundle.tracks.items())[:5]:
            lines.append(f"  {tid}: avg_conf={tinfo.get('avg_conf', '?'):.2f}, "
                         f"frames={tinfo.get('frame_count', '?')}")

    if bundle.ocr_spans:
        lines.append(f"OCR text regions: {len(bundle.ocr_spans)}")

    if bundle.event_candidates:
        lines.append(f"Event candidates: {len(bundle.event_candidates)}")

    return "\n".join(lines) if lines else "No evidence available."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_frame_meta(result: dict) -> Dict[str, Dict[str, Any]]:
    """Build per-frame metadata from workflow result."""
    meta: Dict[str, Dict[str, Any]] = {}
    frames = result.get("frames", {})
    items = frames.get("items", []) if isinstance(frames, dict) else []

    for item in items:
        fid = item.get("id", "")
        ts = item.get("ts", 0)
        caption = item.get("caption")
        meta[fid] = {
            "ts": ts,
            "has_caption": caption is not None and len(str(caption)) > 0,
            "caption_len": len(str(caption)) if caption else 0,
        }

    # Merge detection confidence if objects are present
    objects = result.get("objects", {})
    if isinstance(objects, dict):
        for fpath, dets in objects.items():
            # Try to match frame by path suffix
            matched_fid = None
            for fid, fmeta in meta.items():
                if fpath.endswith(fid) or fid in fpath:
                    matched_fid = fid
                    break
            if matched_fid and isinstance(dets, list):
                confs = [d.get("confidence", 0) for d in dets]
                meta[matched_fid]["avg_det_conf"] = sum(confs) / len(confs) if confs else 0
                meta[matched_fid]["n_detections"] = len(dets)

    return meta


def _build_detection_results(result: dict) -> Dict[str, list]:
    """Extract detection results from workflow output."""
    objects = result.get("objects", {})
    if isinstance(objects, dict):
        return {k: v for k, v in objects.items() if isinstance(v, list)}
    return {}


def _build_tracks(detection_results: dict) -> Dict[str, Dict[str, Any]]:
    """Aggregate detections into pseudo-tracks by object class."""
    class_stats: Dict[str, list] = {}
    for frame_key, dets in detection_results.items():
        for d in dets:
            cls = d.get("class", "unknown")
            conf = d.get("confidence", 0)
            if cls not in class_stats:
                class_stats[cls] = []
            class_stats[cls].append({"frame": frame_key, "conf": conf})

    tracks = {}
    for cls, entries in class_stats.items():
        confs = [e["conf"] for e in entries]
        tracks[cls] = {
            "avg_conf": sum(confs) / len(confs) if confs else 0,
            "frame_count": len(entries),
            "conf_values": confs,
        }
    return tracks


def _build_ocr_spans(result: dict) -> list:
    """Extract OCR results from workflow output."""
    ocr = result.get("ocr", {})
    if isinstance(ocr, list):
        return ocr
    if isinstance(ocr, dict):
        spans = []
        for frame_key, texts in ocr.items():
            if isinstance(texts, list):
                for t in texts:
                    spans.append({"frame": frame_key, "text": t})
            elif isinstance(texts, str):
                spans.append({"frame": frame_key, "text": texts})
        return spans
    return []


def _build_event_candidates(result: dict) -> list:
    """Extract event candidates from timeline chapters/events if available."""
    timeline = result.get("timeline", {})
    candidates = []
    if isinstance(timeline, dict):
        chapters = timeline.get("chapters", [])
        if isinstance(chapters, list):
            for ch in chapters:
                if isinstance(ch, dict):
                    candidates.append({
                        "type": "chapter",
                        "span": [ch.get("start", 0), ch.get("end", 0)],
                        "text": ch.get("title", ch.get("summary", "")),
                    })
    return candidates


def _build_support_trace(base_output: BaseOutput) -> dict:
    """Extract support trace from base output claims."""
    trace = {}
    for claim in base_output.claims:
        trace[claim.claim_id] = {
            "frames": claim.support_refs.get("frames", []),
            "objects": claim.objects,
            "span": claim.span,
        }
    return trace
