"""Assemble an EvidenceBundle from existing perception pipeline results.

When opencv-python and Pillow are available (on compute nodes), this module
computes real frame quality metrics: blur (Laplacian variance), brightness,
contrast, and edge density.  On dev nodes without those packages, it falls
back to caption-length proxies.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.extensions.mra.schemas import BaseOutput, EvidenceBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probes — resolved once at import time
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_evidence(asset: Any,
                     base_output: BaseOutput,
                     workflow_result: dict,
                     config: dict) -> EvidenceBundle:
    """Extract and restructure evidence from an already-completed workflow result.

    When cv2 is available and frame image files exist on disk, real quality
    metrics (blur, brightness, contrast) are computed per frame.
    """
    frame_meta = _build_frame_meta(workflow_result)
    detection_results = _build_detection_results(workflow_result)
    tracks = _build_tracks(detection_results)
    ocr_spans = _build_ocr_spans(workflow_result)
    event_candidates = _build_event_candidates(workflow_result)
    support_trace = _build_support_trace(base_output)

    # Enrich frame_meta with real quality metrics from image files
    _enrich_frame_quality(frame_meta, workflow_result)

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
    lines: List[str] = []

    n_frames = len(bundle.frame_meta)
    if n_frames > 0:
        lines.append(f"Frames analysed: {n_frames}")

    if bundle.frame_meta:
        captioned = sum(1 for m in bundle.frame_meta.values() if m.get("has_caption"))
        lines.append(f"Frames with captions: {captioned}/{n_frames}")

        # Detection confidence
        confs = [m["avg_det_conf"] for m in bundle.frame_meta.values()
                 if m.get("avg_det_conf")]
        if confs:
            lines.append(f"Detection confidence range: {min(confs):.2f} - {max(confs):.2f}")

        # Blur scores
        blurs = [m["blur"] for m in bundle.frame_meta.values() if "blur" in m]
        if blurs:
            low_quality = sum(1 for b in blurs if b < 100)
            lines.append(f"Frame quality (blur): min={min(blurs):.0f}, "
                         f"max={max(blurs):.0f}, low-quality={low_quality}/{len(blurs)}")

        # Brightness
        brights = [m["brightness"] for m in bundle.frame_meta.values() if "brightness" in m]
        if brights:
            dark = sum(1 for b in brights if b < 50)
            bright = sum(1 for b in brights if b > 200)
            lines.append(f"Brightness: avg={sum(brights)/len(brights):.0f}, "
                         f"dark={dark}, overexposed={bright}")

    if bundle.detection_results:
        all_classes: set[str] = set()
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


# ===================================================================
# Frame quality estimation (real metrics when cv2+numpy are available)
# ===================================================================

def estimate_frame_quality(image_path: str) -> Dict[str, Any]:
    """Compute quality metrics for a single frame image.

    Returns dict with keys: blur, brightness, contrast, edge_density.
    All values are floats.  If cv2 is unavailable, returns empty dict.
    """
    if not _HAS_CV2 or not _HAS_NP:
        return {}
    if not os.path.isfile(image_path):
        return {}

    try:
        img = _cv2.imread(image_path)
        if img is None:
            return {}
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)

        # Blur: Laplacian variance — higher = sharper, lower = blurrier
        blur = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())

        # Brightness: mean pixel intensity (0-255)
        brightness = float(gray.mean())

        # Contrast: std of pixel intensity
        contrast = float(gray.std())

        # Edge density: fraction of edge pixels (Canny)
        edges = _cv2.Canny(gray, 100, 200)
        edge_density = float(edges.sum() / 255.0) / max(edges.size, 1)

        return {
            "blur": blur,
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
        }
    except Exception as exc:
        logger.debug("Frame quality estimation failed for %s: %s", image_path, exc)
        return {}


def estimate_motion_proxy(frame_path_a: str, frame_path_b: str) -> float:
    """Estimate inter-frame motion as mean absolute pixel difference.

    Returns a float in [0, 255].  High value = high motion / scene change.
    If cv2 is unavailable or either file is missing, returns -1.
    """
    if not _HAS_CV2 or not _HAS_NP:
        return -1.0
    try:
        a = _cv2.imread(frame_path_a, _cv2.IMREAD_GRAYSCALE)
        b = _cv2.imread(frame_path_b, _cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return -1.0
        # Resize to same shape if needed
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = _cv2.resize(a, (w, h))
            b = _cv2.resize(b, (w, h))
        return float(_np.mean(_np.abs(a.astype(float) - b.astype(float))))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enrich_frame_quality(meta: Dict[str, Dict[str, Any]],
                          result: dict) -> None:
    """Enrich frame_meta with real image quality metrics when possible."""
    frames = result.get("frames", {})
    items = frames.get("items", []) if isinstance(frames, dict) else []

    # Build id -> path mapping
    id_to_path: Dict[str, str] = {}
    for item in items:
        fid = item.get("id", "")
        path = item.get("path", "")
        if fid and path:
            id_to_path[fid] = path

    enriched_count = 0
    prev_path: str | None = None
    for fid, fmeta in meta.items():
        path = id_to_path.get(fid, "")
        if not path:
            continue

        quality = estimate_frame_quality(path)
        if quality:
            fmeta.update(quality)
            enriched_count += 1

        # Motion proxy vs previous frame
        if prev_path is not None:
            motion = estimate_motion_proxy(prev_path, path)
            if motion >= 0:
                fmeta["motion"] = motion
        prev_path = path

    if enriched_count > 0:
        logger.info("Enriched %d/%d frames with quality metrics", enriched_count, len(meta))
    elif meta and _HAS_CV2:
        logger.debug("No frame files found on disk for quality estimation")


def _build_frame_meta(result: dict) -> Dict[str, Dict[str, Any]]:
    """Build per-frame metadata from workflow result."""
    meta: Dict[str, Dict[str, Any]] = {}
    frames = result.get("frames", {})
    items = frames.get("items", []) if isinstance(frames, dict) else []

    for item in items:
        fid = item.get("id", "")
        ts = item.get("ts", 0)
        caption = item.get("caption")
        path = item.get("path", "")
        meta[fid] = {
            "ts": ts,
            "path": path,
            "has_caption": caption is not None and len(str(caption)) > 0,
            "caption_len": len(str(caption)) if caption else 0,
        }

    # Merge detection confidence if objects are present
    objects = result.get("objects", {})
    if isinstance(objects, dict):
        for fpath, dets in objects.items():
            matched_fid = None
            for fid, fmeta in meta.items():
                if fpath.endswith(fid) or fid in fpath or fmeta.get("path") == fpath:
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
