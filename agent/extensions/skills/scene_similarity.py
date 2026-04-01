"""Embedding-based scene-change detection using CLIP.

Inspired by OmniLive's SigLIP-based segmentation: a new segment starts when
cosine similarity between consecutive frame embeddings drops below a threshold
(default 0.9), with min/max segment length constraints.

Falls back to simple pixel-difference heuristic if CLIP is unavailable.
"""
import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

_clip_model = None
_clip_preprocess = None
_clip_available: Optional[bool] = None


def _load_clip():
    """Lazy-load CLIP model. Returns (model, preprocess) or (None, None)."""
    global _clip_model, _clip_preprocess, _clip_available
    if _clip_available is not None:
        return _clip_model, _clip_preprocess

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_available = True
        logger.info("CLIP model loaded (open_clip ViT-B-32)")
    except ImportError:
        try:
            import clip as openai_clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
            _clip_model = model
            _clip_preprocess = preprocess
            _clip_available = True
            logger.info("CLIP model loaded (openai clip ViT-B/32)")
        except ImportError:
            _clip_available = False
            logger.warning("No CLIP library available; falling back to pixel-diff heuristic")

    return _clip_model, _clip_preprocess


def compute_frame_embedding(frame_path: str) -> np.ndarray:
    """Compute a normalized embedding vector for a single frame.

    Returns a 1-D float32 numpy array. Uses CLIP if available,
    otherwise a simple downscaled pixel histogram.
    """
    model, preprocess = _load_clip()

    if model is not None and preprocess is not None:
        return _clip_embedding(frame_path, model, preprocess)
    return _pixel_embedding(frame_path)


def _clip_embedding(frame_path: str, model, preprocess) -> np.ndarray:
    import torch
    from PIL import Image

    image = Image.open(frame_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten().astype(np.float32)


def _pixel_embedding(frame_path: str) -> np.ndarray:
    """Fallback: downscale to 8x8 grayscale and flatten as a pseudo-embedding."""
    import cv2
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(64, dtype=np.float32)
    resized = cv2.resize(img, (8, 8)).astype(np.float32).flatten()
    norm = np.linalg.norm(resized)
    if norm > 0:
        resized /= norm
    return resized


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def is_scene_change(prev_embedding: np.ndarray, curr_embedding: np.ndarray,
                    threshold: float = 0.9) -> bool:
    """Return True if the scene has changed (similarity below threshold)."""
    sim = cosine_similarity(prev_embedding, curr_embedding)
    return sim < threshold


def segment_video_by_similarity(
    frame_paths: List[str],
    threshold: float = 0.9,
    min_frames: int = 3,
    max_frames: int = 16,
) -> List[Tuple[int, int]]:
    """Segment a list of frame paths into groups based on visual similarity.

    Returns a list of (start_idx, end_idx) tuples (end exclusive).
    A new segment starts when:
      - At least min_frames have accumulated AND
      - Either max_frames reached OR similarity drops below threshold
    """
    if not frame_paths:
        return []

    segments: List[Tuple[int, int]] = []
    seg_start = 0
    prev_emb = compute_frame_embedding(frame_paths[0])

    for i in range(1, len(frame_paths)):
        curr_emb = compute_frame_embedding(frame_paths[i])
        frames_in_seg = i - seg_start

        need_new_seg = False
        if frames_in_seg >= min_frames:
            if frames_in_seg >= max_frames:
                need_new_seg = True
            elif is_scene_change(prev_emb, curr_emb, threshold):
                need_new_seg = True

        if need_new_seg:
            segments.append((seg_start, i))
            seg_start = i

        prev_emb = curr_emb

    # Close last segment
    if seg_start < len(frame_paths):
        segments.append((seg_start, len(frame_paths)))

    return segments
