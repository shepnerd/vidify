# agent/workflows/ask.py
import os
import json
import logging
import subprocess
from openai import OpenAI
from agent.extensions.models.vllm_openai_client import make_client, resolve_model_name
from agent.extensions.models.direct_model_loader import make_direct_client
from agent.extensions.skills.persist import load_analysis
from agent.extensions.skills.rag_faiss import search_faiss
from agent.core.schemas import FrameItem, FrameSet, FrameStrategy

logger = logging.getLogger(__name__)

# Keywords that suggest the question needs visual information
_VISUAL_KEYWORDS = {
    "look", "see", "show", "display", "screen", "board", "slide",
    "image", "picture", "color", "wear", "face", "background",
    "written", "text on", "equation", "diagram", "chart", "graph",
    "logo", "sign", "gesture", "scene", "appear", "visible",
    "看", "显示", "屏幕", "画面", "图", "公式", "黑板", "白板",
    "穿", "颜色", "长什么样", "幻灯片",
}


def needs_visual(question: str) -> bool:
    """Check if a question likely requires visual information to answer."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _VISUAL_KEYWORDS)


def targeted_visual_lookup(asset, timestamps: list[tuple[float, float]],
                           llm_model: str, llm_base_url: str,
                           direct_model: bool = False,
                           model_path: str = None,
                           tokenizer_path: str = None) -> list[FrameItem]:
    """Sample and caption frames only from specific timestamp ranges.

    Instead of processing the entire video, extracts 1-2 frames per range
    and runs MLLM captioning on just those frames.
    """
    from agent.extensions.skills.vision_caption import caption_frames

    frames_dir = os.path.join(asset.cache_dir, "targeted_frames")
    os.makedirs(frames_dir, exist_ok=True)

    items = []
    for i, (start, end) in enumerate(timestamps):
        # Sample 1 frame at the midpoint of each range
        mid = (start + end) / 2
        out_path = os.path.join(frames_dir, f"target_{i:04d}.jpg")
        if not os.path.exists(out_path):
            cmd = [
                "ffmpeg", "-y", "-ss", str(mid),
                "-i", asset.local_path,
                "-frames:v", "1",
                "-vf", "scale=512:288:force_original_aspect_ratio=decrease",
                "-q:v", "2",
                out_path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=True, timeout=30)
            except Exception:
                continue
        if os.path.exists(out_path):
            items.append(FrameItem(id=f"target_{i:04d}", ts=mid, path=out_path))

    if not items:
        return []

    # Caption these targeted frames using MLLM
    frameset = FrameSet(items=items, strategy=FrameStrategy(type="scene", params={"source": "targeted"}))
    captioned = caption_frames(frameset, llm_model, llm_base_url, batch_size=len(items),
                               direct_model=direct_model, model_path=model_path,
                               tokenizer_path=tokenizer_path)
    return captioned.items


def wf_ask(asset, question: str,
           llm_base_url: str, llm_model: str,
           embed_base_url: str, embed_model: str,
           top_k: int = 5,
           direct_model: bool = False,
           model_path: str = None,
           tokenizer_path: str = None) -> dict:
    if not direct_model:
        llm_model = resolve_model_name(llm_model, llm_base_url)

    analysis = load_analysis(asset.cache_dir)
    rag = (analysis.get("rag") or {}).get("faiss")
    if not rag:
        raise RuntimeError("No FAISS index. Run wf_index first.")
    hits = search_faiss(rag["index_dir"], question, embed_base_url, embed_model, top_k=top_k)

    # Targeted visual lookup: if the question needs visual info, sample and caption
    # specific frames from the relevant time ranges instead of the entire video
    visual_context = []
    if needs_visual(question):
        logger.info("Question requires visual info, running targeted visual lookup...")
        timestamps = [(h["start"], h["end"]) for h in hits[:3] if "start" in h and "end" in h]
        if timestamps:
            visual_frames = targeted_visual_lookup(
                asset, timestamps, llm_model, llm_base_url,
                direct_model=direct_model, model_path=model_path,
                tokenizer_path=tokenizer_path
            )
            visual_context = [
                {"ts": f.ts, "caption": f.caption}
                for f in visual_frames if f.caption
            ]
            logger.info("Got %d visual frame captions for context.", len(visual_context))

    payload = {"question": question, "chunks": hits}
    if visual_context:
        payload["visual_context"] = visual_context

    if direct_model:
        client = make_direct_client(model_path, tokenizer_path)
        text = client.chat_with_images(llm_model, json.dumps(payload, ensure_ascii=False), [],
                                       max_tokens=800, temperature=0.2)
    else:
        client = make_client(llm_base_url)
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}],
            temperature=0.2,
            max_completion_tokens=800,
        )
        text = resp.choices[0].message.content.strip()

    try:
        result = json.loads(text)
    except Exception:
        result = {"answer": text, "evidence": []}
    return {"result": result, "hits": hits, "visual_context": visual_context}
