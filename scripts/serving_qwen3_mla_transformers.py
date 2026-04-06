#!/usr/bin/env python3
"""OpenAI-compatible API server for Qwen3-MLA using HuggingFace Transformers.

vLLM cannot serve qwen3-mla because MLA replaces GQA with a different attention
mechanism (kv_a_proj_with_mqa + kv_b_proj instead of standard q/k/v projections).
This server provides an OpenAI-compatible /v1/chat/completions endpoint using
the transformers library directly with trust_remote_code.

Usage:
    python scripts/serving_qwen3_mla_transformers.py                     # defaults
    python scripts/serving_qwen3_mla_transformers.py --port 8001         # custom port
    python scripts/serving_qwen3_mla_transformers.py --model /path/to    # custom model
    python scripts/serving_qwen3_mla_transformers.py --tp 2              # 2 GPUs
"""
import argparse
import json
import os
import sys
import time
import uuid
from threading import Lock
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Strip proxy env vars
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)

app = FastAPI(title="Qwen3-MLA Server")

# Global model state
_model = None
_processor = None
_model_name = ""
_lock = Lock()

# ── Pydantic models for OpenAI API compatibility ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: object  # str or list of dicts (multimodal)

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_completion_tokens: Optional[int] = Field(default=512, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    stream: bool = False
    extra_body: Optional[dict] = None

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(model_path: str, tp_size: int = 1):
    """Load qwen3-mla model with trust_remote_code."""
    global _model, _processor, _model_name
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[model] Loading model from {model_path} ...")
    t0 = time.time()

    if tp_size > 1:
        # Multi-GPU: requires accelerate
        try:
            import accelerate  # noqa: F401
            device_map = "auto"
        except ImportError:
            print("[model] WARNING: 'accelerate' not installed. Using single GPU.")
            device_map = "cuda:0"
    else:
        device_map = "cuda:0"

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        trust_remote_code=True,
    )
    _model.eval()

    # Use Qwen3-VL processor (same tokenizer/processor)
    # The MLA model uses the same vocab and chat template
    try:
        _processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception:
        # Fallback: use the original Qwen3-VL processor
        qwen3vl_path = os.environ.get(
            "QWEN3_PROCESSOR_PATH",
            "/mnt/shared-storage-user/sfteval/sfteval_models/Qwen3-VL-8B-Instruct/"
        )
        print(f"[model] Falling back to processor from {qwen3vl_path}")
        _processor = AutoProcessor.from_pretrained(
            qwen3vl_path, trust_remote_code=True
        )

    _model_name = os.path.basename(model_path.rstrip("/")) or "qwen3-mla"
    elapsed = time.time() - t0
    print(f"[model] Model loaded in {elapsed:.1f}s (device_map={device_map})")


# ── Vision utilities ────────────────────────────────────────────────────────

def _process_multimodal_messages(messages: list[ChatMessage]):
    """Convert OpenAI-format messages with images/videos to qwen_vl format."""
    from qwen_vl_utils import process_vision_info

    qwen_messages = []
    for msg in messages:
        if isinstance(msg.content, str):
            qwen_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
        elif isinstance(msg.content, list):
            content = []
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("file://"):
                            url = url[7:]
                        content.append({"type": "image", "image": url})
                    elif part.get("type") == "video_url":
                        url = part["video_url"]["url"]
                        if url.startswith("file://"):
                            url = url[7:]
                        content.append({
                            "type": "video",
                            "video": url,
                            "fps": 1,
                            "min_pixels": 128 * 32 * 32,
                            "max_pixels": 128 * 32 * 32,
                        })
                else:
                    content.append({"type": "text", "text": str(part)})
            qwen_messages.append({"role": msg.role, "content": content})
        else:
            qwen_messages.append({"role": msg.role, "content": [{"type": "text", "text": str(msg.content)}]})

    # Apply chat template
    text = _processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True,
    )

    # Process vision info using qwen_vl_utils
    images, videos, video_kwargs = process_vision_info(
        qwen_messages, return_video_kwargs=True,
    )

    device = next(_model.parameters()).device

    # Build processor kwargs
    proc_kwargs = {}
    proc_images = images if images else None
    proc_videos = None

    if videos:
        # qwen_vl_utils may return nested lists for videos
        if isinstance(videos[0], (list, tuple)):
            proc_videos = videos[0][0]
            if len(videos[0]) > 1:
                proc_kwargs.update(videos[0][1])
        else:
            proc_videos = videos

        if proc_videos is not None and hasattr(proc_videos, 'to'):
            proc_videos = proc_videos.to(device)
            proc_kwargs["do_resize"] = False

    inputs = _processor(
        text=text, images=proc_images, videos=proc_videos,
        return_tensors="pt", **proc_kwargs,
    )
    return inputs.to(device)


def _process_text_messages(messages: list[ChatMessage]):
    """Convert text-only messages."""
    qwen_messages = []
    for msg in messages:
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        qwen_messages.append({"role": msg.role, "content": text})

    text = _processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    device = next(_model.parameters()).device
    inputs = _processor(text=text, return_tensors="pt")
    return inputs.to(device)


def _has_vision_content(messages: list[ChatMessage]) -> bool:
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "video_url"):
                    return True
    return False


# ── API endpoints ───────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    return ModelList(data=[ModelInfo(id=_model_name)])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    with _lock:
        t0 = time.time()

        max_tokens = request.max_completion_tokens or 512

        # Build inputs
        if _has_vision_content(request.messages):
            inputs = _process_multimodal_messages(request.messages)
        else:
            inputs = _process_text_messages(request.messages)

        prompt_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(request.temperature, 0.01),
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.temperature > 0,
                use_cache=True,
            )

        # Decode
        generated_ids = output[0][prompt_len:]
        text = _processor.batch_decode(
            [generated_ids], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        completion_tokens = len(generated_ids)
        elapsed = time.time() - t0

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=_model_name,
            choices=[Choice(
                message=ChatMessage(role="assistant", content=text),
            )],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                total_tokens=prompt_len + completion_tokens,
            ),
        )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model path (default: models/qwen3-mla or checkpoint)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (num GPUs)")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        # Try wrapper directory first, then original checkpoint
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        candidates = [
            os.path.join(project_root, "models/qwen3-mla"),
            "/mnt/shared-storage-gpfs2/sfteval/xtuner_saved_model/internvl3.5/ablate_wuyue2/20260331093205/hf-5615",
        ]
        for c in candidates:
            if os.path.isdir(c):
                model_path = c
                break
        if model_path is None:
            print("ERROR: No qwen3-mla model found. Specify --model.")
            sys.exit(1)

    load_model(model_path, tp_size=args.tp)

    print(f"[server] Starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
