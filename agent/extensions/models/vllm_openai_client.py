# models/vllm_openai_client.py
import os
from openai import OpenAI
from agent.core.retry import retry_with_backoff
from agent.extensions.models.thinking import strip_thinking, make_no_thinking_extra_body

# Strip cluster proxy env vars that corrupt multimodal POST payloads.
# Must happen before httpx internalises the env (first OpenAI() call).
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)


def _is_qwen3_mla(model_name: str) -> bool:
    """Check if the model is a Qwen3-MLA variant (MLA attention, no thinking mode)."""
    name = model_name.lower().replace("-", "").replace("_", "")
    return "qwen3mla" in name or "qwen3vlmla" in name


def _is_qwen35(model_name: str) -> bool:
    """Check if the model is a Qwen3.5 variant (has thinking mode by default).

    Qwen3-MLA is NOT a Qwen3.5 variant — it's based on Qwen3-VL with MLA attention.
    """
    if _is_qwen3_mla(model_name):
        return False
    name = model_name.lower().replace("-", "").replace("_", "")
    return "qwen3.5" in name or "qwen35" in name


def make_client(base_url: str = "http://localhost:8000/v1",
                api_key: str = "EMPTY",
                timeout: float = 120.0) -> OpenAI:
    """Create an OpenAI client pointed at a vLLM endpoint.

    *timeout* guards against MLLM requests on long videos hanging
    indefinitely (default 120 s).
    """
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
def chat_with_images(client: OpenAI, model: str, prompt: str, image_urls: list[str],
                     max_tokens: int = 512, temperature: float = 0.2,
                     disable_thinking: bool = True):
    """Send images + prompt to a multimodal model.

    For Qwen3.5 models, thinking mode is disabled by default for pipeline
    tasks (structured JSON output, captions) where we need clean answers.
    Set *disable_thinking=False* to keep reasoning traces.
    """
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    kwargs = {}
    if _is_qwen35(model) and disable_thinking:
        kwargs["extra_body"] = make_no_thinking_extra_body()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    text = resp.choices[0].message.content
    # Safety net: strip any thinking content that slipped through
    return strip_thinking(text)


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
def chat_with_video(client: OpenAI, model: str, prompt: str, video_path: str,
                    max_tokens: int = 512, temperature: float = 0.2,
                    disable_thinking: bool = True, video_fps: float = None):
    """Send a single video file to a multimodal model (e.g. Qwen3-VL / Qwen3.5).

    For Qwen3.5, *video_fps* can control frame sampling rate via
    ``mm_processor_kwargs``.
    """
    from agent.extensions.utils import make_video_content
    content = [
        {"type": "text", "text": prompt},
        make_video_content(video_path),
    ]

    kwargs = {}
    extra_body = {}
    if _is_qwen35(model):
        if disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if video_fps is not None:
            extra_body["mm_processor_kwargs"] = {
                "fps": video_fps, "do_sample_frames": True,
            }
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    text = resp.choices[0].message.content
    return strip_thinking(text)
