# models/vllm_openai_client.py
import os
from openai import OpenAI
from agent.core.retry import retry_with_backoff

# Strip cluster proxy env vars that corrupt multimodal POST payloads.
# Must happen before httpx internalises the env (first OpenAI() call).
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)


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
                     max_tokens: int = 512, temperature: float = 0.2):
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
def chat_with_video(client: OpenAI, model: str, prompt: str, video_path: str,
                    max_tokens: int = 512, temperature: float = 0.2):
    """Send a single video file to a multimodal model (e.g. Qwen3-VL)."""
    from agent.extensions.utils import make_video_content
    content = [
        {"type": "text", "text": prompt},
        make_video_content(video_path),
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content
