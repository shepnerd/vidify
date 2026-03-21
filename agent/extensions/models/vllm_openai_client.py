# models/vllm_openai_client.py
from openai import OpenAI

def make_client(base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
    return OpenAI(base_url=base_url, api_key=api_key)

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