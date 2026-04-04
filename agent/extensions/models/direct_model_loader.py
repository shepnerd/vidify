import base64
import io
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from agent.extensions.models.thinking import strip_thinking

# Global cache for loaded models to avoid reloading
_MODEL_CACHE = {}

class DirectModelLoader:
    def __init__(self, model_path: str, tokenizer_path: str = None, **kwargs):
        # Set default memory utilization to avoid OOM
        kwargs.setdefault('gpu_memory_utilization', 0.3)  # Increase to allow KV cache
        kwargs.setdefault('max_model_len', 4096)  # Qwen3.5 supports up to 262K but use conservative default
        kwargs.setdefault('enforce_eager', True)  # Use eager mode to save memory
        kwargs.setdefault('max_num_seqs', 1)  # Limit concurrent sequences

        cache_key = (model_path, tokenizer_path, frozenset(kwargs.items()))
        if cache_key in _MODEL_CACHE:
            self.model, self.tokenizer = _MODEL_CACHE[cache_key]
        else:
            self.model = LLM(model=model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
            _MODEL_CACHE[cache_key] = (self.model, self.tokenizer)

    def _img_to_data_url(self, path: str, max_w=512, max_h=256, fmt="JPEG", quality=85) -> str:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"

    def chat_with_images(self, model_name: str, prompt: str, image_urls: list[str],
                         max_tokens: int = 512, temperature: float = 0.2) -> str:
        messages = [{"role": "user", "content": []}]
        messages[0]["content"].append({"type": "text", "text": prompt})
        for url in image_urls:
            if url.startswith("file://"):
                local_path = url[7:]
                data_url = self._img_to_data_url(local_path)
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": url}})

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.model.generate([input_text], sampling_params)
        text = outputs[0].outputs[0].text
        # Strip thinking content from Qwen3.5 responses
        return strip_thinking(text)

# Factory function
def make_direct_client(model_path: str, tokenizer_path: str = None, **kwargs):
    return DirectModelLoader(model_path, tokenizer_path, **kwargs)
