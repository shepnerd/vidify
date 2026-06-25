"""In-process vision-language client used by interactive chat.

The historic chat CLI imports ``TransformersVLClient`` for direct mode.  The
project's supported in-process backend is ``DirectModelLoader``, so this module
keeps the public import path stable while adapting chat's smaller interface.
"""
from __future__ import annotations

from typing import Any

from agent.extensions.models.thinking import strip_thinking


class TransformersVLClient:
    """Adapter for direct, in-process model inference.

    The name is kept for backward compatibility with the chat CLI.  Internally
    it uses ``make_direct_client`` so direct analysis and direct chat follow the
    same model-loading path.
    """

    def __init__(self, model_path: str, dtype: str = "auto",
                 tokenizer_path: str | None = None, **kwargs: Any):
        from agent.extensions.models.direct_model_loader import make_direct_client

        self.model_name = model_path
        self.dtype = dtype
        load_kwargs = dict(kwargs)
        if dtype and dtype != "auto":
            load_kwargs.setdefault("dtype", dtype)
        self._client = make_direct_client(
            model_path,
            tokenizer_path=tokenizer_path,
            **load_kwargs,
        )

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 512,
             temperature: float = 0.2) -> str:
        prompt = self._messages_to_prompt(messages)
        text = self._client.chat_with_images(
            self.model_name,
            prompt,
            [],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return strip_thinking(text).strip()

    def chat_with_images(self, prompt: str, image_paths: list[str],
                         max_tokens: int = 512,
                         temperature: float = 0.2) -> str:
        image_urls = [
            path if path.startswith(("file://", "http://", "https://", "data:"))
            else f"file://{path}"
            for path in image_paths
        ]
        text = self._client.chat_with_images(
            self.model_name,
            prompt,
            image_urls,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return strip_thinking(text).strip()

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip() or "user"
            content = TransformersVLClient._content_to_text(message.get("content", ""))
            if content:
                parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunk for chunk in chunks if chunk)
        return str(content)
