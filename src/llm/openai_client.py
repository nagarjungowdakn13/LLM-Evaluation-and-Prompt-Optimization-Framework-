from typing import Any

import os

from .client import BaseLLMClient, LLMResponse, call_with_retry


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: dict):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e
        self.client = OpenAI(
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=config.get("base_url") or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.retries = int(config.get("retries", 3))
        self.retry_base_delay = float(config.get("retry_base_delay", 0.5))

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        def _do_request():
            return self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"} if kwargs.get("json_mode") else None,
            )

        resp = call_with_retry(_do_request, retries=int(kwargs.get("retries", self.retries)), base_delay=float(kwargs.get("retry_base_delay", self.retry_base_delay)))
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            metadata={"usage": getattr(resp, "usage", None)},
        )
