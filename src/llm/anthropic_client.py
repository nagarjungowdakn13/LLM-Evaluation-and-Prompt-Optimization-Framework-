from typing import Any

import os

from .client import BaseLLMClient, LLMResponse, call_with_retry


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: dict):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        self.client = anthropic.Anthropic(
            api_key=config.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
            base_url=config.get("base_url") or os.getenv("ANTHROPIC_BASE_URL"),
        )
        self.model = config.get("model", "claude-haiku-4-5-20251001")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.retries = int(config.get("retries", 3))
        self.retry_base_delay = float(config.get("retry_base_delay", 0.5))

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        def _do_request():
            return self.client.messages.create(
                model=kwargs.get("model", self.model),
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )

        resp = call_with_retry(_do_request, retries=int(kwargs.get("retries", self.retries)), base_delay=float(kwargs.get("retry_base_delay", self.retry_base_delay)))
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return LLMResponse(
            text=text,
            model=resp.model,
            metadata={"usage": resp.usage.model_dump() if resp.usage else None},
        )
