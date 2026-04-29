from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
import time
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        ...

    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        text = response.text or ""
        if not text:
            return
        chunk_size = max(1, int(kwargs.get("chunk_size", 24)))
        for index in range(0, len(text), chunk_size):
            yield text[index : index + chunk_size]


def _retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "rate limit",
            "429",
            "timeout",
            "temporarily unavailable",
            "overloaded",
            "service unavailable",
            "connection reset",
        )
    )


def call_with_retry(fn, *, retries: int = 3, base_delay: float = 0.5):
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries - 1 or not _retryable_error(exc):
                raise
            time.sleep(base_delay * (2**attempt))
    if last_error is not None:
        raise last_error


def get_client(config: dict) -> BaseLLMClient:
    provider = (config or {}).get("provider", "mock").lower()
    if provider == "mock":
        from .mock_client import MockLLMClient
        return MockLLMClient(config or {})
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(config or {})
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(config or {})
    raise ValueError(f"Unknown LLM provider: {provider}")
