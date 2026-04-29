import hashlib
import json
import re
from typing import Any

from .client import BaseLLMClient, LLMResponse


TEMPLATE_TAG = re.compile(r"\[TEMPLATE_ID:\s*([\w\-]+)\]")
EXAMPLE_TAG = re.compile(r"\[EXAMPLE_ID:\s*([\w\-]+)\]")


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM.

    Looks up canned outputs by (template_id, example_id) tags embedded in
    the rendered prompt. Lets us run the full evaluation pipeline offline
    while still showing meaningful score differences between templates.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.model = self.config.get("model", "mock-llm")
        self._registry: dict[tuple[str, str], list[str]] = {}
        self._call_counts: dict[tuple[str, str], int] = {}

    def register_dataset(self, examples: list[dict]) -> None:
        for ex in examples:
            example_id = ex["id"]
            for template_id, responses in (ex.get("mock_responses") or {}).items():
                if isinstance(responses, str):
                    responses = [responses]
                self._registry[(template_id, example_id)] = list(responses)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        tmpl = TEMPLATE_TAG.search(prompt)
        ex = EXAMPLE_TAG.search(prompt)
        template_id = tmpl.group(1) if tmpl else "unknown"
        example_id = ex.group(1) if ex else "unknown"

        responses = self._registry.get((template_id, example_id))
        if responses:
            idx = self._call_counts.get((template_id, example_id), 0) % len(responses)
            self._call_counts[(template_id, example_id)] = idx + 1
            text = responses[idx]
        else:
            text = self._fallback(prompt)

        return LLMResponse(
            text=text,
            model=self.model,
            metadata={"template_id": template_id, "example_id": example_id},
        )

    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        text = response.text or ""
        if not text:
            return
        for index in range(0, len(text), 18):
            yield text[index : index + 18]

    def _fallback(self, prompt: str) -> str:
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        return json.dumps(
            {
                "answer": f"Unable to determine an answer (mock-{digest}).",
                "confidence": "low",
                "sources": [],
                "answerable": False,
            }
        )
