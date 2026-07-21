import hashlib
import json
import re
from typing import Any

from .client import BaseLLMClient, LLMResponse


TEMPLATE_TAG = re.compile(r"\[TEMPLATE_ID:\s*([\w\-]+)\]")
EXAMPLE_TAG = re.compile(r"\[EXAMPLE_ID:\s*([\w\-]+)\]")


class MockLLMClient(BaseLLMClient):
    """Deterministic, multi-model mock LLM.

    Supports offline evaluation for Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, and GPT-4o
    across Baseline, CoT, Self-Consistency, Self-Verification, and Retrieval-Grounded techniques.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.model = self.config.get("model", "llama-3.1-8b").lower()
        self._registry: dict[tuple[str, str], list[str]] = {}
        self._dataset_samples: list[dict] = []
        self._call_counts: dict[tuple[str, str], int] = {}

    def register_dataset(self, examples: list[dict]) -> None:
        self._dataset_samples = examples
        for ex in examples:
            example_id = ex["id"]
            # Legacy mock_responses mapping
            for template_id, responses in (ex.get("mock_responses") or {}).items():
                if isinstance(responses, str):
                    responses = [responses]
                self._registry[(template_id, example_id)] = list(responses)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        tmpl = TEMPLATE_TAG.search(prompt)
        ex = EXAMPLE_TAG.search(prompt)
        template_id = tmpl.group(1) if tmpl else "unknown"
        example_id = ex.group(1) if ex else "unknown"

        text = None

        # 1. Try structured mock_responses_by_model_tech lookup
        for sample in self._dataset_samples:
            if sample.get("id") == example_id:
                model_responses = sample.get("mock_responses_by_model_tech", {}).get(self.model, {})
                # Normalize tech key (e.g. tech_cot -> cot, qa_v4_cot -> cot)
                tech_key = template_id.replace("tech_", "").replace("qa_v4_", "").replace("qa_v3_", "").replace("qa_v1_", "")
                if tech_key in model_responses:
                    resp = model_responses[tech_key]
                    if isinstance(resp, list):
                        idx = self._call_counts.get((template_id, example_id), 0) % len(resp)
                        self._call_counts[(template_id, example_id)] = idx + 1
                        text = resp[idx]
                    else:
                        text = str(resp)
                    break

        # 2. Try registry fallback
        if not text:
            responses = self._registry.get((template_id, example_id))
            if responses:
                idx = self._call_counts.get((template_id, example_id), 0) % len(responses)
                self._call_counts[(template_id, example_id)] = idx + 1
                text = responses[idx]

        # 3. Dynamic fallback if not found
        if not text:
            text = self._fallback(prompt)

        return LLMResponse(
            text=text,
            model=self.model,
            metadata={"template_id": template_id, "example_id": example_id, "model": self.model},
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
                "answer": f"Unable to determine an answer (mock-{self.model}-{digest}).",
                "confidence": "low",
                "sources": [],
                "answerable": False,
            }
        )
