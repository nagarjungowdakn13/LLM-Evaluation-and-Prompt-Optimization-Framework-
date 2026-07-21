"""Cost-Quality Tradeoff Analysis for Hallucination Reduction Study."""

from __future__ import annotations
from typing import Any


MODEL_COST_PER_1M_TOKENS = {
    "llama-3.1-8b": {"input": 0.15, "output": 0.15},
    "mistral-7b": {"input": 0.15, "output": 0.15},
    "qwen-2.5-7b": {"input": 0.15, "output": 0.15},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

TECHNIQUE_TOKEN_MULTIPLIER = {
    "baseline": {"input_mult": 1.0, "output_mult": 1.0, "calls": 1},
    "cot": {"input_mult": 1.2, "output_mult": 2.8, "calls": 1},
    "self_consistency": {"input_mult": 5.0, "output_mult": 5.0, "calls": 5},
    "self_verification": {"input_mult": 2.2, "output_mult": 2.5, "calls": 2},
    "retrieval_grounded": {"input_mult": 2.5, "output_mult": 1.1, "calls": 1},
}


class CostAnalyzer:
    """Calculates estimated token consumption and monetary cost per technique and model."""

    @staticmethod
    def estimate_cost(
        model: str,
        technique: str,
        base_input_tokens: int = 250,
        base_output_tokens: int = 80,
        n_queries: int = 100
    ) -> dict[str, Any]:
        mdl = model.lower()
        tech = technique.lower().replace("tech_", "").replace("qa_v4_", "").replace("qa_v3_", "").replace("qa_v1_", "")

        pricing = MODEL_COST_PER_1M_TOKENS.get(mdl, {"input": 0.20, "output": 0.20})
        tech_mult = TECHNIQUE_TOKEN_MULTIPLIER.get(tech, {"input_mult": 1.0, "output_mult": 1.0, "calls": 1})

        input_tokens_per_query = int(base_input_tokens * tech_mult["input_mult"])
        output_tokens_per_query = int(base_output_tokens * tech_mult["output_mult"])

        total_input_tokens = input_tokens_per_query * n_queries
        total_output_tokens = output_tokens_per_query * n_queries

        cost_input = (total_input_tokens / 1_000_000) * pricing["input"]
        cost_output = (total_output_tokens / 1_000_000) * pricing["output"]
        total_cost_usd = cost_input + cost_output

        return {
            "model": model,
            "technique": technique,
            "n_queries": n_queries,
            "input_tokens_per_query": input_tokens_per_query,
            "output_tokens_per_query": output_tokens_per_query,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost_usd": round(total_cost_usd, 4),
            "cost_per_1k_queries": round(total_cost_usd * 10, 4),
            "relative_cost_multiplier": round(tech_mult["input_mult"] + tech_mult["output_mult"], 1),
        }

    @classmethod
    def compute_efficiency_score(cls, accuracy: float, hallucination_rate: float, cost_usd: float) -> float:
        """Computes cost-efficiency index: (Accuracy * (1 - Hallucination_Rate)) / max(Cost, 0.001)"""
        quality = accuracy * (1.0 - hallucination_rate)
        return round(quality / max(cost_usd, 0.0001), 2)
