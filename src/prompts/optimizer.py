from collections import Counter
from statistics import mean
from typing import Iterable


class PromptOptimizer:
    """Compares prompt templates over the same dataset and ranks them.

    The optimizer is intentionally agnostic about *how* an evaluation is
    run — any callable that accepts ``(template, example)`` and returns
    an object with an ``overall`` score and a ``to_dict()`` method works.
    """

    def __init__(self, evaluate_fn):
        self.evaluate_fn = evaluate_fn

    def compare(self, templates: Iterable, dataset: list[dict]) -> dict:
        comparison: dict[str, dict] = {}
        for template in templates:
            per_example = []
            scores = []
            sub_scores: dict[str, list[float]] = {
                "correctness": [],
                "consistency": [],
                "hallucination": [],
            }
            grounding_classes: Counter = Counter()
            error_counter: Counter = Counter()
            for example in dataset:
                result = self.evaluate_fn(template, example)
                per_example.append(result.to_dict())
                scores.append(result.overall)
                sub_scores["correctness"].append(result.correctness["overall"])
                sub_scores["consistency"].append(result.consistency["score"])
                sub_scores["hallucination"].append(result.hallucination["combined_score"])

                gc = result.hallucination.get("classification")
                if gc:
                    grounding_classes[gc] += 1
                for ev in getattr(result, "error_events", []) or []:
                    error_counter[ev.get("category", "unknown")] += 1
            comparison[template.name] = {
                "average_score": mean(scores) if scores else 0.0,
                "average_correctness": mean(sub_scores["correctness"]) if scores else 0.0,
                "average_consistency": mean(sub_scores["consistency"]) if scores else 0.0,
                "average_hallucination": mean(sub_scores["hallucination"]) if scores else 0.0,
                "grounding_classes": dict(grounding_classes),
                "error_counts": dict(error_counter),
                "examples": per_example,
            }
        return comparison

    def rank(self, comparison: dict) -> list[tuple[str, float]]:
        return sorted(
            ((name, data["average_score"]) for name, data in comparison.items()),
            key=lambda x: x[1],
            reverse=True,
        )
