import unittest
from collections import Counter

from src.error_taxonomy import ErrorCategory, ErrorEvent
from src.optimization import OptimizationLoop, mutate_for_failures
from src.prompts import PromptTemplate


class TestMutators(unittest.TestCase):
    def test_no_failures_no_change(self):
        body = "Be helpful."
        new, label = mutate_for_failures(body, {"by_category": {}})
        self.assertEqual(new, body)
        self.assertEqual(label, "")

    def test_hallucination_adds_grounding_guard(self):
        summary = {
            "by_category": {
                ErrorCategory.HALLUCINATION.value: {"count": 3, "by_severity": {"high": 3}},
                ErrorCategory.SCHEMA_VIOLATION.value: {"count": 0, "by_severity": {}},
            }
        }
        new, label = mutate_for_failures("Answer the question.", summary)
        self.assertEqual(label, ErrorCategory.HALLUCINATION.value)
        self.assertIn("Only use facts present", new)

    def test_idempotent(self):
        summary = {"by_category": {ErrorCategory.HALLUCINATION.value: {"count": 1, "by_severity": {}}}}
        once, _ = mutate_for_failures("Answer.", summary)
        twice, _ = mutate_for_failures(once, summary)
        self.assertEqual(once, twice)


class _StubResult:
    def __init__(self, score):
        self.overall = score
        self.correctness = {"overall": score}
        self.consistency = {"score": score}
        self.hallucination = {"combined_score": score, "ungrounded_terms": [], "classification": "grounded"}
        self.error_events: list = []

    def to_dict(self):
        return {
            "overall": self.overall, "correctness": self.correctness,
            "consistency": self.consistency, "hallucination": self.hallucination,
            "example_id": "ex", "output": "ok", "expected": {},
            "error_events": self.error_events,
        }


class TestOptimizationLoop(unittest.TestCase):
    def test_loop_runs_and_produces_history(self):
        scores = {"alpha": 0.5}

        def evaluate_fn(template, _example):
            r = _StubResult(scores.get(template.name, 0.5))
            # Force one hallucination on the seed template so a mutation triggers.
            if template.name == "alpha":
                r.error_events = [ErrorEvent(
                    category=ErrorCategory.HALLUCINATION,
                    severity="high",
                    reason="test",
                ).to_dict()]
            return r

        loop = OptimizationLoop(evaluate_fn=evaluate_fn)
        result = loop.run(
            seeds=[PromptTemplate("alpha", "Just answer.")],
            dataset=[{"id": "ex"}],
            max_iterations=2,
        )
        self.assertGreaterEqual(len(result.iterations), 1)
        # Best score history should be populated.
        self.assertTrue(result.score_history)


if __name__ == "__main__":
    unittest.main()
