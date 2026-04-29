import unittest

from src.error_taxonomy import (
    ErrorCategory,
    ErrorTaxonomy,
    aggregate_distribution,
    classify_evaluation,
)


class TestTaxonomy(unittest.TestCase):
    def setUp(self):
        self.tax = ErrorTaxonomy()

    def test_schema_violation_detected(self):
        result = {
            "correctness": {"schema": {"valid": False, "errors": ["root: 'answer' is a required property"]},
                            "semantic_similarity": 0.9, "exact_match": 0.0},
            "hallucination": {"combined_score": 1.0},
            "expected": {},
            "output": "{}",
        }
        events = self.tax.classify(**result)
        cats = {e.category for e in events}
        self.assertIn(ErrorCategory.SCHEMA_VIOLATION, cats)

    def test_formatting_error_for_non_json(self):
        result = {
            "correctness": {
                "schema": {"valid": False, "errors": ["no JSON object found"]},
                "semantic_similarity": 0.9, "exact_match": 0.0,
                "rule_based": {"failed": ["looks_like_json", "has_required_keys"]},
            },
            "hallucination": {"combined_score": 1.0},
            "expected": {},
            "output": "Hello world (not JSON)",
        }
        events = self.tax.classify(**result)
        cats = {e.category for e in events}
        self.assertIn(ErrorCategory.FORMATTING_ERROR, cats)

    def test_hallucination_when_low_grounding(self):
        result = {
            "correctness": {"semantic_similarity": 0.8, "exact_match": 0.0},
            "hallucination": {"combined_score": 0.3, "ungrounded_terms": ["napoleon", "1066"]},
            "expected": {"answer": "Paris"},
            "output": "Paris was founded by Napoleon in 1066",
        }
        events = self.tax.classify(**result)
        self.assertIn(ErrorCategory.HALLUCINATION, {e.category for e in events})

    def test_unanswerable_but_answer_given(self):
        result = {
            "correctness": {"semantic_similarity": 0.6, "exact_match": 0.0},
            "hallucination": {"combined_score": 0.9},
            "expected": {"answer": "not stated", "answerable": False},
            "output": '{"answer": "AUD 102 million", "answerable": true}',
        }
        events = self.tax.classify(**result)
        self.assertIn(ErrorCategory.HALLUCINATION, {e.category for e in events})

    def test_aggregate_distribution(self):
        events_per_example = [
            self.tax.classify(
                correctness={"schema": {"valid": False, "errors": ["root: 'x'"]}, "semantic_similarity": 0.9, "exact_match": 0.0},
                hallucination={"combined_score": 1.0},
                expected={}, output="{}",
            ),
            self.tax.classify(
                correctness={"semantic_similarity": 0.4, "exact_match": 0.0},
                hallucination={"combined_score": 0.3, "ungrounded_terms": ["x"]},
                expected={"answer": "y"}, output="x",
            ),
        ]
        dist = aggregate_distribution(events_per_example)
        self.assertGreaterEqual(dist["total_events"], 2)
        self.assertIn("hallucination", dist["by_category"])

    def test_classify_evaluation_object_shape(self):
        # Accepts the dict returned by EvaluationResult.to_dict()
        events = classify_evaluation(
            {
                "correctness": {"semantic_similarity": 0.4, "exact_match": 0.0},
                "hallucination": {"combined_score": 0.2, "ungrounded_terms": ["xyz"]},
                "expected": {"answer": "y"},
                "output": "z",
            }
        )
        self.assertTrue(events)


if __name__ == "__main__":
    unittest.main()
