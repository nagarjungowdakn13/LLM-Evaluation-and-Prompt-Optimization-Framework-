import unittest

from src.metrics import ExactMatchMetric, RuleBasedMetric, SemanticSimilarityMetric
from src.metrics.rule_based import Rule, default_qa_rules


class TestExactMatch(unittest.TestCase):
    def test_identical_strings_score_one(self):
        m = ExactMatchMetric()
        self.assertEqual(m.score("Hello", "hello"), 1.0)

    def test_case_sensitive_off_by_default(self):
        m = ExactMatchMetric()
        self.assertEqual(m.score("Hello", "HELLO"), 1.0)

    def test_case_sensitive_on(self):
        m = ExactMatchMetric(case_sensitive=True)
        self.assertEqual(m.score("Hello", "hello"), 0.0)

    def test_dict_normalization(self):
        m = ExactMatchMetric()
        a = {"b": 2, "a": 1}
        b = {"a": 1, "b": 2}
        self.assertEqual(m.score(a, b), 1.0)

    def test_different_strings_score_zero(self):
        m = ExactMatchMetric()
        self.assertEqual(m.score("foo", "bar"), 0.0)


class TestSemanticSimilarity(unittest.TestCase):
    def test_identical_text_high(self):
        m = SemanticSimilarityMetric(method="tfidf")
        self.assertGreater(m.score("the cat sat on the mat", "the cat sat on the mat"), 0.99)

    def test_unrelated_text_low(self):
        m = SemanticSimilarityMetric(method="tfidf")
        self.assertLess(
            m.score("Quantum chromodynamics", "My favourite recipe is pasta"),
            0.1,
        )

    def test_partial_overlap_mid_range(self):
        m = SemanticSimilarityMetric(method="tfidf")
        s = m.score("Paris is the capital of France", "Paris is in France")
        self.assertGreater(s, 0.4)
        self.assertLess(s, 1.0)


class TestRuleBased(unittest.TestCase):
    def test_no_rules_scores_one(self):
        m = RuleBasedMetric()
        result = m.score("anything", "anything")
        self.assertEqual(result["score"], 1.0)

    def test_failing_rule(self):
        m = RuleBasedMetric([Rule("never", lambda *_: False)])
        result = m.score("x", "y")
        self.assertEqual(result["score"], 0.0)
        self.assertIn("never", result["failed"])

    def test_default_qa_rules_pass_for_valid_json(self):
        m = RuleBasedMetric(default_qa_rules())
        valid = '{"answer": "42", "confidence": "high", "answerable": true, "sources": []}'
        result = m.score(valid)
        self.assertEqual(result["failed"], [])

    def test_default_qa_rules_fail_for_invalid_json(self):
        m = RuleBasedMetric(default_qa_rules())
        result = m.score("not json")
        self.assertLess(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
