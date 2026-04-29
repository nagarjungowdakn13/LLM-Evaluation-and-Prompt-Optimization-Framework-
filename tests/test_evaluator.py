import unittest

from src.evaluator import (
    CorrectnessEvaluator,
    HallucinationDetector,
    EvaluationRunner,
)
from src.llm.mock_client import MockLLMClient
from src.metrics import ExactMatchMetric, RuleBasedMetric, SemanticSimilarityMetric
from src.metrics.rule_based import default_qa_rules
from src.prompts import PromptManager, PromptTemplate
from src.schema import SchemaValidator


SCHEMA = {
    "type": "object",
    "required": ["answer", "confidence", "answerable", "sources"],
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "answerable": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
}


class TestCorrectness(unittest.TestCase):
    def test_combines_metrics(self):
        ev = CorrectnessEvaluator(
            exact_match=ExactMatchMetric(),
            semantic=SemanticSimilarityMetric(),
            rule_based=RuleBasedMetric(default_qa_rules()),
            schema_validator=SchemaValidator(SCHEMA),
        )
        text = '{"answer": "42", "confidence": "high", "answerable": true, "sources": []}'
        expected = {"answer": "42", "confidence": "high", "answerable": True, "sources": []}
        result = ev.evaluate(text, expected)
        self.assertTrue(result["schema"]["valid"])
        self.assertGreater(result["overall"], 0.7)


class TestHallucination(unittest.TestCase):
    def test_detects_ungrounded_terms(self):
        det = HallucinationDetector(min_grounded_ratio=0.7)
        result = det.detect(
            predicted="Paris was founded by Napoleon in 1066",
            expected="Paris is the capital of France",
            context="Paris is the capital of France.",
        )
        self.assertIn("napoleon", result["ungrounded_terms"])
        self.assertTrue(result["is_hallucinated"])

    def test_grounded_answer_passes(self):
        det = HallucinationDetector(min_grounded_ratio=0.7)
        result = det.detect(
            predicted="Paris is in France",
            expected="Paris is the capital of France",
            context="Paris is the capital of France.",
        )
        self.assertFalse(result["is_hallucinated"])


class TestEvaluationRunner(unittest.TestCase):
    def test_full_pipeline_with_mock(self):
        example = {
            "id": "t-1",
            "context": "The sky is blue because of Rayleigh scattering.",
            "question": "Why is the sky blue?",
            "expected_output": {
                "answer": "Because of Rayleigh scattering.",
                "confidence": "high",
                "answerable": True,
                "sources": ["Rayleigh scattering"],
            },
            "mock_responses": {
                "tmpl": [
                    '{"answer": "Because of Rayleigh scattering.", "confidence": "high", "answerable": true, "sources": ["Rayleigh scattering"]}'
                ]
            },
        }
        client = MockLLMClient({"model": "mock"})
        client.register_dataset([example])

        ev = CorrectnessEvaluator(
            exact_match=ExactMatchMetric(),
            semantic=SemanticSimilarityMetric(),
            rule_based=RuleBasedMetric(default_qa_rules()),
            schema_validator=SchemaValidator(SCHEMA),
        )
        runner = EvaluationRunner(
            llm_client=client,
            correctness=ev,
            hallucination=HallucinationDetector(),
            consistency_runs=2,
        )
        pm = PromptManager()
        pm.register(PromptTemplate(name="tmpl", body="Q: {question}\nC: {context}\nA:"))

        result = runner.evaluate(pm.get("tmpl"), example)
        self.assertEqual(result.template_id, "tmpl")
        self.assertGreater(result.correctness["semantic_similarity"], 0.7)
        self.assertGreater(result.consistency["score"], 0.99)
        self.assertGreater(result.overall, 0.7)


if __name__ == "__main__":
    unittest.main()
