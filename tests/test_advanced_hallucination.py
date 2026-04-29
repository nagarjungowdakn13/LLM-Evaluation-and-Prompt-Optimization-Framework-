import unittest

from src.hallucination_detector import AdvancedHallucinationDetector, GroundingClass


class TestAdvancedDetector(unittest.TestCase):
    def setUp(self):
        self.det = AdvancedHallucinationDetector()

    def test_grounded_answer(self):
        result = self.det.detect(
            predicted="Paris is the capital of France",
            expected="Paris is the capital of France",
            context="Paris is the capital of France.",
        )
        self.assertEqual(result["classification"], GroundingClass.GROUNDED.value)
        self.assertFalse(result["is_hallucinated"])

    def test_hallucinated_when_unsupported_proper_nouns(self):
        result = self.det.detect(
            predicted="Paris was founded by Napoleon and Caesar in Cairo",
            expected="Paris is the capital of France",
            context="Paris is the capital of France.",
        )
        self.assertEqual(result["classification"], GroundingClass.HALLUCINATED.value)
        self.assertTrue(result["is_hallucinated"])
        self.assertTrue(result["unsupported_claims"])

    def test_partially_grounded(self):
        result = self.det.detect(
            predicted="Paris is in France and was founded long ago",
            expected="Paris is the capital of France",
            context="Paris is the capital of France.",
        )
        # Some drift, no proper-noun fabrication → partially grounded.
        self.assertIn(
            result["classification"],
            {GroundingClass.PARTIALLY_GROUNDED.value, GroundingClass.GROUNDED.value},
        )

    def test_missing_entities_detected(self):
        result = self.det.detect(
            predicted="The capital is in France.",
            expected={"answer": "Paris is in France", "sources": ["Paris"]},
            context="Paris is in France.",
        )
        self.assertIn("Paris", result["missing_entities"])

    def test_signals_present(self):
        result = self.det.detect(
            predicted="The Pacific is the largest ocean.",
            expected="The Pacific is the largest ocean.",
            context="Earth has five oceans. The Pacific is the largest.",
        )
        for key in ("grounding", "entity_recall", "claim_support"):
            self.assertIn(key, result["signals"])

    def test_legacy_keys_preserved(self):
        # Old code paths read score/ungrounded_terms/ungrounded_ratio/is_hallucinated
        result = self.det.detect("a", "a", context="a")
        for key in ("score", "ungrounded_terms", "ungrounded_ratio", "is_hallucinated"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
