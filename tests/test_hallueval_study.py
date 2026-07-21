"""Unit tests for the hallueval package and StudyRunner."""

import unittest
from pathlib import Path
from src.study import StudyRunner, CostAnalyzer, TaxonomyExporter
from src.datasets import BenchmarkRegistry


class TestHalluevalStudy(unittest.TestCase):

    def test_benchmark_registry(self):
        datasets = BenchmarkRegistry.list_datasets()
        self.assertIn("truthfulqa", datasets)
        self.assertIn("halueval", datasets)

        tqa = BenchmarkRegistry.get_dataset("truthfulqa")
        self.assertEqual(tqa.name, "TruthfulQA")
        self.assertGreater(len(tqa.get_examples()), 0)

        halu = BenchmarkRegistry.get_dataset("halueval")
        self.assertEqual(halu.name, "HaluEval")
        self.assertGreater(len(halu.get_examples()), 0)

    def test_cost_analyzer(self):
        cost_info = CostAnalyzer.estimate_cost("llama-3.1-8b", "baseline", n_queries=100)
        self.assertEqual(cost_info["model"], "llama-3.1-8b")
        self.assertGreater(cost_info["total_cost_usd"], 0)

        eff = CostAnalyzer.compute_efficiency_score(0.9, 0.1, 0.05)
        self.assertGreater(eff, 0)

    def test_taxonomy_exporter(self):
        result = {
            "output": "Swallowed gum stays in your stomach for 7 years.",
            "expected": {"answerable": True},
            "hallucination": {"combined_score": 0.2, "ungrounded_terms": ["7", "years"]},
            "correctness": {"overall": 0.3},
            "metadata": {"dataset": "TruthfulQA", "model": "llama-3.1-8b", "technique": "baseline"}
        }
        cat = TaxonomyExporter.classify_error_type(result)
        self.assertEqual(cat, "factual_confabulation")

    def test_study_runner_execution(self):
        runner = StudyRunner()
        results = runner.run_study(provider="mock", models=["llama-3.1-8b"], datasets=["truthfulqa"])
        self.assertEqual(results.total_cells, 5)  # 5 techniques x 1 model x 1 dataset
        self.assertTrue(Path("reports/study_results_grid.json").exists())
        self.assertTrue(Path("reports/error_taxonomy_analysis.csv").exists())


if __name__ == "__main__":
    unittest.main()
