import tempfile
import unittest
from pathlib import Path

from src.prompts import PromptManager, PromptTemplate, PromptOptimizer


class _StubResult:
    def __init__(self, score):
        self.overall = score
        self.correctness = {"overall": score}
        self.consistency = {"score": score}
        self.hallucination = {"combined_score": score}

    def to_dict(self):
        return {
            "overall": self.overall,
            "correctness": self.correctness,
            "consistency": self.consistency,
            "hallucination": self.hallucination,
            "example_id": "ex",
            "output": "out",
        }


class TestPromptManager(unittest.TestCase):
    def test_register_and_get(self):
        pm = PromptManager()
        pm.register(PromptTemplate("t", "Hello {name}"))
        self.assertEqual(pm.get("t").body, "Hello {name}")

    def test_render_injects_routing_tags(self):
        pm = PromptManager()
        pm.register(PromptTemplate("greet", "Hi {name}"))
        out = pm.get("greet").render({"id": "ex-1", "name": "world"})
        self.assertIn("[TEMPLATE_ID: greet]", out)
        self.assertIn("[EXAMPLE_ID: ex-1]", out)
        self.assertIn("Hi world", out)

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("alpha")
            (Path(d) / "b.txt").write_text("beta")
            pm = PromptManager().load_directory(d)
            self.assertEqual(set(pm.templates), {"a", "b"})


class TestPromptOptimizer(unittest.TestCase):
    def test_ranking_orders_by_score(self):
        scores = {"good": 0.9, "ok": 0.6, "bad": 0.2}

        def fake_eval(template, _example):
            return _StubResult(scores[template.name])

        opt = PromptOptimizer(evaluate_fn=fake_eval)
        templates = [PromptTemplate(n, "x") for n in scores]
        comparison = opt.compare(templates, [{"id": "ex"}])
        ranking = opt.rank(comparison)
        self.assertEqual([n for n, _ in ranking], ["good", "ok", "bad"])


if __name__ == "__main__":
    unittest.main()
