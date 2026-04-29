import json
import tempfile
import unittest
from pathlib import Path

from src.experiment_tracker import ExperimentTracker
from src.storage import RunDatabase


def _payload(best_score: float, templates: dict[str, float]) -> dict:
    return {
        "comparison": {
            name: {
                "average_score": score,
                "examples": [],
            }
            for name, score in templates.items()
        },
        "ranking": sorted(templates.items(), key=lambda kv: kv[1], reverse=True),
    }


class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "runs.sqlite3"
        self.tracker = ExperimentTracker(db=RunDatabase(self.db_path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_compare(self):
        a = _payload(0.6, {"alpha": 0.6, "beta": 0.5})
        b = _payload(0.8, {"alpha": 0.7, "beta": 0.8})
        ra = self.tracker.record_run(comparison=a["comparison"], ranking=a["ranking"], config={"llm": {"provider": "mock", "model": "m"}})
        rb = self.tracker.record_run(comparison=b["comparison"], ranking=b["ranking"], config={"llm": {"provider": "mock", "model": "m"}})

        delta = self.tracker.compare_runs(ra, rb)
        self.assertAlmostEqual(delta.score_delta, 0.2, places=4)
        self.assertAlmostEqual(delta.template_changes["alpha"], 0.1, places=4)
        self.assertAlmostEqual(delta.template_changes["beta"], 0.3, places=4)

    def test_leaderboard_orders_by_best_score(self):
        self.tracker.record_run(comparison={"x": {"average_score": 0.4, "examples": []}}, ranking=[("x", 0.4)], config={})
        self.tracker.record_run(comparison={"y": {"average_score": 0.9, "examples": []}}, ranking=[("y", 0.9)], config={})
        lb = self.tracker.leaderboard()
        self.assertEqual(lb[0]["summary"]["best_score"], 0.9)

    def test_dataset_signature_recorded(self):
        run_id = self.tracker.record_run(
            comparison={"x": {"average_score": 0.5, "examples": []}},
            ranking=[("x", 0.5)],
            config={},
            dataset=[{"id": "a"}, {"id": "b"}],
            prompt_versions={"x": "v1"},
        )
        loaded = self.tracker.db.load_run(run_id)
        sig = loaded["config"]["dataset_signature"]
        self.assertEqual(sig["size"], 2)
        self.assertEqual(loaded["config"]["prompt_versions"], {"x": "v1"})


if __name__ == "__main__":
    unittest.main()
