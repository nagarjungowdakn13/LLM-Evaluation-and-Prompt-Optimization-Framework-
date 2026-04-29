"""Experiment tracking on top of the SQLite ``RunDatabase``.

What the tracker adds beyond the raw database:

- a single ``record_run`` API that captures hyperparameters, dataset
  signature, and prompt versions alongside the existing comparison +
  ranking payload,
- ``leaderboard`` and ``compare_runs`` helpers used by the dashboard and
  CLI for cross-run analytics,
- a ``best_templates_across_runs`` view that surfaces the strongest
  prompt regardless of which run produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.storage import RunDatabase


@dataclass
class RunDelta:
    a: str
    b: str
    score_delta: float
    template_changes: dict[str, float]


class ExperimentTracker:
    def __init__(self, db: RunDatabase | None = None, sqlite_path: str | None = None):
        self.db = db or RunDatabase(sqlite_path or "reports/runs.sqlite3")

    # ---------- recording ----------

    def record_run(
        self,
        *,
        comparison: dict,
        ranking: list[tuple[str, float]],
        config: dict,
        dataset: list[dict] | None = None,
        prompt_versions: dict[str, str] | None = None,
        report_paths: dict[str, str] | None = None,
        extra: dict | None = None,
    ) -> str:
        config = dict(config or {})
        config.setdefault("hyperparameters", {})
        config["hyperparameters"].update(
            {
                "consistency_runs": config.get("evaluation", {}).get("consistency_runs"),
                "weights": config.get("evaluation", {}).get("weights"),
                "min_grounded_ratio": (
                    config.get("evaluation", {}).get("hallucination", {}).get("min_grounded_ratio")
                ),
                "semantic_method": (
                    config.get("metrics", {}).get("semantic_similarity", {}).get("method")
                ),
            }
        )
        if dataset is not None:
            config["dataset_signature"] = self._dataset_signature(dataset)
        if prompt_versions:
            config["prompt_versions"] = prompt_versions
        if extra:
            config["extra"] = extra
        return self.db.save_run(
            comparison=comparison,
            ranking=ranking,
            config=config,
            report_paths=report_paths,
        )

    # ---------- analytics ----------

    def leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        runs = self.db.list_runs(limit=limit)
        # Order by best_score for cross-run leaderboard view.
        return sorted(
            runs,
            key=lambda r: (r.get("summary") or {}).get("best_score") or 0.0,
            reverse=True,
        )

    def compare_runs(self, run_a: str, run_b: str) -> RunDelta:
        a = self.db.load_run(run_a)
        b = self.db.load_run(run_b)
        if not a or not b:
            raise KeyError(f"unknown run id(s): {run_a if not a else run_b}")
        a_scores = {name: data["average_score"] for name, data in a["comparison"].items()}
        b_scores = {name: data["average_score"] for name, data in b["comparison"].items()}
        names = set(a_scores) | set(b_scores)
        template_changes = {
            n: round(b_scores.get(n, 0.0) - a_scores.get(n, 0.0), 4) for n in sorted(names)
        }
        delta = (b["summary"].get("best_score") or 0.0) - (a["summary"].get("best_score") or 0.0)
        return RunDelta(a=run_a, b=run_b, score_delta=round(delta, 4), template_changes=template_changes)

    def best_templates_across_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        runs = self.db.list_runs(limit=limit)
        rows: list[dict[str, Any]] = []
        for r in runs:
            full = self.db.load_run(r["run_id"])
            if not full:
                continue
            for name, data in full["comparison"].items():
                rows.append(
                    {
                        "run_id": r["run_id"],
                        "template": name,
                        "score": data.get("average_score"),
                        "model": full.get("model"),
                        "created_at": r["created_at"],
                    }
                )
        rows.sort(key=lambda x: x["score"] or 0.0, reverse=True)
        return rows

    # ---------- helpers ----------

    @staticmethod
    def _dataset_signature(dataset: list[dict]) -> dict:
        ids = [str(ex.get("id")) for ex in dataset]
        h = hashlib.sha1(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:12]
        return {"size": len(ids), "id_hash": h, "first_id": ids[0] if ids else None}
