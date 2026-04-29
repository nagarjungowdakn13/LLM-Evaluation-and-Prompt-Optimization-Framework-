from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class RunDatabase:
    def __init__(self, path: str | Path = "reports/runs.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    summary_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    ranking_json TEXT NOT NULL,
                    comparison_json TEXT NOT NULL,
                    report_paths_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def save_run(
        self,
        *,
        comparison: dict,
        ranking: list[tuple[str, float]],
        config: dict,
        report_paths: dict[str, str] | None = None,
    ) -> str:
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        provider = config.get("llm", {}).get("provider", "mock")
        model = config.get("llm", {}).get("model", "unknown")
        summary = {
            "best_template": ranking[0][0] if ranking else None,
            "best_score": ranking[0][1] if ranking else None,
            "template_count": len(comparison),
            "example_count": len(next(iter(comparison.values()))["examples"]) if comparison else 0,
        }
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": provider,
            "model": model,
            "summary_json": self._dump(summary),
            "config_json": self._dump(config),
            "ranking_json": self._dump(ranking),
            "comparison_json": self._dump(comparison),
            "report_paths_json": self._dump(report_paths or {}),
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_at, provider, model, summary_json, config_json,
                    ranking_json, comparison_json, report_paths_json
                ) VALUES (:run_id, :created_at, :provider, :model, :summary_json, :config_json,
                         :ranking_json, :comparison_json, :report_paths_json)
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()
        return run_id

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT run_id, created_at, provider, model, summary_json, report_paths_json
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "run_id": row["run_id"],
                    "created_at": row["created_at"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "summary": json.loads(row["summary_json"]),
                    "report_paths": json.loads(row["report_paths_json"]),
                }
            )
        return result

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "provider": row["provider"],
            "model": row["model"],
            "summary": json.loads(row["summary_json"]),
            "config": json.loads(row["config_json"]),
            "ranking": [tuple(item) for item in json.loads(row["ranking_json"])],
            "comparison": json.loads(row["comparison_json"]),
            "report_paths": json.loads(row["report_paths_json"]),
        }