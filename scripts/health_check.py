"""End-to-end health check for the LLM Evaluation Framework.

Run from the project root:

    python scripts/health_check.py

Exits non-zero if anything fails. Designed to be run before a demo or
presentation to confirm every layer is wired correctly.
"""

from __future__ import annotations

import importlib
import json
import tempfile
import sys
import traceback
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


class Check:
    def __init__(self, name: str, fn: Callable[[], None], required: bool = True):
        self.name = name
        self.fn = fn
        self.required = required


def _run(checks: list[Check]) -> int:
    failures = 0
    warnings = 0
    width = max(len(c.name) for c in checks) + 2
    for check in checks:
        try:
            check.fn()
            print(f"{GREEN}OK  {RESET}{check.name.ljust(width)}")
        except Exception as e:  # noqa: BLE001
            tag = f"{RED}FAIL{RESET}" if check.required else f"{YELLOW}SKIP{RESET}"
            print(f"{tag} {check.name.ljust(width)} {DIM}{type(e).__name__}: {e}{RESET}")
            if check.required:
                failures += 1
            else:
                warnings += 1
            if "-v" in sys.argv or "--verbose" in sys.argv:
                traceback.print_exc()
    print()
    print(
        f"{len(checks) - failures - warnings} passed, "
        f"{failures} failed, {warnings} optional skipped"
    )
    return 1 if failures else 0


# ---------- individual checks ----------


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, got {sys.version}")


def check_required_imports() -> None:
    for mod in ("numpy", "pandas", "yaml", "jsonschema"):
        importlib.import_module(mod)


def check_dashboard_imports() -> None:
    importlib.import_module("streamlit")
    importlib.import_module("plotly")


def check_optional_anthropic() -> None:
    importlib.import_module("anthropic")


def check_optional_openai() -> None:
    importlib.import_module("openai")


def check_optional_embeddings() -> None:
    importlib.import_module("sentence_transformers")


def check_project_layout() -> None:
    required = [
        "main.py",
        "dashboard.py",
        "config.yaml",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "data/dataset.json",
        "data/schemas/qa_schema.json",
        "data/prompts/qa_v1_basic.txt",
        "data/prompts/qa_v2_structured.txt",
        "data/prompts/qa_v3_grounded.txt",
        "src/evaluator/runner.py",
        "src/metrics/exact_match.py",
        "src/metrics/semantic_similarity.py",
        "src/metrics/rule_based.py",
        "src/schema/validator.py",
        "src/llm/mock_client.py",
        "src/prompts/manager.py",
        "src/reporting/reporter.py",
        "src/storage/run_db.py",
        "src/error_taxonomy/taxonomy.py",
        "src/hallucination_detector/detector.py",
        "src/prompts/strategies.py",
        "src/optimization/loop.py",
        "src/optimization/mutators.py",
        "src/experiment_tracker/tracker.py",
        "data/prompts/qa_v4_cot.txt",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing files: {missing}")


def check_schema_loads() -> None:
    from src.schema import SchemaValidator, load_schema

    schema = load_schema(ROOT / "data/schemas/qa_schema.json")
    v = SchemaValidator(schema)
    good = '{"answer": "x", "confidence": "high", "answerable": true, "sources": []}'
    if not v.validate(good)["valid"]:
        raise AssertionError("known-good payload rejected")
    if v.validate("not json")["valid"]:
        raise AssertionError("invalid JSON accepted")


def check_metrics() -> None:
    from src.metrics import (
        ExactMatchMetric, RuleBasedMetric, SemanticSimilarityMetric,
    )
    from src.metrics.rule_based import default_qa_rules

    if ExactMatchMetric().score("Hi", "hi") != 1.0:
        raise AssertionError("exact match wrong")
    if SemanticSimilarityMetric().score("a b c", "a b c") < 0.99:
        raise AssertionError("semantic identity wrong")
    rb = RuleBasedMetric(default_qa_rules()).score(
        '{"answer": "x", "confidence": "high", "answerable": true, "sources": []}'
    )
    if rb["score"] < 1.0:
        raise AssertionError(f"rule-based score not 1.0: {rb}")


def check_dataset_loads() -> None:
    payload = json.loads((ROOT / "data/dataset.json").read_text(encoding="utf-8"))
    if not payload.get("examples"):
        raise AssertionError("dataset has no examples")
    for ex in payload["examples"]:
        for field in ("id", "context", "question", "expected_output"):
            if field not in ex:
                raise AssertionError(f"example {ex.get('id')} missing {field}")


def check_full_pipeline() -> None:
    from main import build_runner, load_config, load_dataset
    from src.prompts import PromptManager, PromptOptimizer

    config = load_config(ROOT / "config.yaml")
    dataset = load_dataset(ROOT / config["dataset"]["path"])
    pm = PromptManager().load_directory(ROOT / config["prompts"]["directory"])
    templates = pm.list(config["prompts"]["templates"])
    if len(templates) < 2:
        raise AssertionError("need at least 2 templates to compare")
    runner = build_runner(config, dataset)
    optimizer = PromptOptimizer(evaluate_fn=runner.evaluate)
    comparison = optimizer.compare(templates, dataset)
    ranking = optimizer.rank(comparison)

    # Sanity: ranking is sorted, all scores in [0,1], best != worst on this dataset
    scores = [s for _, s in ranking]
    if scores != sorted(scores, reverse=True):
        raise AssertionError("ranking not sorted")
    if not all(0.0 <= s <= 1.0 for s in scores):
        raise AssertionError(f"scores out of range: {scores}")
    if abs(scores[0] - scores[-1]) < 0.01:
        raise AssertionError("evaluation produced no signal — best == worst")


def check_report_writer() -> None:
    from src.reporting import Reporter

    fake_comparison = {
        "good": {
            "average_score": 0.9, "average_correctness": 0.9,
            "average_consistency": 0.9, "average_hallucination": 0.9,
            "examples": [{
                "example_id": "e1", "overall": 0.9,
                "correctness": {"exact_match": 1.0, "semantic_similarity": 0.9,
                                "schema": {"valid": True, "errors": []},
                                "rule_based": {"score": 1.0, "passed": [], "failed": []}},
                "hallucination": {"combined_score": 0.95, "ungrounded_terms": []},
                "consistency": {"score": 1.0, "n_runs": 3, "outputs": ["x"]},
                "output": "ok",
            }],
        }
    }
    out_dir = ROOT / "reports" / "_health_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(output_dir=out_dir)
    paths = reporter.write(fake_comparison, [("good", 0.9)], formats={"markdown", "json"})
    for p in paths.values():
        if not p.exists() or p.stat().st_size == 0:
            raise AssertionError(f"report file missing or empty: {p}")
        p.unlink()
    out_dir.rmdir()


def check_run_storage() -> None:
    from src.storage import RunDatabase

    with tempfile.TemporaryDirectory() as tmp:
        db = RunDatabase(Path(tmp) / "runs.sqlite3")
        run_id = db.save_run(
            comparison={"a": {"examples": []}},
            ranking=[("a", 1.0)],
            config={"llm": {"provider": "mock", "model": "mock"}},
            report_paths={},
        )
        loaded = db.load_run(run_id)
        if loaded is None or loaded["run_id"] != run_id:
            raise AssertionError("saved run could not be reloaded")
        runs = db.list_runs(limit=5)
        if not runs or runs[0]["run_id"] != run_id:
            raise AssertionError("run history missing latest run")


def check_dashboard_module() -> None:
    # Importing dashboard.py executes all top-level Streamlit calls. Silence
    # the various "no runtime" / "missing ScriptRunContext" messages that
    # Streamlit emits when imported outside `streamlit run`.
    import io
    import contextlib
    import logging

    for name in (
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.caching",
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.scriptrunner_utils",
        "streamlit.runtime.state",
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    sink = io.StringIO()
    with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
        import dashboard

    for attr in ("run_evaluation", "highlight_ungrounded", "PALETTE", "EARTH_SCALE"):
        if not hasattr(dashboard, attr):
            raise AssertionError(f"dashboard missing {attr}")


def check_taxonomy_and_advanced_detector() -> None:
    from src.error_taxonomy import ErrorCategory, ErrorTaxonomy, aggregate_distribution
    from src.hallucination_detector import AdvancedHallucinationDetector, GroundingClass

    tax = ErrorTaxonomy()
    events = tax.classify(
        correctness={"schema": {"valid": False, "errors": ["root: 'x'"]}, "semantic_similarity": 0.9, "exact_match": 0.0},
        hallucination={"combined_score": 1.0},
        expected={}, output="{}",
    )
    if not any(e.category == ErrorCategory.SCHEMA_VIOLATION for e in events):
        raise AssertionError("schema violation not detected by taxonomy")
    dist = aggregate_distribution([events])
    if dist["total_events"] < 1:
        raise AssertionError("aggregate distribution did not count events")

    det = AdvancedHallucinationDetector()
    bad = det.detect(
        predicted="Paris was founded by Napoleon in 1066",
        expected="Paris is the capital of France",
        context="Paris is the capital of France.",
    )
    if bad["classification"] != GroundingClass.HALLUCINATED.value:
        raise AssertionError(f"unsupported claims not classified as hallucinated: {bad}")


def check_optimization_loop() -> None:
    from src.error_taxonomy import ErrorCategory, ErrorEvent
    from src.optimization import OptimizationLoop
    from src.prompts import PromptTemplate

    class _R:
        def __init__(self):
            self.overall = 0.5
            self.correctness = {"overall": 0.5}
            self.consistency = {"score": 0.5}
            self.hallucination = {"combined_score": 0.5, "ungrounded_terms": [], "classification": "partially_grounded"}
            self.error_events = [ErrorEvent(category=ErrorCategory.HALLUCINATION, severity="high", reason="t").to_dict()]

        def to_dict(self):
            return {
                "overall": 0.5, "correctness": {"overall": 0.5}, "consistency": {"score": 0.5},
                "hallucination": self.hallucination, "example_id": "ex", "output": "x",
                "expected": {}, "error_events": self.error_events,
            }

    loop = OptimizationLoop(evaluate_fn=lambda t, e: _R())
    res = loop.run(seeds=[PromptTemplate("seed", "Just answer.")], dataset=[{"id": "ex"}], max_iterations=2)
    if not res.iterations:
        raise AssertionError("optimisation loop produced no iterations")


def check_unit_tests() -> None:
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(verbosity=0, stream=open(__import__("os").devnull, "w"))
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError(
            f"{len(result.failures)} failed, {len(result.errors)} errored"
        )


# ---------- main ----------


CHECKS = [
    Check("python >= 3.10", check_python_version),
    Check("required imports (numpy/pandas/yaml/jsonschema)", check_required_imports),
    Check("dashboard imports (streamlit/plotly)", check_dashboard_imports),
    Check("optional: anthropic", check_optional_anthropic, required=False),
    Check("optional: openai", check_optional_openai, required=False),
    Check("optional: sentence-transformers", check_optional_embeddings, required=False),
    Check("project layout", check_project_layout),
    Check("dataset loads + has required fields", check_dataset_loads),
    Check("schema validator", check_schema_loads),
    Check("metrics (exact / semantic / rules)", check_metrics),
    Check("end-to-end pipeline (mock LLM, ranks 3 prompts)", check_full_pipeline),
    Check("report writer (markdown + json)", check_report_writer),
    Check("run storage (SQLite round-trip)", check_run_storage),
    Check("error taxonomy + advanced hallucination detector", check_taxonomy_and_advanced_detector),
    Check("closed-loop optimisation", check_optimization_loop),
    Check("dashboard module imports", check_dashboard_module),
    Check("unit tests", check_unit_tests),
]


if __name__ == "__main__":
    print("LLM Evaluation Framework - Health Check\n")
    sys.exit(_run(CHECKS))
