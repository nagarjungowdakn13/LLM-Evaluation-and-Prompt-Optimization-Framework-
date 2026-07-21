"""CLI entry point for the LLM Evaluation and Prompt Optimization Framework & Study."""

from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path

import yaml

from src.evaluator import (
    CorrectnessEvaluator,
    EvaluationRunner,
)
from src.error_taxonomy import ErrorTaxonomy, aggregate_distribution
from src.experiment_tracker import ExperimentTracker
from src.hallucination_detector import AdvancedHallucinationDetector
from src.llm import get_client
from src.metrics import (
    ExactMatchMetric,
    RuleBasedMetric,
    SemanticSimilarityMetric,
)
from src.metrics.rule_based import default_qa_rules
from src.optimization import OptimizationLoop
from src.prompts import (
    PromptManager,
    PromptOptimizer,
    annotate_all,
    rank_strategies,
)
from src.reporting import Reporter
from src.schema import SchemaValidator, load_schema
from src.utils.logger import get_logger
from src.study import StudyRunner


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_dataset(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["examples"]


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    runtime = copy.deepcopy(config)
    if getattr(args, "dataset", None):
        runtime.setdefault("dataset", {})["path"] = args.dataset
    if getattr(args, "prompts_dir", None):
        runtime.setdefault("prompts", {})["directory"] = args.prompts_dir
    if getattr(args, "schema_path", None):
        runtime.setdefault("prompts", {})["schema_path"] = args.schema_path
    if getattr(args, "report_dir", None):
        runtime.setdefault("reporting", {})["output_dir"] = args.report_dir
    if getattr(args, "storage_db", None):
        runtime.setdefault("storage", {})["sqlite_path"] = args.storage_db
    return runtime


def build_runner(config: dict, dataset: list[dict]):
    log = get_logger()

    llm = get_client(config.get("llm", {}))
    if hasattr(llm, "register_dataset"):
        llm.register_dataset(dataset)
        log.info("Registered %d examples with mock LLM client", len(dataset))

    metrics_cfg = config.get("metrics", {})
    exact = ExactMatchMetric(
        case_sensitive=metrics_cfg.get("exact_match", {}).get("case_sensitive", False),
        strip_whitespace=metrics_cfg.get("exact_match", {}).get("strip_whitespace", True),
    )
    semantic = SemanticSimilarityMetric(
        method=metrics_cfg.get("semantic_similarity", {}).get("method", "tfidf"),
        embedding_model=metrics_cfg.get("semantic_similarity", {}).get(
            "embedding_model", "all-MiniLM-L6-v2"
        ),
    )
    rule_based = (
        RuleBasedMetric(rules=default_qa_rules())
        if metrics_cfg.get("rule_based", {}).get("enabled", True)
        else None
    )

    schema_path = config.get("prompts", {}).get("schema_path")
    schema_validator = SchemaValidator(load_schema(schema_path)) if schema_path else None

    correctness = CorrectnessEvaluator(
        exact_match=exact,
        semantic=semantic,
        rule_based=rule_based,
        schema_validator=schema_validator,
    )

    halluc_cfg = config.get("evaluation", {}).get("hallucination", {})
    hallucination = AdvancedHallucinationDetector(
        min_grounded_ratio=halluc_cfg.get("min_grounded_ratio", 0.7),
        high_threshold=halluc_cfg.get("high_threshold", 0.85),
        low_threshold=halluc_cfg.get("low_threshold", 0.55),
    )

    taxonomy = ErrorTaxonomy(thresholds=config.get("evaluation", {}).get("error_thresholds"))

    runner = EvaluationRunner(
        llm_client=llm,
        correctness=correctness,
        hallucination=hallucination,
        consistency_runs=config.get("evaluation", {}).get("consistency_runs", 3),
        weights=config.get("evaluation", {}).get("weights"),
        taxonomy=taxonomy,
    )
    return runner


def cmd_run(args: argparse.Namespace) -> int:
    log = get_logger()
    config = apply_overrides(load_config(args.config), args)
    dataset = load_dataset(config["dataset"]["path"])

    pm = PromptManager().load_directory(config["prompts"]["directory"])
    template_names = args.templates or config["prompts"]["templates"]
    templates = pm.list(template_names)
    if not templates:
        log.error("No templates resolved. Available: %s", list(pm.templates))
        return 2

    runner = build_runner(config, dataset)
    optimizer = PromptOptimizer(evaluate_fn=runner.evaluate)

    log.info(
        "Evaluating %d templates against %d examples (consistency runs=%d)",
        len(templates), len(dataset), runner.consistency_runs,
    )
    comparison = optimizer.compare(templates, dataset)
    ranking = optimizer.rank(comparison)

    print("\n=== Prompt Ranking ===")
    for rank, (name, score) in enumerate(ranking, 1):
        print(f"  {rank}. {name:<25} avg={score:.3f}")

    annotated = annotate_all(templates)
    strategy_rank = rank_strategies(comparison, annotated)
    if strategy_rank:
        print("\n=== Strategy Ranking ===")
        for rank, (strategy, score, n) in enumerate(strategy_rank, 1):
            print(f"  {rank}. {strategy:<20} avg={score:.3f} ({n} templates)")

    print("\n=== Error Distribution (per template) ===")
    for name, _ in ranking:
        counts = comparison[name].get("error_counts", {})
        if not counts:
            print(f"  {name:<25} no errors")
            continue
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {name:<25} {breakdown}")

    reporter = Reporter(output_dir=config["reporting"]["output_dir"])
    formats = (config["reporting"]["format"], "json") if config["reporting"]["format"] != "json" else ("json",)
    paths = reporter.write(comparison, ranking, formats=set(formats))
    for fmt, path in paths.items():
        log.info("Wrote %s report -> %s", fmt, path)

    storage_cfg = config.get("storage", {})
    if storage_cfg.get("enabled", True):
        tracker = ExperimentTracker(sqlite_path=storage_cfg.get("sqlite_path", "reports/runs.sqlite3"))
        prompt_versions = {st.name: st.version for st in annotated}
        run_id = tracker.record_run(
            comparison=comparison,
            ranking=ranking,
            config=config,
            dataset=dataset,
            prompt_versions=prompt_versions,
            report_paths={fmt: str(path) for fmt, path in paths.items()},
            extra={
                "strategy_ranking": strategy_rank,
                "error_distribution": {n: c.get("error_counts", {}) for n, c in comparison.items()},
            },
        )
        log.info("Saved run metadata -> %s", run_id)
    return 0


def cmd_run_study(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider = getattr(args, "provider", None) or config.get("llm", {}).get("provider", "mock")
    models = [m.strip() for m in args.models.split(",") if m.strip()] if getattr(args, "models", None) else None
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] if getattr(args, "datasets", None) else None

    runner = StudyRunner(config=config)
    runner.run_study(provider=provider, models=models, datasets=datasets)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    log = get_logger()
    config = apply_overrides(load_config(args.config), args)
    dataset = load_dataset(config["dataset"]["path"])

    pm = PromptManager().load_directory(config["prompts"]["directory"])
    template_names = args.templates or config["prompts"]["templates"]
    templates = pm.list(template_names)
    if not templates:
        log.error("No templates resolved. Available: %s", list(pm.templates))
        return 2

    runner = build_runner(config, dataset)
    loop = OptimizationLoop(evaluate_fn=runner.evaluate)
    iterations = max(1, int(args.iterations))
    log.info(
        "Starting optimisation loop: %d iterations across %d seed prompts",
        iterations, len(templates),
    )
    result = loop.run(seeds=templates, dataset=dataset, max_iterations=iterations)

    print("\n=== Optimisation history ===")
    for record in result.iterations:
        line = f"  iter {record.iteration}: best={record.best_template} score={record.best_score:.3f}"
        if record.proposed_child:
            line += f"  -> proposed {record.proposed_child} (mutation={record.mutation_label})"
        print(line)

    if result.score_history:
        first, last = result.score_history[0], result.score_history[-1]
        print(f"\nLift over loop: {first:.3f} -> {last:.3f}  (+{(last-first):.3f})")
        print(f"Best template: {result.best_template_name} ({result.best_score:.3f})")

    if args.write_best and result.best_template_name and result.best_template_body:
        out_path = Path(config["prompts"]["directory"]) / f"{result.best_template_name}.txt"
        out_path.write_text(result.best_template_body, encoding="utf-8")
        log.info("Wrote optimised prompt -> %s", out_path)

    storage_cfg = config.get("storage", {})
    if storage_cfg.get("enabled", True) and result.iterations:
        tracker = ExperimentTracker(sqlite_path=storage_cfg.get("sqlite_path", "reports/runs.sqlite3"))
        last_record = result.iterations[-1]
        run_id = tracker.record_run(
            comparison=result.final_comparison,
            ranking=last_record.ranking,
            config=config,
            dataset=dataset,
            prompt_versions={st.name: st.version for st in annotate_all(templates)},
            extra={
                "optimization": result.to_dict(),
            },
        )
        log.info("Saved optimisation run metadata -> %s", run_id)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    config = apply_overrides(load_config(args.config), args)
    pm = PromptManager().load_directory(config["prompts"]["directory"])
    print("Templates:")
    for name in pm.templates:
        print(f"  - {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Run prompt-template evaluations and produce a ranked report.",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", help="Override the dataset path from config")
    parser.add_argument("--prompts-dir", help="Override the prompt directory from config")
    parser.add_argument("--schema-path", help="Override the schema path from config")
    parser.add_argument("--report-dir", help="Override the report output directory")
    parser.add_argument("--storage-db", help="Override the SQLite database used for run history")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run the full batch evaluation")
    p_run.add_argument("--templates", nargs="*", help="Override templates from config")
    p_run.set_defaults(func=cmd_run)

    p_study = sub.add_parser("run-study", help="Run the Hallucination Reduction Study grid")
    p_study.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"])
    p_study.add_argument("--models", help="Comma-separated models to evaluate")
    p_study.add_argument("--datasets", help="Comma-separated datasets to evaluate")
    p_study.set_defaults(func=cmd_run_study)

    p_opt = sub.add_parser(
        "optimize",
        help="Run the closed-loop optimiser: evaluate, analyse failures, mutate, re-evaluate",
    )
    p_opt.add_argument("--templates", nargs="*", help="Seed templates (override config)")
    p_opt.add_argument("--iterations", type=int, default=3, help="Maximum optimisation iterations")
    p_opt.add_argument(
        "--write-best",
        action="store_true",
        help="If set, persist the best optimised prompt under data/prompts/",
    )
    p_opt.set_defaults(func=cmd_optimize)

    p_list = sub.add_parser("list-templates", help="List available prompt templates")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        args = parser.parse_args(["run"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
