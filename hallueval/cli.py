"""CLI Entrypoint for hallueval package."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from src.study import StudyRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hallueval",
        description="hallueval: LLM Hallucination Reduction Study & Evaluation Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run-study subcommand
    study_parser = subparsers.add_parser("run-study", help="Run the full multi-technique hallucination study grid")
    study_parser.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"], help="LLM Provider")
    study_parser.add_argument("--models", default="llama-3.1-8b,mistral-7b,qwen-2.5-7b,gpt-4o", help="Comma-separated models")
    study_parser.add_argument("--datasets", default="truthfulqa,halueval", help="Comma-separated benchmark datasets")

    # list-datasets subcommand
    subparsers.add_parser("list-datasets", help="List registered evaluation benchmark datasets")

    # version subcommand
    subparsers.add_parser("version", help="Print hallueval version")

    args = parser.parse_args()

    if args.command == "run-study":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        runner = StudyRunner()
        runner.run_study(provider=args.provider, models=models, datasets=datasets)

    elif args.command == "list-datasets":
        from src.datasets import BenchmarkRegistry
        print("Registered Datasets:")
        for name in BenchmarkRegistry.list_datasets():
            ds = BenchmarkRegistry.get_dataset(name)
            print(f"  - {ds.name}: {ds.description}")

    elif args.command == "version":
        from hallueval import __version__
        print(f"hallueval version {__version__}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
