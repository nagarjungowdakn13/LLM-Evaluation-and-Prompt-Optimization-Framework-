"""Closed-loop prompt optimisation.

Algorithm
---------

For each iteration ``k = 1..max_iterations``:

1. Evaluate every active template against the dataset.
2. Classify per-example failures via the error taxonomy.
3. Pick the **best-performing** template that still has failures and
   apply a targeted mutation (:mod:`mutators`) to produce a child.
4. The child enters the active pool with version ``v{k+1}``.
5. Stop when the global best score plateaus or no candidate has failures.

Each iteration is recorded so the user can plot improvement and inspect
which mutation was applied at which step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

from src.error_taxonomy import (
    ErrorTaxonomy,
    aggregate_distribution,
    classify_evaluation,
)
from src.prompts import PromptManager, PromptOptimizer, PromptTemplate
from src.prompts.strategies import StrategyTemplate, annotate

from .mutators import HeuristicProposer, PromptProposer, mutate_for_failures


@dataclass
class IterationRecord:
    iteration: int
    ranking: list[tuple[str, float]]
    best_template: str
    best_score: float
    failure_distribution: dict
    proposed_child: str | None = None
    parent_template: str | None = None
    mutation_label: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationResult:
    iterations: list[IterationRecord] = field(default_factory=list)
    best_template_name: str = ""
    best_template_body: str = ""
    best_score: float = 0.0
    score_history: list[float] = field(default_factory=list)
    final_comparison: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "best_template_name": self.best_template_name,
            "best_template_body": self.best_template_body,
            "best_score": self.best_score,
            "score_history": self.score_history,
            "iterations": [r.to_dict() for r in self.iterations],
        }


class OptimizationLoop:
    """Iterative evaluator → analyser → proposer → re-evaluator loop."""

    def __init__(
        self,
        evaluate_fn: Callable,
        proposer: PromptProposer | None = None,
        taxonomy: ErrorTaxonomy | None = None,
        plateau_eps: float = 0.005,
    ):
        self.evaluate_fn = evaluate_fn
        self.proposer = proposer or HeuristicProposer()
        self.taxonomy = taxonomy or ErrorTaxonomy()
        self.plateau_eps = plateau_eps

    def run(
        self,
        seeds: Iterable[PromptTemplate | StrategyTemplate],
        dataset: list[dict],
        max_iterations: int = 3,
    ) -> OptimizationResult:
        active: list[StrategyTemplate] = [
            s if isinstance(s, StrategyTemplate) else annotate(s) for s in seeds
        ]
        result = OptimizationResult()
        prev_best = -1.0

        for k in range(1, max_iterations + 1):
            optimizer = PromptOptimizer(evaluate_fn=self.evaluate_fn)
            comparison = optimizer.compare(active, dataset)
            ranking = optimizer.rank(comparison)

            failure_dist_by_template = {}
            for tmpl in active:
                events_per_example = []
                for ex in comparison[tmpl.name]["examples"]:
                    events = classify_evaluation(ex, taxonomy=self.taxonomy)
                    events_per_example.append(events)
                    ex["error_events"] = [e.to_dict() for e in events]
                failure_dist_by_template[tmpl.name] = aggregate_distribution(
                    events_per_example
                )
                comparison[tmpl.name]["failure_distribution"] = failure_dist_by_template[
                    tmpl.name
                ]

            best_name, best_score = ranking[0]
            result.score_history.append(best_score)
            result.final_comparison = comparison

            # Pick the best template that still has at least one failure to mutate.
            parent: StrategyTemplate | None = None
            for name, _score in ranking:
                if failure_dist_by_template[name]["total_events"] > 0:
                    parent = next(t for t in active if t.name == name)
                    break

            iteration_record = IterationRecord(
                iteration=k,
                ranking=ranking,
                best_template=best_name,
                best_score=best_score,
                failure_distribution=failure_dist_by_template,
            )

            # Plateau check (after the first iteration).
            if k > 1 and abs(best_score - prev_best) < self.plateau_eps and parent is None:
                result.iterations.append(iteration_record)
                break
            prev_best = best_score

            # Final iteration → no need to propose.
            if k == max_iterations or parent is None:
                result.iterations.append(iteration_record)
                break

            child_body = self.proposer.propose(
                parent_body=parent.body,
                failure_summary=failure_dist_by_template[parent.name],
                version=f"v{k+1}",
            )
            new_body, mutation_label = mutate_for_failures(
                parent.body,
                failure_dist_by_template[parent.name],
            )
            # If proposer returned the parent verbatim, fall back to heuristic mutator.
            if child_body == parent.body:
                child_body = new_body
            if child_body == parent.body:
                # Nothing more to try.
                result.iterations.append(iteration_record)
                break

            child_name = f"{parent.name}__opt_v{k+1}"
            child_template = PromptTemplate(name=child_name, body=child_body)
            child_annotated = annotate(child_template, version=f"v{k+1}", parent=parent.name)
            active.append(child_annotated)

            iteration_record.proposed_child = child_name
            iteration_record.parent_template = parent.name
            iteration_record.mutation_label = mutation_label
            result.iterations.append(iteration_record)

        # Final best across all iterations.
        if result.score_history:
            best_idx = max(range(len(result.score_history)), key=result.score_history.__getitem__)
            best_record = result.iterations[best_idx]
            result.best_template_name = best_record.best_template
            result.best_score = best_record.best_score
            best_tmpl = next(
                (t for t in active if t.name == best_record.best_template),
                None,
            )
            result.best_template_body = best_tmpl.body if best_tmpl else ""

        return result
