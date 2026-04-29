"""Prompt strategy framework.

Three strategies are supported out of the box:

- ``instruction``    — direct imperative prompt; no examples.
- ``few_shot``       — prefixes one or more solved examples before the task.
- ``chain_of_thought`` — instructs the model to reason step-by-step then
   produce the final structured answer.

A strategy is *metadata on a template*. It does not change how the
template is rendered — but the experiment tracker can group results by
strategy, and the optimisation loop can apply strategy-specific
mutations (e.g. "this prompt is doing instruction; try a CoT variant").

A template's strategy is detected from a header line of the form::

    # strategy: chain_of_thought

If the header is absent the strategy is inferred from the body
(few-shot if it contains ``Example:`` blocks, CoT if it mentions
``step by step``/``reason step``, otherwise instruction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .manager import PromptTemplate


class PromptStrategy(str, Enum):
    INSTRUCTION = "instruction"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


_HEADER = re.compile(r"^\s*#\s*strategy\s*:\s*([\w_]+)\s*$", re.MULTILINE | re.IGNORECASE)


@dataclass
class StrategyTemplate:
    """A ``PromptTemplate`` annotated with strategy + version metadata."""

    template: PromptTemplate
    strategy: PromptStrategy
    version: str = "v1"
    parent: str | None = None  # populated by the optimiser when mutating

    @property
    def name(self) -> str:
        return self.template.name

    @property
    def body(self) -> str:
        return self.template.body

    def render(self, example: dict) -> str:
        return self.template.render(example)


def detect_strategy(body: str) -> PromptStrategy:
    """Infer strategy from a template body.

    Header takes precedence; otherwise look for textual cues. Defaults to
    ``instruction`` so unannotated templates still work.
    """
    m = _HEADER.search(body or "")
    if m:
        try:
            return PromptStrategy(m.group(1).lower())
        except ValueError:
            pass
    text = (body or "").lower()
    if "step by step" in text or "step-by-step" in text or "reason step" in text:
        return PromptStrategy.CHAIN_OF_THOUGHT
    if "example:" in text or "examples:" in text or "few-shot" in text:
        return PromptStrategy.FEW_SHOT
    return PromptStrategy.INSTRUCTION


def annotate(template: PromptTemplate, version: str = "v1", parent: str | None = None) -> StrategyTemplate:
    return StrategyTemplate(
        template=template,
        strategy=detect_strategy(template.body),
        version=version,
        parent=parent,
    )


def annotate_all(templates: list[PromptTemplate]) -> list[StrategyTemplate]:
    return [annotate(t) for t in templates]


def group_by_strategy(annotated: list[StrategyTemplate]) -> dict[str, list[StrategyTemplate]]:
    groups: dict[str, list[StrategyTemplate]] = {}
    for st in annotated:
        groups.setdefault(st.strategy.value, []).append(st)
    return groups


def rank_strategies(comparison: dict, annotated: list[StrategyTemplate]) -> list[tuple[str, float, int]]:
    """Aggregate ``optimizer.compare`` results by strategy.

    Returns a list of ``(strategy, average_score, template_count)`` tuples
    sorted by score descending.
    """
    by_strategy: dict[str, list[float]] = {}
    for st in annotated:
        if st.name in comparison:
            by_strategy.setdefault(st.strategy.value, []).append(comparison[st.name]["average_score"])
    rows = [
        (strategy, sum(scores) / len(scores), len(scores))
        for strategy, scores in by_strategy.items()
        if scores
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
