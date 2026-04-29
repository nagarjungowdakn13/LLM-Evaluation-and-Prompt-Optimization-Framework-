"""Prompt mutation strategies driven by the dominant error category.

Each mutator inspects the failure distribution for a parent prompt and
appends a targeted instruction. The mutators are deliberately small and
composable so the optimisation loop can stack them across iterations.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from src.error_taxonomy import ErrorCategory


class PromptProposer(Protocol):
    def propose(self, *, parent_body: str, failure_summary: dict, version: str) -> str:
        ...


# ---------- per-category mutators ----------

_GUARDS = {
    ErrorCategory.HALLUCINATION.value: (
        "\n\nIMPORTANT: Only use facts present in the provided context. "
        "If a detail is not stated, write 'not stated in context' rather than guessing."
    ),
    ErrorCategory.SCHEMA_VIOLATION.value: (
        "\n\nFormat: respond with a single valid JSON object on one line, "
        "with NO markdown, NO code fences, and NO commentary outside the JSON."
    ),
    ErrorCategory.FORMATTING_ERROR.value: (
        "\n\nFormat: your entire reply must be a JSON object that begins with '{' "
        "and ends with '}'. Do not add prose before or after."
    ),
    ErrorCategory.MISSING_INFORMATION.value: (
        "\n\nBefore producing the JSON, list every entity, name, number and date "
        "from the context that is relevant to the question, then ensure each "
        "appears verbatim in the answer or the sources field."
    ),
    ErrorCategory.INCORRECT_REASONING.value: (
        "\n\nReason step by step on a scratch line beginning with 'reasoning:' "
        "before writing the final JSON answer on the last line."
    ),
}


def _ranked_categories(failure_summary: dict) -> list[str]:
    counts = Counter()
    for cat, info in (failure_summary.get("by_category") or {}).items():
        counts[cat] = info.get("count", 0)
    return [cat for cat, n in counts.most_common() if n > 0]


def mutate_for_failures(parent_body: str, failure_summary: dict) -> tuple[str, str]:
    """Return ``(new_body, applied_mutation_label)``.

    Walks through the failure categories in descending frequency order
    and applies the first guard not already present in the parent body.
    This lets the optimisation loop make progress across multiple
    iterations even after the top issue has been addressed.
    """
    for cat in _ranked_categories(failure_summary):
        guard = _GUARDS.get(cat)
        if not guard:
            continue
        if guard.strip() in parent_body:
            continue
        return parent_body.rstrip() + guard, cat
    return parent_body, ""


# ---------- proposer ----------


class HeuristicProposer:
    """Default in-process prompt proposer.

    Simply applies :func:`mutate_for_failures`. Implements
    :class:`PromptProposer` so callers can swap it for an LLM-backed
    proposer without touching the loop.
    """

    def propose(self, *, parent_body: str, failure_summary: dict, version: str) -> str:
        new_body, _ = mutate_for_failures(parent_body, failure_summary)
        return new_body
