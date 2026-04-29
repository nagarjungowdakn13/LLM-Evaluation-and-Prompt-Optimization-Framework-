"""Structured error categorisation for evaluation failures.

The taxonomy turns a heterogeneous bag of sub-scores (schema validity,
hallucination, rules, correctness) into a small set of named failure
modes that humans can act on. The same labels are used to:

- annotate per-example results,
- aggregate per-prompt error distributions,
- drive failure-driven prompt mutations in the optimisation loop.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable


class ErrorCategory(str, Enum):
    HALLUCINATION = "hallucination"
    SCHEMA_VIOLATION = "schema_violation"
    FORMATTING_ERROR = "formatting_error"
    MISSING_INFORMATION = "missing_information"
    INCORRECT_REASONING = "incorrect_reasoning"


@dataclass
class ErrorEvent:
    category: ErrorCategory
    severity: str          # "low" | "medium" | "high"
    reason: str            # short human-readable explanation
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d


# ---------- thresholds ----------

DEFAULTS = {
    "semantic_low":      0.55,
    "semantic_medium":   0.75,
    "hallucination_bad": 0.60,    # combined_score below = hallucination
    "missing_info_low":  0.50,    # semantic low + answerable mismatch
}


# ---------- classifier ----------


class ErrorTaxonomy:
    """Classifies a single evaluation result into zero or more error events.

    The classifier intentionally produces *multiple* labels per sample — a
    response can simultaneously be schema-violating and hallucinated. Down-
    stream aggregation reports this honestly rather than collapsing.
    """

    def __init__(self, thresholds: dict | None = None):
        self.t = {**DEFAULTS, **(thresholds or {})}

    def classify(
        self,
        *,
        correctness: dict,
        hallucination: dict,
        expected: Any,
        output: str,
    ) -> list[ErrorEvent]:
        events: list[ErrorEvent] = []

        schema = correctness.get("schema") or {}
        schema_valid = schema.get("valid")
        schema_errors = schema.get("errors") or []

        if schema_valid is False:
            severity = "high" if any("required" in e.lower() for e in schema_errors) else "medium"
            events.append(
                ErrorEvent(
                    category=ErrorCategory.SCHEMA_VIOLATION,
                    severity=severity,
                    reason="output failed JSON-schema validation",
                    evidence={"schema_errors": schema_errors[:5]},
                )
            )

        # Only flag a formatting error if a schema was required AND the output
        # could not be parsed. Reasoning-then-JSON outputs (chain-of-thought)
        # often start with prose but end with valid JSON — those are fine.
        if schema_valid is False and not _looks_like_json(output):
            events.append(
                ErrorEvent(
                    category=ErrorCategory.FORMATTING_ERROR,
                    severity="high",
                    reason="output is not parseable JSON despite a schema being required",
                    evidence={"first_chars": output[:60]},
                )
            )

        rb = correctness.get("rule_based") or {}
        failed_rules = rb.get("failed") or []
        format_rule_failed = {"looks_like_json", "has_required_keys"} & set(failed_rules)
        # Only count rule-based format failures when the schema validator
        # *also* failed. If the schema extracted and validated the JSON the
        # output is structurally fine even if rules expected stricter shape
        # (e.g. JSON on the very first character — false positive for CoT).
        if format_rule_failed and schema_valid is False and not any(
            e.category == ErrorCategory.FORMATTING_ERROR for e in events
        ):
            events.append(
                ErrorEvent(
                    category=ErrorCategory.FORMATTING_ERROR,
                    severity="medium",
                    reason="format-related rules failed",
                    evidence={"failed_rules": sorted(format_rule_failed)},
                )
            )

        h_combined = hallucination.get("combined_score", 1.0)
        ungrounded = hallucination.get("ungrounded_terms") or []
        if h_combined < self.t["hallucination_bad"]:
            severity = "high" if h_combined < 0.4 else "medium"
            events.append(
                ErrorEvent(
                    category=ErrorCategory.HALLUCINATION,
                    severity=severity,
                    reason=f"grounding score {h_combined:.2f} below threshold {self.t['hallucination_bad']:.2f}",
                    evidence={"ungrounded_terms": ungrounded[:8]},
                )
            )

        missing_entities = hallucination.get("missing_entities") or []
        if missing_entities:
            events.append(
                ErrorEvent(
                    category=ErrorCategory.MISSING_INFORMATION,
                    severity="medium" if len(missing_entities) <= 2 else "high",
                    reason="expected entities absent from the output",
                    evidence={"missing_entities": missing_entities},
                )
            )

        sem = correctness.get("semantic_similarity", 1.0)
        exact = correctness.get("exact_match", 1.0)
        if sem < self.t["semantic_low"] and not any(
            e.category == ErrorCategory.HALLUCINATION for e in events
        ):
            events.append(
                ErrorEvent(
                    category=ErrorCategory.INCORRECT_REASONING,
                    severity="high" if sem < 0.3 else "medium",
                    reason=f"semantic similarity to expected output is {sem:.2f}",
                    evidence={"semantic": sem, "exact_match": exact},
                )
            )

        if (
            isinstance(expected, dict)
            and expected.get("answerable") is False
            and _claims_an_answer(output)
        ):
            events.append(
                ErrorEvent(
                    category=ErrorCategory.HALLUCINATION,
                    severity="high",
                    reason="expected output is unanswerable but model produced an answer",
                    evidence={"first_chars": output[:120]},
                )
            )

        return events


# ---------- aggregation ----------


def aggregate_distribution(per_example_events: Iterable[Iterable[ErrorEvent]]) -> dict:
    """Sum error events into a per-category distribution.

    Returns a dict shape: {category: {count, by_severity: {low, medium, high}}}.
    """
    counts: dict[str, dict] = {c.value: {"count": 0, "by_severity": Counter()} for c in ErrorCategory}
    total = 0
    for events in per_example_events:
        for ev in events:
            cat = ev.category.value
            counts[cat]["count"] += 1
            counts[cat]["by_severity"][ev.severity] += 1
            total += 1
    for cat in counts:
        counts[cat]["by_severity"] = dict(counts[cat]["by_severity"])
    return {"total_events": total, "by_category": counts}


# ---------- helpers ----------


def _looks_like_json(text: str) -> bool:
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if s.startswith("{") and s.endswith("}"):
        return True
    if s.startswith("```") and "}" in s:
        return True
    return False


def _claims_an_answer(output: str) -> bool:
    if not isinstance(output, str):
        return False
    try:
        obj = json.loads(output)
        if isinstance(obj, dict):
            return bool(obj.get("answerable", True))
    except Exception:
        pass
    refusal_markers = ("not enough information", "cannot determine", "not stated", "does not state")
    return not any(m in output.lower() for m in refusal_markers)


# ---------- convenience: classify a whole evaluation result ----------


def classify_evaluation(result, taxonomy: ErrorTaxonomy | None = None) -> list[ErrorEvent]:
    """Classify an ``EvaluationResult`` (or its ``to_dict()`` shape)."""
    tax = taxonomy or ErrorTaxonomy()
    if hasattr(result, "to_dict"):
        d = result.to_dict()
    else:
        d = result
    return tax.classify(
        correctness=d.get("correctness", {}),
        hallucination=d.get("hallucination", {}),
        expected=d.get("expected"),
        output=d.get("output", ""),
    )
