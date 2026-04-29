import re
from dataclasses import dataclass
from typing import Any, Callable

from .base import BaseMetric


@dataclass
class Rule:
    name: str
    check: Callable[[Any, Any, dict], bool]
    weight: float = 1.0


class RuleBasedMetric(BaseMetric):
    name = "rule_based"

    def __init__(self, rules: list[Rule] | None = None):
        self.rules: list[Rule] = list(rules or [])

    def add(self, rule: Rule) -> "RuleBasedMetric":
        self.rules.append(rule)
        return self

    def score(self, predicted: Any, expected: Any = None, **context: Any) -> dict:
        passed: list[str] = []
        failed: list[str] = []
        total_weight = 0.0
        passed_weight = 0.0
        for rule in self.rules:
            total_weight += rule.weight
            try:
                ok = bool(rule.check(predicted, expected, context))
            except Exception:
                ok = False
            if ok:
                passed.append(rule.name)
                passed_weight += rule.weight
            else:
                failed.append(rule.name)
        score = passed_weight / total_weight if total_weight else 1.0
        return {"score": score, "passed": passed, "failed": failed}


def default_qa_rules() -> list[Rule]:
    """Sensible default rules for the bundled QA dataset."""

    def looks_like_json(pred, _exp, _ctx):
        text = pred if isinstance(pred, str) else ""
        return text.strip().startswith("{") and text.strip().endswith("}")

    def has_required_keys(pred, _exp, _ctx):
        import json
        try:
            obj = json.loads(pred) if isinstance(pred, str) else pred
        except Exception:
            return False
        return all(k in obj for k in ("answer", "confidence", "answerable"))

    def confidence_is_valid(pred, _exp, _ctx):
        import json
        try:
            obj = json.loads(pred) if isinstance(pred, str) else pred
        except Exception:
            return False
        return obj.get("confidence") in {"low", "medium", "high"}

    def answer_not_empty(pred, _exp, _ctx):
        import json
        try:
            obj = json.loads(pred) if isinstance(pred, str) else pred
        except Exception:
            return False
        ans = obj.get("answer", "")
        return isinstance(ans, str) and len(ans.strip()) > 0

    def no_apology_phrases(pred, _exp, _ctx):
        text = pred if isinstance(pred, str) else str(pred)
        banned = ("as an ai language model", "i cannot", "i'm sorry, but")
        return not any(b in text.lower() for b in banned)

    return [
        Rule("looks_like_json", looks_like_json, 1.0),
        Rule("has_required_keys", has_required_keys, 2.0),
        Rule("confidence_is_valid", confidence_is_valid, 1.0),
        Rule("answer_not_empty", answer_not_empty, 2.0),
        Rule("no_apology_phrases", no_apology_phrases, 0.5),
    ]
