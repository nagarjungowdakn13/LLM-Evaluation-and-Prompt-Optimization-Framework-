import json
from typing import Any


class CorrectnessEvaluator:
    """Combines exact-match, semantic similarity, schema validity, and rules.

    Operates on either raw text or parsed dicts. When the schema validator
    is supplied, the parsed JSON object is used as the basis for exact and
    semantic comparison so noisy text wrappers don't penalise the model.
    """

    def __init__(
        self,
        exact_match,
        semantic,
        rule_based=None,
        schema_validator=None,
        weights: dict | None = None,
    ):
        self.exact = exact_match
        self.semantic = semantic
        self.rule_based = rule_based
        self.schema = schema_validator
        self.weights = weights or {
            "exact_match": 0.25,
            "semantic": 0.40,
            "schema": 0.20,
            "rule_based": 0.15,
        }

    def evaluate(self, predicted: str, expected: Any) -> dict:
        result: dict[str, Any] = {"raw_output": predicted}
        parsed = None
        schema_valid = None
        schema_errors: list[str] = []

        if self.schema is not None:
            schema_result = self.schema.validate(predicted)
            schema_valid = schema_result["valid"]
            schema_errors = schema_result["errors"]
            parsed = schema_result["parsed"]
            result["schema"] = {"valid": schema_valid, "errors": schema_errors}

        compare_pred = parsed if parsed is not None else predicted
        compare_exp = expected

        result["exact_match"] = self.exact.score(compare_pred, compare_exp)
        result["semantic_similarity"] = self.semantic.score(
            self._stringify(compare_pred), self._stringify(compare_exp)
        )

        if self.rule_based is not None:
            rb = self.rule_based.score(predicted)
            result["rule_based"] = rb

        result["overall"] = self._aggregate(result, schema_valid)
        return result

    def _aggregate(self, parts: dict, schema_valid: bool | None) -> float:
        w = self.weights
        total = 0.0
        weight_sum = 0.0
        total += w["exact_match"] * parts["exact_match"]
        weight_sum += w["exact_match"]
        total += w["semantic"] * parts["semantic_similarity"]
        weight_sum += w["semantic"]
        if schema_valid is not None:
            total += w["schema"] * (1.0 if schema_valid else 0.0)
            weight_sum += w["schema"]
        if "rule_based" in parts:
            total += w["rule_based"] * parts["rule_based"]["score"]
            weight_sum += w["rule_based"]
        return total / weight_sum if weight_sum else 0.0

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)
