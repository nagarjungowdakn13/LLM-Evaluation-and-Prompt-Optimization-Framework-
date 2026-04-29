import json
from typing import Any

from .base import BaseMetric


class ExactMatchMetric(BaseMetric):
    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def _normalize(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        text = str(value)
        if self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    def score(self, predicted: Any, expected: Any, **_: Any) -> float:
        return 1.0 if self._normalize(predicted) == self._normalize(expected) else 0.0
