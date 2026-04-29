from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, predicted: str, expected: str, **context: Any) -> float | dict:
        ...
