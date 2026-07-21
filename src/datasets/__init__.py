"""Dataset Registry and Loaders for Benchmarks."""

from .truthfulqa import TruthfulQADataset
from .halueval import HaluEvalDataset

class BenchmarkRegistry:
    """Registry managing benchmark datasets for the Hallucination Reduction Study."""

    _DATASETS = {
        "truthfulqa": TruthfulQADataset,
        "halueval": HaluEvalDataset,
    }

    @classmethod
    def get_dataset(cls, name: str):
        key = name.lower().replace("_", "").replace("-", "")
        for k, v in cls._DATASETS.items():
            if k in key or key in k:
                return v()
        raise ValueError(f"Unknown benchmark dataset '{name}'. Available: {list(cls._DATASETS.keys())}")

    @classmethod
    def list_datasets(cls) -> list[str]:
        return list(cls._DATASETS.keys())

__all__ = ["TruthfulQADataset", "HaluEvalDataset", "BenchmarkRegistry"]
