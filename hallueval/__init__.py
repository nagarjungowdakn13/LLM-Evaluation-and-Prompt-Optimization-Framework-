"""
hallueval: An Empirical Framework and Study Instrument for Hallucination Reduction in LLMs.
"""

from src.study import StudyRunner, StudyResults, CostAnalyzer, TaxonomyExporter
from src.datasets import TruthfulQADataset, HaluEvalDataset, BenchmarkRegistry
from src.evaluator.runner import EvaluationRunner, EvaluationResult
from src.error_taxonomy import ErrorTaxonomy, ErrorCategory, ErrorEvent

__version__ = "0.1.0"

__all__ = [
    "StudyRunner",
    "StudyResults",
    "CostAnalyzer",
    "TaxonomyExporter",
    "TruthfulQADataset",
    "HaluEvalDataset",
    "BenchmarkRegistry",
    "EvaluationRunner",
    "EvaluationResult",
    "ErrorTaxonomy",
    "ErrorCategory",
    "ErrorEvent",
]
