import json
import math
import re
from collections import Counter
from typing import Any

from .base import BaseMetric

_TOKEN = re.compile(r"\w+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by",
    "as", "it", "this", "that", "these", "those", "from", "into", "than",
    "then", "so", "such", "if", "while",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if t not in _STOP]


def _to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _cosine_bow(a: str, b: str) -> float:
    ta, tb = Counter(_tokenize(a)), Counter(_tokenize(b))
    if not ta or not tb:
        return 0.0
    keys = set(ta) | set(tb)
    dot = sum(ta[k] * tb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


class SemanticSimilarityMetric(BaseMetric):
    """Cosine similarity between predicted and expected text.

    Uses sentence-transformers when available, else falls back to a
    TF-style bag-of-words cosine. Both produce values in [0, 1].
    """

    name = "semantic_similarity"

    def __init__(self, method: str = "tfidf", embedding_model: str = "all-MiniLM-L6-v2"):
        self.method = method
        self._embedder = None
        if method == "embeddings":
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(embedding_model)
            except ImportError:
                self.method = "tfidf"

    def score(self, predicted: Any, expected: Any, **_: Any) -> float:
        a, b = _to_text(predicted), _to_text(expected)
        if self.method == "embeddings" and self._embedder is not None:
            import numpy as np

            embs = self._embedder.encode([a, b])
            x, y = embs[0], embs[1]
            denom = float(np.linalg.norm(x) * np.linalg.norm(y))
            return float(np.dot(x, y) / denom) if denom else 0.0
        return _cosine_bow(a, b)
