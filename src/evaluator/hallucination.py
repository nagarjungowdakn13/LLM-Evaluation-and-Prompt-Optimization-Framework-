"""Legacy hallucination detector (kept for backward compatibility).

For new code prefer ``src.hallucination_detector.AdvancedHallucinationDetector``,
which produces the same fields as this class plus ``classification``,
``missing_entities``, ``unsupported_claims`` and ``signals``.
"""

import json
import re
from typing import Any

_TOKEN = re.compile(r"\w+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by", "as",
    "it", "this", "that", "these", "those", "from", "into", "than", "then",
    "so", "such", "if", "while", "i", "you", "we", "they", "he", "she",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "can", "about",
}


def _content_terms(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _TOKEN.finditer(text)
        if m.group(0).lower() not in _STOP and len(m.group(0)) > 2
    }


def _number_terms(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d,\.]*\b", text))


def _to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class HallucinationDetector:
    """Grounding-based hallucination signal.

    Compares content tokens (and numbers) in the model output against a
    reference set built from the expected answer plus any provided
    grounding context. Output terms that appear in neither are flagged
    as ungrounded. The score is ``1 - ungrounded_ratio``.

    A separate consistency-based signal can be folded in by callers via
    ``combine_with_consistency``.
    """

    def __init__(self, min_grounded_ratio: float = 0.7):
        self.min_grounded_ratio = min_grounded_ratio

    def detect(
        self,
        predicted: Any,
        expected: Any,
        context: str | None = None,
    ) -> dict:
        pred_text = _to_text(predicted)
        ref_text = _to_text(expected) + " " + (context or "")

        pred_terms = _content_terms(pred_text) | _number_terms(pred_text)
        ref_terms = _content_terms(ref_text) | _number_terms(ref_text)

        if not pred_terms:
            return {
                "score": 1.0,
                "ungrounded_terms": [],
                "ungrounded_ratio": 0.0,
                "is_hallucinated": False,
            }

        ungrounded = sorted(pred_terms - ref_terms)
        ratio = len(ungrounded) / len(pred_terms)
        score = 1.0 - ratio
        return {
            "score": score,
            "ungrounded_terms": ungrounded,
            "ungrounded_ratio": ratio,
            "is_hallucinated": score < self.min_grounded_ratio,
        }

    @staticmethod
    def combine_with_consistency(grounding_score: float, consistency_score: float) -> float:
        # Low consistency is itself evidence of hallucination — multiply.
        return grounding_score * consistency_score
