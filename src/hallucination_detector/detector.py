"""Advanced hallucination detection.

Three independent signals are combined:

1. **Grounding ratio** — content tokens (and numbers) in the output that
   are absent from the expected answer + the supplied context.
2. **Missing key entities** — multi-character tokens / named-entity-shaped
   substrings that appear in the expected answer but not in the output.
3. **Unsupported claims** — numbers and capitalised proper nouns in the
   output that do not appear anywhere in the context.

These map onto a ternary classification:

- ``grounded``            – every signal is clean (≥ ``high_threshold``).
- ``partially_grounded``  – some drift but still mostly safe.
- ``hallucinated``        – at least one signal failed badly.

The detector is **drop-in compatible** with the previous detector: the
returned dict still contains ``score``, ``ungrounded_terms``,
``ungrounded_ratio``, and ``is_hallucinated``. New keys are additive:
``classification``, ``missing_entities``, ``unsupported_claims``,
``signals``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


_TOKEN = re.compile(r"\w+")
_NUMBER = re.compile(r"\b\d[\d,\.]*\b")
# Capitalised words / multi-word proper-noun-like spans
_PROPER = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b")

_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by", "as",
    "it", "this", "that", "these", "those", "from", "into", "than", "then",
    "so", "such", "if", "while", "i", "you", "we", "they", "he", "she",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "can", "about", "not",
}


class GroundingClass(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    HALLUCINATED = "hallucinated"


@dataclass
class GroundingResult:
    score: float
    classification: GroundingClass
    ungrounded_terms: list[str]
    ungrounded_ratio: float
    missing_entities: list[str]
    unsupported_claims: list[str]
    signals: dict
    is_hallucinated: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["classification"] = self.classification.value
        return d


# ---------- helpers ----------


def _to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _content_terms(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _TOKEN.finditer(text)
        if m.group(0).lower() not in _STOP and len(m.group(0)) > 2
    }


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _proper_nouns(text: str) -> set[str]:
    # Strip JSON syntax artefacts before scanning
    cleaned = re.sub(r"[\"\\{}\[\]:,]", " ", text)
    return {m.group(1).strip() for m in _PROPER.finditer(cleaned)}


# ---------- detector ----------


class AdvancedHallucinationDetector:
    """Drop-in upgrade for the original ``HallucinationDetector``.

    Parameters
    ----------
    high_threshold:
        score at or above which a sample is classified ``grounded``.
    low_threshold:
        score below which a sample is classified ``hallucinated``. Between
        the two thresholds the sample is ``partially_grounded``.
    weights:
        weights for combining the three signals into the final score.
    """

    def __init__(
        self,
        min_grounded_ratio: float = 0.7,
        high_threshold: float = 0.85,
        low_threshold: float = 0.55,
        weights: dict | None = None,
    ):
        self.min_grounded_ratio = min_grounded_ratio
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.weights = weights or {
            "grounding": 0.5,
            "entity_recall": 0.3,
            "claim_support": 0.2,
        }

    # Keep the legacy method name so existing call sites (runner.py) work.
    def detect(
        self,
        predicted: Any,
        expected: Any,
        context: str | None = None,
    ) -> dict:
        pred_text = _to_text(predicted)
        ref_text = _to_text(expected) + " " + (context or "")
        ctx_text = context or ""

        # --- signal 1: grounding ratio ---
        pred_terms = _content_terms(pred_text) | _numbers(pred_text)
        ref_terms = _content_terms(ref_text) | _numbers(ref_text)
        if pred_terms:
            ungrounded = sorted(pred_terms - ref_terms)
            grounding_ratio = 1.0 - (len(ungrounded) / len(pred_terms))
        else:
            ungrounded = []
            grounding_ratio = 1.0

        # --- signal 2: missing key entities ---
        expected_entities = _proper_nouns(_to_text(expected)) | _numbers(_to_text(expected))
        pred_lower = pred_text.lower()
        missing = sorted(e for e in expected_entities if e.lower() not in pred_lower)
        if expected_entities:
            entity_recall = 1.0 - (len(missing) / len(expected_entities))
        else:
            entity_recall = 1.0

        # --- signal 3: unsupported claims (proper nouns + numbers in output not in context) ---
        pred_propers = _proper_nouns(pred_text)
        pred_numbers = _numbers(pred_text)
        ctx_lower = ctx_text.lower()
        unsupported_propers = sorted(p for p in pred_propers if p.lower() not in ctx_lower)
        unsupported_numbers = sorted(n for n in pred_numbers if n not in ctx_text)
        unsupported = unsupported_propers + unsupported_numbers
        # claim_support reflects what fraction of "claims" are anchored in context
        total_claims = len(pred_propers) + len(pred_numbers)
        if total_claims:
            claim_support = 1.0 - (len(unsupported) / total_claims)
        else:
            claim_support = 1.0

        # --- combined score ---
        w = self.weights
        score = (
            w["grounding"] * grounding_ratio
            + w["entity_recall"] * entity_recall
            + w["claim_support"] * claim_support
        ) / sum(w.values())

        # --- ternary classification ---
        if score >= self.high_threshold and not missing and not unsupported:
            classification = GroundingClass.GROUNDED
        elif score < self.low_threshold or len(unsupported) >= 3:
            classification = GroundingClass.HALLUCINATED
        else:
            classification = GroundingClass.PARTIALLY_GROUNDED

        result = GroundingResult(
            score=score,
            classification=classification,
            ungrounded_terms=ungrounded,
            ungrounded_ratio=1.0 - grounding_ratio,
            missing_entities=missing,
            unsupported_claims=unsupported,
            signals={
                "grounding": round(grounding_ratio, 4),
                "entity_recall": round(entity_recall, 4),
                "claim_support": round(claim_support, 4),
            },
            is_hallucinated=score < self.min_grounded_ratio
            or classification == GroundingClass.HALLUCINATED,
        )
        return result.to_dict()

    @staticmethod
    def combine_with_consistency(grounding_score: float, consistency_score: float) -> float:
        return grounding_score * consistency_score
