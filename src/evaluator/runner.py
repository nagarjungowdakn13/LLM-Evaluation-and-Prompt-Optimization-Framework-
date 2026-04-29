from dataclasses import dataclass, field, asdict
from typing import Callable
from typing import Any

from .consistency import ConsistencyEvaluator
from .correctness import CorrectnessEvaluator
from .hallucination import HallucinationDetector

# Soft imports — runner stays usable even if the optional modules are missing.
try:
    from src.error_taxonomy import ErrorTaxonomy, classify_evaluation
except ImportError:  # pragma: no cover
    ErrorTaxonomy = None  # type: ignore
    classify_evaluation = None  # type: ignore


@dataclass
class EvaluationResult:
    template_id: str
    example_id: str
    prompt: str
    output: str
    expected: Any
    correctness: dict
    consistency: dict
    hallucination: dict
    overall: float
    error_events: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Trim verbose consistency.outputs — caller can rerun for full text
        if "outputs" in d.get("consistency", {}):
            d["consistency"]["outputs"] = [
                o[:240] + "…" if len(o) > 240 else o for o in d["consistency"]["outputs"]
            ]
        return d


class EvaluationRunner:
    """Orchestrates a single (template, example) evaluation pass."""

    def __init__(
        self,
        llm_client,
        correctness: CorrectnessEvaluator,
        hallucination,                 # HallucinationDetector or AdvancedHallucinationDetector
        consistency_runs: int = 3,
        weights: dict | None = None,
        taxonomy=None,                 # ErrorTaxonomy | None
    ):
        self.llm = llm_client
        self.correctness = correctness
        self.hallucination = hallucination
        self.consistency_runs = consistency_runs
        self.weights = weights or {
            "correctness": 0.5,
            "consistency": 0.2,
            "hallucination": 0.3,
        }
        self.taxonomy = taxonomy or (ErrorTaxonomy() if ErrorTaxonomy else None)

    def _complete(self, prompt: str) -> str:
        return self.llm.complete(prompt).text

    def _complete_streaming(self, prompt: str, stream_callback: Callable[[str], None] | None = None) -> str:
        if stream_callback is None or not hasattr(self.llm, "stream_complete"):
            return self._complete(prompt)

        chunks: list[str] = []
        for chunk in self.llm.stream_complete(prompt):
            if not chunk:
                continue
            chunks.append(chunk)
            stream_callback(chunk)
        return "".join(chunks)

    def evaluate(self, template, example: dict, stream_callback: Callable[[str], None] | None = None) -> EvaluationResult:
        prompt = template.render(example)
        output = self._complete_streaming(prompt, stream_callback=stream_callback)
        if not output:
            output = self._complete(prompt)

        correctness = self.correctness.evaluate(output, example["expected_output"])

        consistency_eval = ConsistencyEvaluator(
            runner_fn=self._complete,
            semantic_metric=self.correctness.semantic,
            n_runs=self.consistency_runs,
        )
        consistency = consistency_eval.evaluate(prompt)

        hallucination = self.hallucination.detect(
            predicted=output,
            expected=example["expected_output"],
            context=example.get("context"),
        )
        hallucination["combined_score"] = HallucinationDetector.combine_with_consistency(
            hallucination["score"], consistency["score"]
        )

        overall = (
            self.weights["correctness"] * correctness["overall"]
            + self.weights["consistency"] * consistency["score"]
            + self.weights["hallucination"] * hallucination["combined_score"]
        ) / sum(self.weights.values())

        result = EvaluationResult(
            template_id=template.name,
            example_id=example["id"],
            prompt=prompt,
            output=output,
            expected=example["expected_output"],
            correctness=correctness,
            consistency=consistency,
            hallucination=hallucination,
            overall=overall,
            metadata={"model": getattr(self.llm, "model", "unknown")},
        )

        if self.taxonomy is not None and classify_evaluation is not None:
            events = classify_evaluation(result, taxonomy=self.taxonomy)
            result.error_events = [e.to_dict() for e in events]

        return result
