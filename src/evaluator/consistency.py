from statistics import mean


class ConsistencyEvaluator:
    """Measures how stable the model's output is across N runs of the same prompt.

    Stability is the mean pairwise semantic similarity between runs. A score
    near 1.0 means the model is reliably reproducing the same answer; a low
    score signals nondeterminism that often correlates with hallucination.
    """

    def __init__(self, runner_fn, semantic_metric, n_runs: int = 3):
        self.runner_fn = runner_fn
        self.semantic = semantic_metric
        self.n_runs = max(2, int(n_runs))

    def evaluate(self, prompt: str) -> dict:
        outputs = [self.runner_fn(prompt) for _ in range(self.n_runs)]
        sims: list[float] = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sims.append(self.semantic.score(outputs[i], outputs[j]))
        score = mean(sims) if sims else 1.0
        return {
            "score": score,
            "n_runs": self.n_runs,
            "outputs": outputs,
            "pairwise_similarities": sims,
        }
