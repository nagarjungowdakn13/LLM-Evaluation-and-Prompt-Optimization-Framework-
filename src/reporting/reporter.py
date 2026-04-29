import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


class Reporter:
    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, comparison: dict, ranking: list[tuple[str, float]], formats: Iterable[str] = ("markdown", "json")) -> dict[str, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs: dict[str, Path] = {}
        if "json" in formats:
            path = self.output_dir / f"evaluation_{ts}.json"
            payload = {"ranking": ranking, "comparison": comparison}
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            outputs["json"] = path
        if "markdown" in formats:
            path = self.output_dir / f"evaluation_{ts}.md"
            path.write_text(self._render_markdown(comparison, ranking), encoding="utf-8")
            outputs["markdown"] = path
        return outputs

    def _render_markdown(self, comparison: dict, ranking: list[tuple[str, float]]) -> str:
        lines: list[str] = []
        lines.append("# LLM Evaluation Report")
        lines.append("")
        lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
        lines.append("")

        lines.append("## Prompt Ranking")
        lines.append("")
        lines.append("| Rank | Template | Overall | Correctness | Consistency | Hallucination |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for rank, (name, _score) in enumerate(ranking, start=1):
            d = comparison[name]
            lines.append(
                f"| {rank} | `{name}` | {d['average_score']:.3f} "
                f"| {d['average_correctness']:.3f} "
                f"| {d['average_consistency']:.3f} "
                f"| {d['average_hallucination']:.3f} |"
            )
        lines.append("")

        for name, data in comparison.items():
            lines.append(f"## Template: `{name}`")
            lines.append("")
            lines.append(
                f"- Average overall: **{data['average_score']:.3f}**  "
                f"(correctness {data['average_correctness']:.3f}, "
                f"consistency {data['average_consistency']:.3f}, "
                f"hallucination {data['average_hallucination']:.3f})"
            )
            lines.append("")
            lines.append("### Per-example results")
            lines.append("")
            lines.append("| Example | Overall | Exact | Semantic | Schema | Rules | Hallucination | Ungrounded terms |")
            lines.append("|---|---:|---:|---:|:---:|---:|---:|---|")
            for ex in data["examples"]:
                schema_ok = ex["correctness"].get("schema", {}).get("valid")
                schema_cell = "✓" if schema_ok else ("—" if schema_ok is None else "✗")
                rule_score = ex["correctness"].get("rule_based", {}).get("score", float("nan"))
                rule_cell = f"{rule_score:.2f}" if rule_score == rule_score else "—"
                ungrounded = ", ".join(ex["hallucination"]["ungrounded_terms"][:6])
                lines.append(
                    f"| `{ex['example_id']}` "
                    f"| {ex['overall']:.3f} "
                    f"| {ex['correctness']['exact_match']:.2f} "
                    f"| {ex['correctness']['semantic_similarity']:.2f} "
                    f"| {schema_cell} "
                    f"| {rule_cell} "
                    f"| {ex['hallucination']['combined_score']:.2f} "
                    f"| {ungrounded or '—'} |"
                )
            lines.append("")
            lines.append("### Sample output (first example)")
            lines.append("")
            first = data["examples"][0]
            lines.append("```text")
            lines.append(first["output"][:600])
            lines.append("```")
            lines.append("")
            schema_errors = first["correctness"].get("schema", {}).get("errors") or []
            if schema_errors:
                lines.append("**Schema errors:**")
                for e in schema_errors:
                    lines.append(f"- {e}")
                lines.append("")

        lines.append("## Notes")
        lines.append("")
        lines.append(
            "- **Overall** = weighted blend of correctness, consistency, and hallucination."
        )
        lines.append(
            "- **Correctness** combines exact match, semantic similarity, JSON schema validity, and rule checks."
        )
        lines.append(
            "- **Consistency** is mean pairwise semantic similarity across N runs."
        )
        lines.append(
            "- **Hallucination** is grounding ratio multiplied by consistency — output terms must appear in the expected answer or provided context."
        )
        return "\n".join(lines) + "\n"
