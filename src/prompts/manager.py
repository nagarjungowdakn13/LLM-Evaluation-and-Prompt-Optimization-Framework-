import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    """A named prompt with a Python ``str.format``-style body.

    Templates always receive an ``example`` mapping when rendered. The
    render step also injects routing tags ``[TEMPLATE_ID:...]`` and
    ``[EXAMPLE_ID:...]`` so the offline mock client can dispatch outputs
    deterministically.
    """

    name: str
    body: str
    metadata: dict = field(default_factory=dict)

    def render(self, example: dict[str, Any]) -> str:
        formatter = _SafeFormatter()
        # Drop annotation lines like "# strategy: ..." so they don't leak to the LLM.
        cleaned_body = "\n".join(
            line for line in self.body.splitlines()
            if not line.strip().lower().startswith("# strategy:")
        )
        rendered = formatter.format(cleaned_body, **example)
        header = f"[TEMPLATE_ID: {self.name}]\n[EXAMPLE_ID: {example['id']}]\n"
        return header + rendered


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)


class PromptManager:
    def __init__(self) -> None:
        self.templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        self.templates[template.name] = template

    def load_directory(self, directory: str | Path) -> "PromptManager":
        for path in sorted(Path(directory).glob("*.txt")):
            body = path.read_text(encoding="utf-8")
            self.register(PromptTemplate(name=path.stem, body=body))
        return self

    def get(self, name: str) -> PromptTemplate:
        return self.templates[name]

    def list(self, names: list[str] | None = None) -> list[PromptTemplate]:
        if names is None:
            return list(self.templates.values())
        return [self.templates[n] for n in names if n in self.templates]
