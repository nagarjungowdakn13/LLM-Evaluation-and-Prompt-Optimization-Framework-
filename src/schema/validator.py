import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def load_schema(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class SchemaValidator:
    """Parses LLM output as JSON and validates it against a JSON Schema.

    Tolerates models that wrap JSON in code fences or commentary by
    extracting the first balanced ``{...}`` block before parsing.
    """

    def __init__(self, schema: dict):
        self.schema = schema
        self.validator = Draft7Validator(schema)

    def extract_json(self, text: str) -> tuple[Any, str | None]:
        if not isinstance(text, str):
            return None, "input is not a string"
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass
        match = _JSON_BLOCK.search(text)
        if not match:
            return None, "no JSON object found"
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e.msg} at line {e.lineno} col {e.colno}"

    def validate(self, text: str | dict) -> dict:
        if isinstance(text, dict):
            obj, parse_error = text, None
        else:
            obj, parse_error = self.extract_json(text)

        if parse_error or obj is None:
            return {
                "valid": False,
                "parsed": None,
                "errors": [parse_error or "could not parse JSON"],
            }

        errors = sorted(self.validator.iter_errors(obj), key=lambda e: list(e.path))
        formatted = [self._format(e) for e in errors]
        return {"valid": not formatted, "parsed": obj, "errors": formatted}

    @staticmethod
    def _format(err) -> str:
        path = ".".join(str(p) for p in err.path) or "<root>"
        return f"{path}: {err.message}"
