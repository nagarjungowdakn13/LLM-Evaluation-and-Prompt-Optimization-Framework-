import unittest

from src.schema import SchemaValidator


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "confidence", "answerable", "sources"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "answerable": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
}


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.v = SchemaValidator(SCHEMA)

    def test_valid_object(self):
        text = '{"answer": "42", "confidence": "high", "answerable": true, "sources": []}'
        result = self.v.validate(text)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_invalid_json(self):
        result = self.v.validate("not json at all")
        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])

    def test_missing_required_field(self):
        text = '{"answer": "hi", "confidence": "high", "sources": []}'
        result = self.v.validate(text)
        self.assertFalse(result["valid"])
        self.assertTrue(any("answerable" in e for e in result["errors"]))

    def test_extracts_json_from_wrapper_text(self):
        text = 'Here is your answer: {"answer": "ok", "confidence": "low", "answerable": true, "sources": []}'
        result = self.v.validate(text)
        self.assertTrue(result["valid"])

    def test_invalid_enum(self):
        text = '{"answer": "x", "confidence": "very high", "answerable": true, "sources": []}'
        result = self.v.validate(text)
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
