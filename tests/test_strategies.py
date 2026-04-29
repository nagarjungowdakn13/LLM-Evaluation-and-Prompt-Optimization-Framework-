import unittest

from src.prompts import PromptTemplate, annotate, annotate_all, group_by_strategy, rank_strategies
from src.prompts.strategies import PromptStrategy, detect_strategy


class TestStrategyDetection(unittest.TestCase):
    def test_explicit_header_wins(self):
        body = "# strategy: chain_of_thought\nDo the thing."
        self.assertEqual(detect_strategy(body), PromptStrategy.CHAIN_OF_THOUGHT)

    def test_inferred_few_shot(self):
        body = "Solve the problem.\n\nExample: in -> out\n\nQuestion: ..."
        self.assertEqual(detect_strategy(body), PromptStrategy.FEW_SHOT)

    def test_inferred_cot(self):
        body = "Reason step by step then answer."
        self.assertEqual(detect_strategy(body), PromptStrategy.CHAIN_OF_THOUGHT)

    def test_default_instruction(self):
        body = "Answer the question politely."
        self.assertEqual(detect_strategy(body), PromptStrategy.INSTRUCTION)


class TestRanking(unittest.TestCase):
    def test_rank_strategies(self):
        templates = [
            PromptTemplate("a", "# strategy: instruction\nx"),
            PromptTemplate("b", "# strategy: few_shot\nx"),
            PromptTemplate("c", "# strategy: chain_of_thought\nx"),
        ]
        annotated = annotate_all(templates)
        comparison = {
            "a": {"average_score": 0.4},
            "b": {"average_score": 0.7},
            "c": {"average_score": 0.9},
        }
        ranked = rank_strategies(comparison, annotated)
        self.assertEqual(ranked[0][0], PromptStrategy.CHAIN_OF_THOUGHT.value)
        self.assertEqual(ranked[-1][0], PromptStrategy.INSTRUCTION.value)

    def test_group_by_strategy(self):
        annotated = annotate_all([
            PromptTemplate("a", "# strategy: instruction\nx"),
            PromptTemplate("b", "# strategy: instruction\ny"),
            PromptTemplate("c", "# strategy: chain_of_thought\nz"),
        ])
        groups = group_by_strategy(annotated)
        self.assertEqual(len(groups[PromptStrategy.INSTRUCTION.value]), 2)
        self.assertEqual(len(groups[PromptStrategy.CHAIN_OF_THOUGHT.value]), 1)


if __name__ == "__main__":
    unittest.main()
