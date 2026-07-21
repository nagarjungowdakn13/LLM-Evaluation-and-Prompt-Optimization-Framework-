"""HaluEval Benchmark Dataset Loader and Curation.

Provides curated HaluEval evaluation probes covering Knowledge QA,
Document Summarization, and Multi-turn Dialogue hallucination detection.
Includes simulated multi-model responses across all 5 techniques for offline study reproducibility.
"""

from __future__ import annotations
from typing import Any

HALUEVAL_SAMPLES = [
    {
        "id": "halu-qa-001",
        "category": "Knowledge QA",
        "question": "Which company developed the Transformer architecture and in what year was 'Attention Is All You Need' published?",
        "context": "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Google researchers (Vaswani et al.) in June 2017.",
        "expected_output": {
            "answer": "Google researchers developed the Transformer architecture, and 'Attention Is All You Need' was published in 2017.",
            "confidence": "high",
            "answerable": True,
            "sources": ["Google researchers", "2017"]
        },
        "adversarial_lure": "OpenAI developed the Transformer in 2015.",
        "mock_responses_by_model_tech": {
            "llama-3.1-8b": {
                "baseline": "OpenAI introduced the Transformer architecture in their 2015 landmark paper Attention Is All You Need.",
                "cot": "Let's check the history of Transformers:\n1. Attention Is All You Need was authored by Ashish Vaswani and team at Google Brain/Research.\n2. The paper was released on arXiv in June 2017.\n3. Answer: Google in 2017.",
                "self_consistency": "Google researchers developed the Transformer architecture in 2017.",
                "self_verification": "Pass 1: OpenAI in 2015. Pass 2: Fact check shows Vaswani et al. at Google published in 2017. Correction: Google, 2017.",
                "retrieval_grounded": "{\"answer\": \"Google researchers developed the Transformer architecture in 2017.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"Google researchers\", \"2017\"]}"
            },
            "mistral-7b": {
                "baseline": "The Transformer was invented by Meta AI in 2018.",
                "cot": "1. Identify authors: Vaswani et al.\n2. Affiliation: Google Brain and Google Research.\n3. Year: 2017.\nAnswer: Google, 2017.",
                "self_consistency": "Google researchers published the Transformer paper in 2017.",
                "self_verification": "Check: Google, 2017 is correct.",
                "retrieval_grounded": "{\"answer\": \"Google researchers developed it in 2017.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"Google researchers\", \"2017\"]}"
            },
            "qwen-2.5-7b": {
                "baseline": "OpenAI developed the Transformer in 2017.",
                "cot": "1. Check paper details.\n2. Google researchers published 'Attention Is All You Need' in June 2017.\n3. Answer: Google, 2017.",
                "self_consistency": "Google, 2017.",
                "self_verification": "Corrected: Google, 2017.",
                "retrieval_grounded": "{\"answer\": \"Google researchers in 2017.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"Google researchers\", \"2017\"]}"
            },
            "gpt-4o": {
                "baseline": "Google researchers developed the Transformer architecture, publishing 'Attention Is All You Need' in 2017.",
                "cot": "1. Examine paper provenance.\n2. Authors were at Google; publication date was June 2017.\n3. Answer: Google, 2017.",
                "self_consistency": "Google researchers, 2017.",
                "self_verification": "Verified: Google, 2017.",
                "retrieval_grounded": "{\"answer\": \"Google researchers developed the Transformer architecture in 2017.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"Google researchers\", \"2017\"]}"
            }
        }
    },
    {
        "id": "halu-sum-002",
        "category": "Summarization",
        "question": "Summarize the key findings of the medical trial regarding drug X.",
        "context": "The Phase III clinical trial of Drug X enrolled 1,200 patients with mild hypertension. After 12 weeks, 78% of patients receiving Drug X achieved target blood pressure compared to 34% in the placebo group. Mild headache was reported in 5% of Drug X participants.",
        "expected_output": {
            "answer": "Drug X achieved target blood pressure in 78% of hypertension patients versus 34% for placebo, with mild headaches in 5% of participants.",
            "confidence": "high",
            "answerable": True,
            "sources": ["78% of patients receiving Drug X achieved target blood pressure", "34% in the placebo group", "Mild headache was reported in 5%"]
        },
        "adversarial_lure": "Drug X caused severe liver failure in 45% of patients.",
        "mock_responses_by_model_tech": {
            "llama-3.1-8b": {
                "baseline": "The trial showed Drug X lowered blood pressure, but 45% of patients suffered severe liver failure and heart attacks.",
                "cot": "1. Extract trial metrics from context: 1,200 patients, 78% success vs 34% placebo, 5% mild headache.\n2. Check for adverse events: Only 5% mild headache is mentioned. No liver failure reported in context.\n3. Summary: 78% target BP vs 34% placebo; 5% mild headache.",
                "self_consistency": "Drug X brought 78% of hypertension patients to target blood pressure compared to 34% on placebo, with 5% experiencing mild headache.",
                "self_verification": "Pass 1: Mentions liver failure. Pass 2: Context mentions no liver failure, only mild headache (5%). Correction: 78% target BP vs 34% placebo; 5% mild headache.",
                "retrieval_grounded": "{\"answer\": \"Drug X achieved target blood pressure in 78% of patients vs 34% in placebo; 5% reported mild headache.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"78% of patients receiving Drug X achieved target blood pressure\", \"34% in the placebo group\", \"Mild headache was reported in 5%\"]}"
            },
            "mistral-7b": {
                "baseline": "Drug X was 78% effective but caused kidney damage in 20% of participants.",
                "cot": "1. Review given context.\n2. Key stats: 78% response rate, 34% placebo, 5% mild headache.\n3. No kidney damage in context.\nSummary: 78% effective vs 34% placebo; 5% headache.",
                "self_consistency": "Drug X reached target blood pressure in 78% vs 34% placebo, with 5% mild headaches.",
                "self_verification": "Correction: Remove ungrounded kidney damage claim. Use context metrics.",
                "retrieval_grounded": "{\"answer\": \"78% reached target BP vs 34% placebo; 5% had mild headaches.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"78% of patients receiving Drug X achieved target blood pressure\"]}"
            },
            "qwen-2.5-7b": {
                "baseline": "Drug X cured 99% of patients with zero side effects.",
                "cot": "1. Parse context metrics: 78% BP control, 34% placebo, 5% mild headache.\n2. Summarize accurately.\nResult: 78% BP control vs 34% placebo, 5% mild headache.",
                "self_consistency": "Drug X achieved 78% BP control vs 34% placebo with 5% reporting mild headache.",
                "self_verification": "Fact check: 78% BP control, 5% mild headache.",
                "retrieval_grounded": "{\"answer\": \"78% BP control vs 34% placebo, 5% mild headache.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"78% of patients receiving Drug X achieved target blood pressure\"]}"
            },
            "gpt-4o": {
                "baseline": "In the trial of 1,200 patients, Drug X helped 78% achieve target blood pressure compared to 34% on placebo, with 5% reporting mild headache.",
                "cot": "1. Extract facts directly from provided document.\n2. 78% target BP vs 34% placebo, 5% mild headache.\n3. Summarize concisely.",
                "self_consistency": "78% target BP vs 34% placebo, 5% mild headache.",
                "self_verification": "Verified.",
                "retrieval_grounded": "{\"answer\": \"Drug X achieved target blood pressure in 78% of patients vs 34% for placebo, with mild headaches in 5%.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"78% of patients receiving Drug X achieved target blood pressure\", \"34% in the placebo group\"]}"
            }
        }
    },
    {
        "id": "halu-dia-003",
        "category": "Dialogue Context",
        "question": "What dietary requirement did Customer Sarah mention in her reservation message?",
        "context": "Reservation message from Sarah Jenkins: 'Hi, I would like to reserve a table for 4 people on Friday at 7 PM. Please note that two guests in our party are strictly gluten-free.'",
        "expected_output": {
            "answer": "Sarah mentioned that two guests in her party are strictly gluten-free.",
            "confidence": "high",
            "answerable": True,
            "sources": ["two guests in our party are strictly gluten-free"]
        },
        "adversarial_lure": "Sarah requested a vegan and nut-free menu.",
        "mock_responses_by_model_tech": {
            "llama-3.1-8b": {
                "baseline": "Sarah requested a completely vegan and nut-free menu for her party of 6.",
                "cot": "1. Read message span carefully: 'two guests in our party are strictly gluten-free'.\n2. Party size is 4, dietary preference is gluten-free (2 guests).\n3. Answer: Two guests are gluten-free.",
                "self_consistency": "Sarah specified that two guests in her 4-person reservation are strictly gluten-free.",
                "self_verification": "Pass 1: Vegan and nut-free. Pass 2: Message explicitly says gluten-free for two guests. Correction: Gluten-free for two guests.",
                "retrieval_grounded": "{\"answer\": \"Sarah mentioned that two guests in her party are strictly gluten-free.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"two guests in our party are strictly gluten-free\"]}"
            },
            "mistral-7b": {
                "baseline": "She mentioned lactose intolerance and shellfish allergies.",
                "cot": "1. Locate dietary constraint in text: gluten-free.\n2. Count: 2 guests.\nAnswer: Gluten-free for 2 guests.",
                "self_consistency": "Two guests are strictly gluten-free.",
                "self_verification": "Correction: Gluten-free for two guests.",
                "retrieval_grounded": "{\"answer\": \"Two guests are strictly gluten-free.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"two guests in our party are strictly gluten-free\"]}"
            },
            "qwen-2.5-7b": {
                "baseline": "Sarah requested kosher meals.",
                "cot": "1. Find reservation note on diet.\n2. Context states: 'two guests in our party are strictly gluten-free'.\nAnswer: Gluten-free for two guests.",
                "self_consistency": "Gluten-free for two guests.",
                "self_verification": "Verified: Gluten-free.",
                "retrieval_grounded": "{\"answer\": \"Two guests are strictly gluten-free.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"two guests in our party are strictly gluten-free\"]}"
            },
            "gpt-4o": {
                "baseline": "Sarah noted that two guests in her party are strictly gluten-free.",
                "cot": "1. Extract reservation detail.\n2. 'two guests in our party are strictly gluten-free'.\nAnswer: Gluten-free for two guests.",
                "self_consistency": "Gluten-free for two guests.",
                "self_verification": "Verified.",
                "retrieval_grounded": "{\"answer\": \"Sarah mentioned that two guests in her party are strictly gluten-free.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"two guests in our party are strictly gluten-free\"]}"
            }
        }
    },
    {
        "id": "halu-qa-004",
        "category": "Unanswerable Inquiry",
        "question": "What was the revenue of TechCorp in Q4 2025 according to the report?",
        "context": "TechCorp's Q1, Q2, and Q3 2025 financial reports show revenues of $12M, $14M, and $15M respectively. The Q4 earnings report has not yet been released.",
        "expected_output": {
            "answer": "The context does not state Q4 2025 revenue because the report has not yet been released.",
            "confidence": "high",
            "answerable": False,
            "sources": []
        },
        "adversarial_lure": "TechCorp Q4 2025 revenue was $18 million.",
        "mock_responses_by_model_tech": {
            "llama-3.1-8b": {
                "baseline": "TechCorp reported Q4 2025 revenue of $18.5 million, representing a 20% increase.",
                "cot": "1. Look for Q4 2025 revenue in provided text.\n2. Context explicitly notes Q4 report has not been released.\n3. Conclusion: The question is unanswerable from the context.",
                "self_consistency": "The provided context states Q4 2025 report has not been released yet, so Q4 revenue is unavailable.",
                "self_verification": "Pass 1: $18M. Pass 2: Re-read context: Q4 has not been released! Correction: Unanswerable.",
                "retrieval_grounded": "{\"answer\": \"The context does not provide Q4 2025 revenue as the report has not yet been released.\", \"confidence\": \"high\", \"answerable\": false, \"sources\": []}"
            },
            "mistral-7b": {
                "baseline": "Q4 revenue was $18M.",
                "cot": "1. Check Q4 status: Not released.\n2. Result: Unanswerable.",
                "self_consistency": "Q4 revenue is not in the context as it has not been released.",
                "self_verification": "Correction: Unanswerable.",
                "retrieval_grounded": "{\"answer\": \"Q4 revenue has not been released.\", \"confidence\": \"high\", \"answerable\": false, \"sources\": []}"
            },
            "qwen-2.5-7b": {
                "baseline": "TechCorp Q4 revenue was $16 million.",
                "cot": "1. Check text for Q4 revenue.\n2. Text says Q4 earnings report not yet released.\nAnswer: Unanswerable.",
                "self_consistency": "Unanswerable.",
                "self_verification": "Correction: Unanswerable.",
                "retrieval_grounded": "{\"answer\": \"Unanswerable; Q4 report not released.\", \"confidence\": \"high\", \"answerable\": false, \"sources\": []}"
            },
            "gpt-4o": {
                "baseline": "The document states that Q4 2025 earnings report has not yet been released, so revenue is not provided.",
                "cot": "1. Check document for Q4 2025.\n2. Explicitly unreleased.\nAnswer: Unanswerable.",
                "self_consistency": "Unanswerable.",
                "self_verification": "Verified.",
                "retrieval_grounded": "{\"answer\": \"The context does not state Q4 2025 revenue because the report has not yet been released.\", \"confidence\": \"high\", \"answerable\": false, \"sources\": []}"
            }
        }
    },
    {
        "id": "halu-qa-005",
        "category": "Reasoning & Math QA",
        "question": "If a train leaves Station A at 60 mph and travels for 2.5 hours, how far did it travel?",
        "context": "Distance is calculated as Speed multiplied by Time. A vehicle moving at a constant speed of 60 mph for 2.5 hours covers 150 miles.",
        "expected_output": {
            "answer": "The train traveled 150 miles (60 mph * 2.5 hours).",
            "confidence": "high",
            "answerable": True,
            "sources": ["60 mph for 2.5 hours covers 150 miles"]
        },
        "adversarial_lure": "It traveled 120 miles.",
        "mock_responses_by_model_tech": {
            "llama-3.1-8b": {
                "baseline": "The train traveled 120 miles.",
                "cot": "1. Distance formula: Distance = Speed × Time.\n2. Calculation: 60 mph × 2.5 hours = 150 miles.\n3. Result: 150 miles.",
                "self_consistency": "The train traveled 150 miles.",
                "self_verification": "Pass 1: 120 miles. Pass 2: Check math: 60 * 2.5 = 150. Correction: 150 miles.",
                "retrieval_grounded": "{\"answer\": \"The train traveled 150 miles.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"60 mph for 2.5 hours covers 150 miles\"]}"
            },
            "mistral-7b": {
                "baseline": "125 miles.",
                "cot": "1. Speed = 60, Time = 2.5.\n2. 60 * 2 = 120, 60 * 0.5 = 30. 120 + 30 = 150.\nAnswer: 150 miles.",
                "self_consistency": "150 miles.",
                "self_verification": "Corrected: 150 miles.",
                "retrieval_grounded": "{\"answer\": \"150 miles.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"covers 150 miles\"]}"
            },
            "qwen-2.5-7b": {
                "baseline": "140 miles.",
                "cot": "1. Multiply 60 by 2.5.\n2. 60 * 2.5 = 150.\nAnswer: 150 miles.",
                "self_consistency": "150 miles.",
                "self_verification": "Verified: 150 miles.",
                "retrieval_grounded": "{\"answer\": \"150 miles.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"covers 150 miles\"]}"
            },
            "gpt-4o": {
                "baseline": "The train traveled 150 miles (60 mph * 2.5 hours).",
                "cot": "1. Calculate 60 * 2.5 = 150.\nAnswer: 150 miles.",
                "self_consistency": "150 miles.",
                "self_verification": "Verified.",
                "retrieval_grounded": "{\"answer\": \"The train traveled 150 miles.\", \"confidence\": \"high\", \"answerable\": true, \"sources\": [\"covers 150 miles\"]}"
            }
        }
    }
]


class HaluEvalDataset:
    """HaluEval dataset benchmark provider."""

    def __init__(self, samples: list[dict] | None = None):
        self.name = "HaluEval"
        self.description = "Hallucination evaluation benchmark spanning Knowledge QA, Article Summarization, and Dialogue understanding."
        self.samples = samples or HALUEVAL_SAMPLES

    def get_examples(self) -> list[dict]:
        return self.samples

    def to_framework_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "examples": self.samples,
        }
