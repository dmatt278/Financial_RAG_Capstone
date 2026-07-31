import json
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.rag.math_agent import (  # noqa: E402
    execute_math_program,
    math_agent,
    validate_math_program,
)
from app.evaluation import evaluate_docfinqa_answer_metrics  # noqa: E402


def _program(steps, answer_unit="number", answer_scale="none"):
    return {
        "status": "ok",
        "reason": "The cited values provide the required operands.",
        "steps": steps,
        "answer_unit": answer_unit,
        "answer_scale": answer_scale,
    }


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class FakeClient:
    def __init__(self, content):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class MathProgramExecutionTests(unittest.TestCase):
    def test_executes_references_and_formats_percentage(self):
        program = _program(
            [
                {
                    "operation": "subtract",
                    "arguments": ["125", "100"],
                    "source_ids": [1],
                },
                {
                    "operation": "divide",
                    "arguments": ["#0", "100"],
                    "source_ids": [1],
                },
                {
                    "operation": "multiply",
                    "arguments": ["#1", "const_100"],
                    "source_ids": [],
                },
            ],
            answer_unit="percent",
        )

        result = execute_math_program(program)

        self.assertEqual(result["answer"], "25%")
        self.assertAlmostEqual(result["raw_answer"], 25)

    def test_greater_returns_yes_or_no(self):
        yes_result = execute_math_program(
            _program(
                [
                    {
                        "operation": "greater",
                        "arguments": ["125", "100"],
                        "source_ids": [1],
                    }
                ],
                answer_unit="yes_no",
            )
        )
        no_result = execute_math_program(
            _program(
                [
                    {
                        "operation": "greater",
                        "arguments": ["75", "100"],
                        "source_ids": [1],
                    }
                ],
                answer_unit="yes_no",
            )
        )

        self.assertEqual(yes_result["answer"], "yes")
        self.assertEqual(no_result["answer"], "no")

    def test_formats_the_requested_magnitude_scale(self):
        result = execute_math_program(
            _program(
                [
                    {
                        "operation": "identity",
                        "arguments": ["13"],
                        "source_ids": [1],
                    }
                ],
                answer_scale="million",
            )
        )

        self.assertEqual(result["answer"], "13 million")

    def test_rejects_arbitrary_constants(self):
        with self.assertRaisesRegex(ValueError, "Unsupported mathematical constant"):
            validate_math_program(
                _program(
                    [
                        {
                            "operation": "identity",
                            "arguments": ["const_380"],
                            "source_ids": [],
                        }
                    ]
                )
            )

    def test_rejects_forward_reference(self):
        with self.assertRaisesRegex(ValueError, "forward or self reference"):
            validate_math_program(
                _program(
                    [
                        {
                            "operation": "add",
                            "arguments": ["#0", "1"],
                            "source_ids": [1],
                        }
                    ]
                )
            )

    def test_rejects_division_by_zero(self):
        with self.assertRaisesRegex(ValueError, "Division by zero"):
            execute_math_program(
                _program(
                    [
                        {
                            "operation": "divide",
                            "arguments": ["100", "0"],
                            "source_ids": [1],
                        }
                    ]
                )
            )


class MathAgentGenerationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "chunk-1",
                "chunk_id": "chunk-1",
                "text": (
                    "Revenue was $100 million in 2022 and "
                    "$125 million in 2023."
                ),
                "metadata": {"question_id": "example-question"},
            }
        ]

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, model, prompt_builder: (
            retrieved_chunks
        ),
    )
    def test_llm_plan_is_executed_without_a_gold_program(self, _limit_chunks):
        predicted_program = _program(
            [
                {
                    "operation": "subtract",
                    "arguments": ["125", "100"],
                    "source_ids": [1],
                },
                {
                    "operation": "divide",
                    "arguments": ["#0", "100"],
                    "source_ids": [1],
                },
                {
                    "operation": "multiply",
                    "arguments": ["#1", "const_100"],
                    "source_ids": [],
                },
            ],
            answer_unit="percent",
        )
        client = FakeClient(json.dumps(predicted_program))

        result = math_agent(
            question=(
                "What was the percentage increase in revenue from "
                "2022 to 2023?"
            ),
            chunks=self.chunks,
            client=client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "25%")
        self.assertTrue(result["program_parse_succeeded"])
        self.assertTrue(result["execution_succeeded"])
        self.assertEqual(result["operand_grounding_rate"], 1.0)

        request = client.completions.last_request
        prompt_text = "\n".join(
            message["content"]
            for message in request["messages"]
        )
        self.assertIn("percentage increase in revenue", prompt_text)
        self.assertIn("$125 million", prompt_text)
        self.assertNotIn("subtract(125, 100)", prompt_text)
        self.assertNotIn("program", inspect.signature(math_agent).parameters)
        self.assertEqual(request["temperature"], 0)

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, model, prompt_builder: (
            retrieved_chunks
        ),
    )
    def test_invalid_model_json_becomes_a_logged_failure(self, _limit_chunks):
        result = math_agent(
            question="What was the increase?",
            chunks=self.chunks,
            client=FakeClient("not valid json"),
        )

        self.assertEqual(result["status"], "generation_error")
        self.assertFalse(result["program_parse_succeeded"])
        self.assertFalse(result["execution_succeeded"])
        self.assertIn("JSON object", result["error"])

    def test_no_chunks_does_not_call_the_model(self):
        result = math_agent(
            question="What was the increase?",
            chunks=[],
            client=FakeClient("{}"),
        )

        self.assertEqual(result["status"], "insufficient_context")
        self.assertIsNone(result["answer"])

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, model, prompt_builder: (
            retrieved_chunks
        ),
    )
    def test_ungrounded_operand_is_not_executed(self, _limit_chunks):
        predicted_program = _program(
            [
                {
                    "operation": "identity",
                    "arguments": ["999"],
                    "source_ids": [1],
                }
            ]
        )

        result = math_agent(
            question="What was the revenue?",
            chunks=self.chunks,
            client=FakeClient(json.dumps(predicted_program)),
        )

        self.assertEqual(result["status"], "grounding_error")
        self.assertFalse(result["execution_succeeded"])
        self.assertEqual(result["operand_grounding_rate"], 0.0)

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, model, prompt_builder: (
            retrieved_chunks
        ),
    )
    def test_question_operand_can_use_source_zero(self, _limit_chunks):
        predicted_program = _program(
            [
                {
                    "operation": "identity",
                    "arguments": ["5"],
                    "source_ids": [0],
                }
            ]
        )

        result = math_agent(
            question="How many years were included? Use 5 years.",
            chunks=self.chunks,
            client=FakeClient(json.dumps(predicted_program)),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "5")
        self.assertEqual(result["operand_grounding_rate"], 1.0)


class DocFinQAAnswerMetricTests(unittest.TestCase):
    def test_uses_the_gold_answers_displayed_rounding_precision(self):
        metrics = evaluate_docfinqa_answer_metrics(
            generated_answer="53.231693%",
            gold_answer="53%",
        )

        self.assertTrue(metrics["docfinqa_answer_correct"])
        self.assertFalse(metrics["exact_normalized_match"])
        self.assertEqual(metrics["gold_display_decimal_places"], 0)

    def test_preserves_magnitude_units_for_comparison(self):
        matching = evaluate_docfinqa_answer_metrics(
            generated_answer="13 million",
            gold_answer="$ 13 million",
        )
        omitted_generated_scale = evaluate_docfinqa_answer_metrics(
            generated_answer="13",
            gold_answer="$ 13 million",
        )
        omitted_gold_scale = evaluate_docfinqa_answer_metrics(
            generated_answer="41932 million",
            gold_answer="41932",
        )
        conflicting_scales = evaluate_docfinqa_answer_metrics(
            generated_answer="13 billion",
            gold_answer="$ 13 million",
        )
        equivalent_scale_labels = evaluate_docfinqa_answer_metrics(
            generated_answer="13m",
            gold_answer="$ 13 million",
        )

        self.assertTrue(matching["docfinqa_answer_correct"])
        self.assertEqual(matching["normalized_generated_answer"], 13)
        self.assertTrue(omitted_generated_scale["docfinqa_answer_correct"])
        self.assertTrue(omitted_gold_scale["docfinqa_answer_correct"])
        self.assertFalse(conflicting_scales["docfinqa_answer_correct"])
        self.assertTrue(conflicting_scales["explicit_magnitude_conflict"])
        self.assertTrue(equivalent_scale_labels["docfinqa_answer_correct"])


if __name__ == "__main__":
    unittest.main()
