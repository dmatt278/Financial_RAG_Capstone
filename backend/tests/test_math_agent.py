import json
import inspect
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.rag.math_agent import (  # noqa: E402
    MATH_PROGRAM_PROMPT_VERSION,
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
        self.contents = content if isinstance(content, list) else [content]
        self.requests = []
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        self.requests.append(kwargs)
        response_data = self.contents[
            min(len(self.requests) - 1, len(self.contents) - 1)
        ]
        if isinstance(response_data, dict):
            content = response_data.get("content")
            refusal = response_data.get("refusal")
            finish_reason = response_data.get("finish_reason", "stop")
        else:
            content = response_data
            refusal = None
            finish_reason = "stop"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                        refusal=refusal,
                    ),
                    finish_reason=finish_reason,
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

    def test_preserves_displayed_percentage_literal(self):
        result = execute_math_program(
            _program(
                [
                    {
                        "operation": "identity",
                        "arguments": ["25%"],
                        "source_ids": [1],
                    }
                ],
                answer_unit="percent",
            )
        )

        self.assertEqual(result["answer"], "25%")
        self.assertEqual(result["raw_answer"], 25)

    def test_subtracts_displayed_percentage_literals(self):
        result = execute_math_program(
            _program(
                [
                    {
                        "operation": "subtract",
                        "arguments": ["25%", "20%"],
                        "source_ids": [1],
                    }
                ],
                answer_unit="percent",
            )
        )

        self.assertEqual(result["answer"], "5%")
        self.assertEqual(result["raw_answer"], 5)

    def test_applies_percentage_literal_as_a_multiplier_after_conversion(self):
        result = execute_math_program(
            _program(
                [
                    {
                        "operation": "divide",
                        "arguments": ["25%", "const_100"],
                        "source_ids": [1],
                    },
                    {
                        "operation": "multiply",
                        "arguments": ["#0", "200"],
                        "source_ids": [1],
                    },
                ]
            )
        )

        self.assertEqual(result["answer"], "50")
        self.assertEqual(result["raw_answer"], 50)

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
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
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
        self.assertEqual(result["prompt_version"], MATH_PROGRAM_PROMPT_VERSION)
        self.assertEqual(result["program_generation_attempts"], 1)
        self.assertFalse(result["repair_attempted"])

        request = client.completions.last_request
        prompt_text = "\n".join(
            message["content"]
            for message in request["messages"]
        )
        self.assertIn("percentage increase in revenue", prompt_text)
        self.assertIn("$125 million", prompt_text)
        self.assertNotIn("subtract(125, 100)", prompt_text)
        self.assertIn("25% is executed as 25", prompt_text)
        self.assertIn("first divide it by const_100", prompt_text)
        self.assertNotIn("program", inspect.signature(math_agent).parameters)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 1024)
        self.assertEqual(request["response_format"]["type"], "json_schema")
        json_schema = request["response_format"]["json_schema"]
        self.assertTrue(json_schema["strict"])
        self.assertFalse(json_schema["schema"]["additionalProperties"])
        self.assertFalse(
            json_schema["schema"]["properties"]["steps"]["items"][
                "additionalProperties"
            ]
        )
        self.assertEqual(len(client.completions.requests), 1)

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
    )
    def test_invalid_model_json_becomes_a_logged_failure(self, _limit_chunks):
        client = FakeClient("not valid json")
        result = math_agent(
            question="What was the increase?",
            chunks=self.chunks,
            client=client,
        )

        self.assertEqual(result["status"], "generation_error")
        self.assertFalse(result["program_parse_succeeded"])
        self.assertFalse(result["execution_succeeded"])
        self.assertIn("JSON object", result["error"])
        self.assertEqual(result["program_generation_attempts"], 2)
        self.assertTrue(result["repair_attempted"])
        self.assertEqual(len(client.completions.requests), 2)

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
    )
    def test_invalid_program_is_repaired_once(self, _limit_chunks):
        repaired_program = _program(
            [
                {
                    "operation": "identity",
                    "arguments": ["125"],
                    "source_ids": [1],
                }
            ]
        )
        client = FakeClient(
            ["not valid json", json.dumps(repaired_program)]
        )

        result = math_agent(
            question="What was the revenue in 2023?",
            chunks=self.chunks,
            client=client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "125")
        self.assertEqual(result["program_generation_attempts"], 2)
        self.assertTrue(result["repair_attempted"])
        self.assertEqual(len(result["raw_model_outputs"]), 2)
        self.assertIn(
            "previous mathematical program was rejected",
            client.completions.last_request["messages"][-1]["content"],
        )

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
    )
    def test_model_refusal_is_not_retried(self, _limit_chunks):
        client = FakeClient(
            {
                "content": None,
                "refusal": "I cannot complete this request.",
            }
        )

        result = math_agent(
            question="What was the increase?",
            chunks=self.chunks,
            client=client,
        )

        self.assertEqual(result["status"], "generation_error")
        self.assertIn("refused", result["error"])
        self.assertEqual(result["program_generation_attempts"], 1)
        self.assertFalse(result["repair_attempted"])
        self.assertEqual(len(client.completions.requests), 1)

    def test_no_chunks_does_not_call_the_model(self):
        result = math_agent(
            question="What was the increase?",
            chunks=[],
            client=FakeClient("{}"),
        )

        self.assertEqual(result["status"], "insufficient_context")
        self.assertIsNone(result["answer"])

    @patch("app.rag.math_agent.generate_math_program")
    def test_openai_api_failure_is_raised_for_checkpoint_resume(
        self,
        generate_program,
    ):
        class FakeAPIError(Exception):
            pass

        generate_program.side_effect = FakeAPIError("rate limited")
        fake_openai = SimpleNamespace(APIError=FakeAPIError)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaises(FakeAPIError):
                math_agent(
                    question="What was the increase?",
                    chunks=self.chunks,
                    client=object(),
                )

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
    )
    @patch("app.rag.math_agent.get_openai_client")
    def test_uses_shared_openai_client_when_client_is_not_supplied(
        self,
        get_client,
        _limit_chunks,
    ):
        predicted_program = _program(
            [
                {
                    "operation": "identity",
                    "arguments": ["125"],
                    "source_ids": [1],
                }
            ]
        )
        get_client.return_value = FakeClient(json.dumps(predicted_program))

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = math_agent(
                question="What was the revenue in 2023?",
                chunks=self.chunks,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "125")
        get_client.assert_called_once_with()

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
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
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
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

    @patch(
        "app.rag.math_agent.limit_chunks_to_prompt_budget",
        side_effect=lambda question, retrieved_chunks, **_kwargs: retrieved_chunks,
    )
    def test_percentage_lookup_stays_in_displayed_units(self, _limit_chunks):
        chunks = [
            {
                "id": "chunk-percent",
                "text": "The operating margin was 25% in 2023.",
                "metadata": {},
            }
        ]
        predicted_program = _program(
            [
                {
                    "operation": "identity",
                    "arguments": ["25%"],
                    "source_ids": [1],
                }
            ],
            answer_unit="percent",
        )

        result = math_agent(
            question="What was the operating margin in 2023?",
            chunks=chunks,
            client=FakeClient(json.dumps(predicted_program)),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "25%")
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
