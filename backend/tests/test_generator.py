import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.rag.generator import (  # noqa: E402
    _generation_context_metrics,
    build_no_context_answer_prompt,
    generate_no_context_answer,
    limit_chunks_to_prompt_budget,
)


class NoContextGeneratorTests(unittest.TestCase):
    def test_prompt_uses_question_without_retrieved_context(self):
        question = "What was the company's revenue growth?"

        messages = build_no_context_answer_prompt(question)
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn(question, prompt_text)
        self.assertIn("internal knowledge and reasoning", prompt_text)
        self.assertIn("FINAL_ANSWER:", prompt_text)
        self.assertNotIn("Retrieved context:", prompt_text)

    def test_missing_api_key_fails_before_generation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                generate_no_context_answer("What was the revenue?")


class _CharacterTokenizer:
    def encode(self, text):
        return list(str(text))

    def decode(self, tokens):
        return "".join(tokens)


class GenerationContextTests(unittest.TestCase):
    def test_context_metrics_report_unchanged_and_truncated_context(self):
        source_chunks = [
            {"id": "one", "text": "abcdef"},
            {"id": "two", "text": "ghij"},
        ]

        unchanged = _generation_context_metrics(
            source_chunks,
            [dict(chunk) for chunk in source_chunks],
        )
        truncated = _generation_context_metrics(
            source_chunks,
            [{"id": "one", "text": "abc"}],
        )

        self.assertFalse(unchanged["prompt_was_truncated"])
        self.assertEqual(unchanged["generation_context_chunk_count"], 2)
        self.assertTrue(truncated["prompt_was_truncated"])
        self.assertEqual(truncated["source_context_chunk_count"], 2)
        self.assertEqual(truncated["generation_context_chunk_count"], 1)
        self.assertEqual(truncated["generation_context_chunk_ids"], ["one"])
        self.assertEqual(truncated["source_context_character_count"], 10)
        self.assertEqual(truncated["generation_context_character_count"], 3)

    @patch(
        "app.rag.generator._get_tokenizer",
        return_value=_CharacterTokenizer(),
    )
    @patch(
        "app.rag.generator._count_prompt_tokens",
        side_effect=lambda messages, _model: sum(
            len(message["content"]) for message in messages
        ),
    )
    def test_oversized_first_chunk_is_trimmed_instead_of_dropped(
        self,
        _count_tokens,
        _get_tokenizer,
    ):
        def prompt_builder(question, retrieved_chunks):
            return [
                {
                    "role": "user",
                    "content": "".join(
                        f"SOURCE:{chunk['text']}"
                        for chunk in retrieved_chunks
                    ),
                }
            ]

        limited = limit_chunks_to_prompt_budget(
            question="question",
            retrieved_chunks=[{"id": "one", "text": "x" * 100}],
            model="test-model",
            max_prompt_tokens=20,
            prompt_builder=prompt_builder,
        )

        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["text"], "x" * 13)
        self.assertLessEqual(
            sum(
                len(message["content"])
                for message in prompt_builder("question", limited)
            ),
            20,
        )


if __name__ == "__main__":
    unittest.main()
