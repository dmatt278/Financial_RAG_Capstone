import importlib.util
from importlib.machinery import ModuleSpec
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


if importlib.util.find_spec("ijson") is None:
    ijson_stub = ModuleType("ijson")
    ijson_stub.__spec__ = ModuleSpec("ijson", loader=None)
    ijson_stub.items = lambda source, _prefix: iter(json.load(source))
    sys.modules["ijson"] = ijson_stub

if importlib.util.find_spec("huggingface_hub") is None:
    huggingface_stub = ModuleType("huggingface_hub")
    huggingface_stub.__spec__ = ModuleSpec("huggingface_hub", loader=None)
    huggingface_stub.hf_hub_download = lambda **_kwargs: ""
    sys.modules["huggingface_hub"] = huggingface_stub


from app.data.data_loader import (  # noqa: E402
    iter_docfinqa_examples,
    iter_sampled_docfinqa_examples,
)


def _raw_example(index: int) -> dict:
    return {
        "Question": f"Question {index}",
        "Answer": str(index),
        "Context": f"Document {index // 2}",
        "Program": f"add({index}, 0)",
    }


class DocFinQASamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_path = Path(self.temporary_directory.name) / "train.json"
        self.dataset_path.write_text(
            json.dumps([_raw_example(index) for index in range(30)]),
            encoding="utf-8",
        )
        path_patch = patch(
            "app.data.data_loader.get_docfinqa_file_path",
            return_value=str(self.dataset_path),
        )
        path_patch.start()
        self.addCleanup(path_patch.stop)
        items_patch = patch(
            "app.data.data_loader.ijson.items",
            side_effect=lambda source, _prefix: iter(json.load(source)),
        )
        items_patch.start()
        self.addCleanup(items_patch.stop)

    def test_seeded_sample_is_exact_reproducible_and_keeps_original_ids(self):
        first = list(
            iter_sampled_docfinqa_examples(
                split="train",
                sample_size=8,
                seed=42,
            )
        )
        second = list(
            iter_sampled_docfinqa_examples(
                split="train",
                sample_size=8,
                seed=42,
            )
        )

        first_ids = [example["question_id"] for example in first]
        second_ids = [example["question_id"] for example in second]

        self.assertEqual(len(first_ids), 8)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            [int(question_id) for question_id in first_ids],
            sorted(int(question_id) for question_id in first_ids),
        )
        self.assertNotEqual(first_ids, [str(index) for index in range(8)])

    def test_different_seed_selects_a_different_question_sample(self):
        first_ids = {
            example["question_id"]
            for example in iter_sampled_docfinqa_examples(
                split="train",
                sample_size=8,
                seed=42,
            )
        }
        second_ids = {
            example["question_id"]
            for example in iter_sampled_docfinqa_examples(
                split="train",
                sample_size=8,
                seed=43,
            )
        }

        self.assertNotEqual(first_ids, second_ids)

    def test_none_preserves_the_existing_full_iteration_behavior(self):
        sampled = list(
            iter_sampled_docfinqa_examples(
                split="train",
                sample_size=None,
            )
        )
        existing = list(iter_docfinqa_examples(split="train"))

        self.assertEqual(sampled, existing)
        self.assertEqual(len(sampled), 30)

    def test_rejects_sample_larger_than_the_split(self):
        with self.assertRaisesRegex(
            ValueError,
            "exceeds the 30 questions available",
        ):
            list(
                iter_sampled_docfinqa_examples(
                    split="train",
                    sample_size=31,
                    seed=42,
                )
            )

    def test_rejects_invalid_sample_arguments(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            list(
                iter_sampled_docfinqa_examples(
                    split="train",
                    sample_size=-1,
                )
            )

        with self.assertRaisesRegex(TypeError, "seed must be an integer"):
            list(
                iter_sampled_docfinqa_examples(
                    split="train",
                    sample_size=8,
                    seed="42",
                )
            )


if __name__ == "__main__":
    unittest.main()
