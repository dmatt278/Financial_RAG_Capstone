import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app import evaluation  # noqa: E402
from app.evaluation import align_docfinqa_evidence  # noqa: E402


class DocFinQAEvidenceAlignmentTests(unittest.TestCase):
    def test_alignment_preserves_long_evidence_ties_and_sorted_ids(self):
        chunks = [
            {
                "id": "chunk-b",
                "text": "Revenue increased from 100 to 120 during 2023.",
            },
            {
                "id": "chunk-a",
                "text": "Revenue increased from 100 to 120 during 2023.",
            },
            {
                "id": "chunk-c",
                "text": "Revenue decreased during 2022.",
            },
        ]

        alignments = align_docfinqa_evidence(
            ["Revenue increased from 100 to 120 during 2023."],
            chunks,
        )

        self.assertEqual(
            alignments,
            [
                {
                    "evidence": (
                        "Revenue increased from 100 to 120 during 2023."
                    ),
                    "score": 1.0,
                    "target_chunk_ids": ["chunk-a", "chunk-b"],
                }
            ],
        )

    def test_alignment_preserves_short_evidence_overlap_and_duplicates(self):
        chunks = [
            {"id": "chunk-b", "text": "Cash growth remained strong."},
            {"id": "chunk-a", "text": "Strong cash growth continued."},
            {"id": "chunk-c", "text": "Cash declined."},
        ]

        alignments = align_docfinqa_evidence(
            ["cash cash growth", "CASH, cash growth!"],
            chunks,
        )

        self.assertEqual(
            alignments,
            [
                {
                    "evidence": "cash cash growth",
                    "score": 1.0,
                    "target_chunk_ids": ["chunk-a", "chunk-b"],
                }
            ],
        )

    def test_alignment_precomputes_each_chunk_and_evidence_once(self):
        chunks = [
            {"id": "chunk-a", "text": "one two three four five six"},
            {"id": "chunk-b", "text": "seven eight nine ten eleven twelve"},
            {"id": "chunk-c", "text": "thirteen fourteen fifteen sixteen"},
        ]
        evidence = [
            "one two three four five",
            "seven eight nine ten eleven",
        ]

        original_tokens = evaluation._tokens
        original_ngrams = evaluation._ngrams
        with patch(
            "app.evaluation._tokens",
            wraps=original_tokens,
        ) as token_spy, patch(
            "app.evaluation._ngrams",
            wraps=original_ngrams,
        ) as ngram_spy:
            align_docfinqa_evidence(evidence, chunks)

        expected_feature_count = len(chunks) + len(evidence)
        self.assertEqual(token_spy.call_count, expected_feature_count)
        self.assertEqual(ngram_spy.call_count, expected_feature_count)


if __name__ == "__main__":
    unittest.main()
