import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.rag.parameter_selection import (  # noqa: E402
    RETRIEVAL_SCORE_WEIGHTS,
    build_reranker_configs,
    is_complete_parameter_sweep,
    record_parameter_result,
    record_retrieval_result,
    rank_retrieval_parameter_configs,
    select_best_parameter_config,
)


def _record(
    scores,
    config_key,
    *,
    correct,
    within_one_percent=False,
    all_evidence_hit=0.0,
    evidence_recall=0.0,
    reciprocal_rank=0.0,
    precision=0.0,
):
    record_parameter_result(
        parameter_scores=scores,
        config_key=config_key,
        is_correct=correct,
        retrieval_metrics={
            "all_evidence_hit_at_k": all_evidence_hit,
            "evidence_recall_at_k": evidence_recall,
            "reciprocal_rank": reciprocal_rank,
            "precision_at_k": precision,
        },
        generation_metrics={
            "within_one_percent": within_one_percent,
        },
    )


class ParameterSelectionTests(unittest.TestCase):
    def test_retrieval_ranking_uses_preregistered_weights(self):
        scores = {}
        balanced_config = (512, "section", "hybrid", 5, True, 20)
        all_evidence_config = (256, "fixed", "semantic", 3, False, None)

        record_retrieval_result(
            parameter_scores=scores,
            config_key=balanced_config,
            retrieval_metrics={
                "all_evidence_hit_at_k": 0.8,
                "evidence_recall_at_k": 0.8,
                "reciprocal_rank": 0.8,
                "precision_at_k": 0.8,
            },
        )
        record_retrieval_result(
            parameter_scores=scores,
            config_key=all_evidence_config,
            retrieval_metrics={
                "all_evidence_hit_at_k": 1.0,
                "evidence_recall_at_k": 0.5,
                "reciprocal_rank": 0.0,
                "precision_at_k": 0.0,
            },
        )

        ranked = rank_retrieval_parameter_configs(scores)

        self.assertEqual(
            RETRIEVAL_SCORE_WEIGHTS,
            {
                "avg_all_evidence_hit_at_k": 0.50,
                "avg_evidence_recall_at_k": 0.30,
                "avg_reciprocal_rank": 0.10,
                "avg_precision_at_k": 0.10,
            },
        )
        self.assertEqual(ranked[0]["chunk_size"], 512)
        self.assertAlmostEqual(
            ranked[0]["selection_metrics"]["retrieval_score"],
            0.8,
        )
        self.assertTrue(
            ranked[0]["selection_metrics"]["retrieval_metrics_complete"]
        )
        self.assertAlmostEqual(
            ranked[1]["selection_metrics"]["retrieval_score"],
            0.65,
        )

    def test_retrieval_shortlist_returns_exact_requested_limit(self):
        scores = {}
        for index in range(15):
            record_retrieval_result(
                parameter_scores=scores,
                config_key=(
                    256 + index,
                    "fixed",
                    "semantic",
                    3,
                    False,
                    None,
                ),
                retrieval_metrics={
                    "all_evidence_hit_at_k": index / 15,
                    "evidence_recall_at_k": 0.5,
                    "reciprocal_rank": 0.5,
                    "precision_at_k": 0.5,
                },
            )

        ranked = rank_retrieval_parameter_configs(scores, limit=12)

        self.assertEqual(len(ranked), 12)
        self.assertEqual(ranked[0]["chunk_size"], 270)
        self.assertEqual(ranked[-1]["chunk_size"], 259)

    def test_restricted_grid_is_not_a_complete_parameter_sweep(self):
        common_arguments = {
            "strategies": ["fixed", "sentence", "section"],
            "retrieval_methods": ["keyword", "semantic", "hybrid"],
            "chunk_sizes": [256, 512, 1024],
            "reranker_configs": [
                (False, None),
                (True, 10),
                (True, 20),
                (True, 40),
            ],
            "expected_top_k_values": (3, 5, 10),
            "expected_strategies": ("fixed", "sentence", "section"),
            "expected_retrieval_methods": (
                "keyword",
                "semantic",
                "hybrid",
            ),
            "expected_chunk_sizes": (256, 512, 1024),
            "expected_reranker_configs": (
                (False, None),
                (True, 10),
                (True, 20),
                (True, 40),
            ),
        }

        self.assertTrue(
            is_complete_parameter_sweep(
                top_k_values=[3, 5, 10],
                **common_arguments,
            )
        )
        self.assertFalse(
            is_complete_parameter_sweep(
                top_k_values=[3],
                **common_arguments,
            )
        )
        self.assertFalse(
            is_complete_parameter_sweep(
                top_k_values=[3, 5, 10],
                **{
                    **common_arguments,
                    "reranker_configs": [
                        (False, None),
                        (True, 10),
                        (True, 20),
                    ],
                },
            )
        )

    def test_reranker_configs_include_off_once_and_each_enabled_pool(self):
        self.assertEqual(
            build_reranker_configs(
                reranker_enabled_values=[False, True],
                reranker_pool_sizes=[10, 20, 40],
            ),
            [
                (False, None),
                (True, 10),
                (True, 20),
                (True, 40),
            ],
        )

    def test_answer_accuracy_is_the_primary_selection_metric(self):
        scores = {}
        high_retrieval_config = (
            256,
            "fixed",
            "semantic",
            3,
            False,
            None,
        )
        high_accuracy_config = (
            512,
            "section",
            "hybrid",
            5,
            True,
            20,
        )

        _record(
            scores,
            high_retrieval_config,
            correct=False,
            all_evidence_hit=1.0,
            evidence_recall=1.0,
            reciprocal_rank=1.0,
            precision=1.0,
        )
        _record(
            scores,
            high_accuracy_config,
            correct=True,
            all_evidence_hit=0.0,
            evidence_recall=0.0,
            reciprocal_rank=0.0,
            precision=0.0,
        )

        best = select_best_parameter_config(scores)

        self.assertEqual(best["strategy"], "section")
        self.assertEqual(best["retrieval_method"], "hybrid")
        self.assertEqual(best["selection_metrics"]["accuracy"], 1.0)
        self.assertTrue(best["reranker_enabled"])
        self.assertEqual(best["reranker_pool_size"], 20)

    def test_retrieval_metrics_and_smaller_top_k_break_accuracy_ties(self):
        scores = {}
        top_k_three = (512, "section", "hybrid", 3, False, None)
        top_k_five = (512, "section", "hybrid", 5, False, None)

        for config in (top_k_three, top_k_five):
            _record(
                scores,
                config,
                correct=True,
                within_one_percent=True,
                all_evidence_hit=1.0,
                evidence_recall=1.0,
                reciprocal_rank=1.0,
                precision=0.5,
            )

        best = select_best_parameter_config(scores)

        self.assertEqual(best["top_k"], 3)

    def test_equal_metrics_prefer_no_reranker_then_smaller_pool(self):
        scores = {}
        configs = (
            (512, "section", "hybrid", 5, True, 40),
            (512, "section", "hybrid", 5, True, 10),
            (512, "section", "hybrid", 5, False, None),
        )
        for config in configs:
            _record(
                scores,
                config,
                correct=True,
                within_one_percent=True,
                all_evidence_hit=1.0,
                evidence_recall=1.0,
                reciprocal_rank=1.0,
                precision=1.0,
            )

        best = select_best_parameter_config(scores)

        self.assertFalse(best["reranker_enabled"])
        self.assertIsNone(best["reranker_pool_size"])

        enabled_scores = {}
        for pool_size in (40, 10):
            _record(
                enabled_scores,
                (512, "section", "hybrid", 5, True, pool_size),
                correct=True,
                within_one_percent=True,
                all_evidence_hit=1.0,
                evidence_recall=1.0,
                reciprocal_rank=1.0,
                precision=1.0,
            )

        enabled_best = select_best_parameter_config(enabled_scores)
        self.assertEqual(enabled_best["reranker_pool_size"], 10)

    def test_no_results_has_no_best_configuration(self):
        self.assertIsNone(select_best_parameter_config({}))


if __name__ == "__main__":
    unittest.main()
