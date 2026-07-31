import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.statistical_analysis import (  # noqa: E402
    analyze_paired_binary_outcomes,
    cochrans_q,
    exact_mcnemar,
    holm_adjust,
    wilson_score_interval,
)


class StatisticalAnalysisTests(unittest.TestCase):
    def test_cochrans_q_known_result(self):
        rows = [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
        ]

        result = cochrans_q(rows)

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["statistic"], 12.0)
        self.assertEqual(result["degrees_of_freedom"], 2)
        self.assertAlmostEqual(result["p_value"], 0.0024787521766663585)
        self.assertTrue(result["reject_null"])

    def test_cochrans_q_reports_no_within_question_variation(self):
        result = cochrans_q(
            [
                [True, True, True],
                [False, False, False],
            ]
        )

        self.assertEqual(result["status"], "not_testable")
        self.assertEqual(result["reason"], "no_within_question_variation")
        self.assertIsNone(result["p_value"])
        self.assertFalse(result["reject_null"])

    def test_exact_mcnemar_known_result(self):
        outcomes_a = [True] * 8 + [False] * 2
        outcomes_b = [False] * 8 + [True] * 2

        result = exact_mcnemar(outcomes_a, outcomes_b)

        self.assertEqual(result["a_correct_b_wrong"], 8)
        self.assertEqual(result["a_wrong_b_correct"], 2)
        self.assertEqual(result["discordant_pairs"], 10)
        self.assertAlmostEqual(result["accuracy_difference"], 0.6)
        self.assertAlmostEqual(result["raw_p_value"], 0.109375)

    def test_exact_mcnemar_boundary_avoids_misleading_wald_interval(self):
        result = exact_mcnemar([True] * 10, [False] * 10)

        self.assertEqual(result["accuracy_difference"], 1.0)
        self.assertAlmostEqual(result["raw_p_value"], 0.001953125)
        self.assertNotIn("accuracy_difference_confidence_interval", result)

    def test_exact_mcnemar_zero_discordance_has_p_value_one(self):
        result = exact_mcnemar(
            [True, False, True],
            [True, False, True],
        )

        self.assertEqual(result["discordant_pairs"], 0)
        self.assertEqual(result["raw_p_value"], 1.0)

    def test_holm_adjustment_preserves_original_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_wilson_score_interval_known_values(self):
        halfway = wilson_score_interval(50, 100)
        none_correct = wilson_score_interval(0, 10)
        all_correct = wilson_score_interval(10, 10)

        self.assertAlmostEqual(halfway["lower"], 0.4038315303659956)
        self.assertAlmostEqual(halfway["upper"], 0.5961684696340044)
        self.assertAlmostEqual(none_correct["lower"], 0.0)
        self.assertAlmostEqual(none_correct["upper"], 0.2775327998628892)
        self.assertAlmostEqual(all_correct["lower"], 0.7224672001371107)
        self.assertAlmostEqual(all_correct["upper"], 1.0)

    def test_analysis_pairs_by_question_id_not_dictionary_order(self):
        outcomes = {
            "a": {"1": True, "2": False, "3": True},
            "b": {"3": False, "1": True, "2": True},
        }

        result = analyze_paired_binary_outcomes(
            outcomes,
            primary_system="a",
            analysis_role="test",
        )
        comparison = result["pairwise"]["comparisons"][0]

        self.assertEqual(comparison["both_correct"], 1)
        self.assertEqual(comparison["a_correct_b_wrong"], 1)
        self.assertEqual(comparison["a_wrong_b_correct"], 1)

    def test_analysis_rejects_mismatched_question_sets(self):
        with self.assertRaisesRegex(ValueError, "same question ids"):
            analyze_paired_binary_outcomes(
                {
                    "a": {"1": True, "2": False},
                    "b": {"1": True},
                },
                analysis_role="test",
            )

    def test_twelve_systems_produce_all_66_pairs(self):
        outcomes = {
            f"config_{system_index}": {
                str(question_index): question_index < 20 - system_index
                for question_index in range(20)
            }
            for system_index in range(12)
        }

        result = analyze_paired_binary_outcomes(
            outcomes,
            primary_system="config_0",
            analysis_role="exploratory_parameter_diagnostics",
        )

        self.assertEqual(result["systems_compared"], 12)
        self.assertEqual(result["pairwise"]["scope"], "all_pairs")
        self.assertEqual(result["pairwise"]["family_size"], 66)
        self.assertEqual(len(result["pairwise"]["comparisons"]), 66)
        self.assertTrue(result["omnibus"]["reject_null"])

    def test_specified_comparisons_define_holm_family(self):
        outcomes = {
            "optimized": {"1": True, "2": True, "3": False},
            "baseline_a": {"1": False, "2": True, "3": False},
            "baseline_b": {"1": False, "2": False, "3": True},
        }

        result = analyze_paired_binary_outcomes(
            outcomes,
            primary_system="optimized",
            comparisons=[
                ("optimized", "baseline_a"),
                ("optimized", "baseline_b"),
            ],
            analysis_role="confirmatory_method_comparison",
        )

        self.assertEqual(result["pairwise"]["scope"], "specified_pairs")
        self.assertEqual(result["pairwise"]["family_size"], 2)
        self.assertTrue(
            all(
                comparison["involves_primary"]
                for comparison in result["pairwise"]["comparisons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
