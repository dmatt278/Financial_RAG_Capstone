from itertools import combinations
from math import erfc, exp, lgamma, log, sqrt
from statistics import NormalDist
from typing import Any


DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE_LEVEL = 0.95


def wilson_score_interval(
    correct_count: int,
    total: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, float | str]:
    """Returns a Wilson score interval for a binary success proportion."""

    if total <= 0:
        raise ValueError("total must be greater than zero.")
    if correct_count < 0 or correct_count > total:
        raise ValueError("correct_count must be between zero and total.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")

    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    proportion = correct_count / total
    z_squared = z_value ** 2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z_value
        * sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total ** 2)
        )
        / denominator
    )

    return {
        "confidence_level": confidence_level,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
        "method": "wilson_score",
    }


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Computes the chi-square survival function for an integer df."""

    if statistic < 0:
        raise ValueError("statistic cannot be negative.")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    if statistic == 0:
        return 1.0

    half_statistic = statistic / 2
    target_shape = degrees_of_freedom / 2

    if degrees_of_freedom % 2 == 0:
        shape = 1.0
        survival = exp(-half_statistic)
    else:
        shape = 0.5
        survival = erfc(sqrt(half_statistic))

    while shape < target_shape:
        log_increment = (
            shape * log(half_statistic)
            - half_statistic
            - lgamma(shape + 1)
        )
        survival += exp(log_increment)
        shape += 1

    return min(1.0, max(0.0, survival))


def cochrans_q(
    outcome_rows: list[list[bool]],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Runs Cochran's Q over matched binary outcomes."""

    if not outcome_rows:
        raise ValueError("At least one matched question is required.")

    system_count = len(outcome_rows[0])
    if system_count < 2:
        raise ValueError("At least two systems are required.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    if any(len(row) != system_count for row in outcome_rows):
        raise ValueError("Every outcome row must contain every system.")
    if any(type(value) is not bool for row in outcome_rows for value in row):
        raise TypeError("Cochran's Q outcomes must be Boolean values.")

    column_totals = [
        sum(int(row[column_index]) for row in outcome_rows)
        for column_index in range(system_count)
    ]
    row_totals = [sum(int(value) for value in row) for row in outcome_rows]
    grand_total = sum(column_totals)
    denominator = (
        system_count * grand_total
        - sum(row_total ** 2 for row_total in row_totals)
    )
    degrees_of_freedom = system_count - 1

    if denominator == 0:
        return {
            "test": "cochrans_q",
            "status": "not_testable",
            "reason": "no_within_question_variation",
            "statistic": 0.0,
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": None,
            "alpha": alpha,
            "reject_null": False,
        }

    numerator = (system_count - 1) * (
        system_count * sum(total ** 2 for total in column_totals)
        - grand_total ** 2
    )
    statistic = numerator / denominator
    p_value = _chi_square_survival(statistic, degrees_of_freedom)

    return {
        "test": "cochrans_q",
        "status": "ok",
        "reason": None,
        "statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "alpha": alpha,
        "reject_null": p_value <= alpha,
    }


def _exact_two_sided_binomial_p_value(successes: int, trials: int) -> float:
    """Exact two-sided Binomial(n, 0.5) p-value used by McNemar."""

    if trials == 0:
        return 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials.")

    lower_tail_end = min(successes, trials - successes)
    log_probabilities = [
        (
            lgamma(trials + 1)
            - lgamma(success_count + 1)
            - lgamma(trials - success_count + 1)
            - trials * log(2)
        )
        for success_count in range(lower_tail_end + 1)
    ]
    largest_log_probability = max(log_probabilities)
    lower_tail = exp(largest_log_probability) * sum(
        exp(value - largest_log_probability)
        for value in log_probabilities
    )
    return min(1.0, 2 * lower_tail)


def exact_mcnemar(
    outcomes_a: list[bool],
    outcomes_b: list[bool],
) -> dict[str, Any]:
    """Runs an exact paired McNemar test and reports its effect size."""

    if not outcomes_a or len(outcomes_a) != len(outcomes_b):
        raise ValueError("McNemar outcomes must be nonempty and paired.")
    if any(type(value) is not bool for value in outcomes_a + outcomes_b):
        raise TypeError("McNemar outcomes must be Boolean values.")

    both_correct = sum(a and b for a, b in zip(outcomes_a, outcomes_b))
    a_correct_b_wrong = sum(a and not b for a, b in zip(outcomes_a, outcomes_b))
    a_wrong_b_correct = sum(not a and b for a, b in zip(outcomes_a, outcomes_b))
    both_wrong = sum(not a and not b for a, b in zip(outcomes_a, outcomes_b))
    discordant_pairs = a_correct_b_wrong + a_wrong_b_correct
    p_value = _exact_two_sided_binomial_p_value(
        successes=a_correct_b_wrong,
        trials=discordant_pairs,
    )
    question_count = len(outcomes_a)
    accuracy_difference = (
        a_correct_b_wrong - a_wrong_b_correct
    ) / question_count

    return {
        "test": "exact_mcnemar",
        "questions_run": question_count,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant_pairs,
        "accuracy_difference": accuracy_difference,
        "raw_p_value": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Applies Holm's step-down family-wise error correction."""

    if any(p_value < 0 or p_value > 1 for p_value in p_values):
        raise ValueError("Every p-value must be between zero and one.")
    if not p_values:
        return []

    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_maximum = 0.0

    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = (len(p_values) - rank) * p_value
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)

    return adjusted


def analyze_paired_binary_outcomes(
    outcomes_by_system: dict[str, dict[str, bool]],
    *,
    system_metadata: dict[str, dict[str, Any]] | None = None,
    primary_system: str | None = None,
    comparisons: list[tuple[str, str]] | None = None,
    analysis_role: str,
    alpha: float = DEFAULT_ALPHA,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Builds a complete paired binary statistical analysis."""

    if len(outcomes_by_system) < 2:
        raise ValueError("At least two systems are required for comparison.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")

    system_ids = list(outcomes_by_system)
    if any(not isinstance(system_id, str) or not system_id for system_id in system_ids):
        raise ValueError("Every system must have a nonempty string id.")
    if primary_system is not None and primary_system not in outcomes_by_system:
        raise ValueError("primary_system must identify one of the systems.")

    first_outcomes = outcomes_by_system[system_ids[0]]
    if not first_outcomes:
        raise ValueError("At least one matched question is required.")
    question_ids = list(first_outcomes)
    expected_question_ids = set(question_ids)

    for system_id, outcomes in outcomes_by_system.items():
        if set(outcomes) != expected_question_ids:
            raise ValueError(
                f"System '{system_id}' does not contain the same question ids."
            )
        if any(type(value) is not bool for value in outcomes.values()):
            raise TypeError(
                f"System '{system_id}' contains a non-Boolean outcome."
            )

    metadata = system_metadata or {}
    outcome_lists = {
        system_id: [
            outcomes_by_system[system_id][question_id]
            for question_id in question_ids
        ]
        for system_id in system_ids
    }
    outcome_rows = [
        [outcome_lists[system_id][index] for system_id in system_ids]
        for index in range(len(question_ids))
    ]

    system_summaries = []
    for system_id in system_ids:
        correct_count = sum(outcome_lists[system_id])
        system_summaries.append(
            {
                "system_id": system_id,
                "is_primary": system_id == primary_system,
                "metadata": metadata.get(system_id, {}),
                "questions_run": len(question_ids),
                "correct_count": correct_count,
                "accuracy": correct_count / len(question_ids),
                "accuracy_confidence_interval": wilson_score_interval(
                    correct_count,
                    len(question_ids),
                    confidence_level,
                ),
            }
        )

    if comparisons is None:
        selected_comparisons = list(combinations(system_ids, 2))
        pairwise_scope = "all_pairs"
    else:
        selected_comparisons = list(comparisons)
        pairwise_scope = "specified_pairs"

    seen_comparisons = set()
    pairwise_results = []
    for system_a, system_b in selected_comparisons:
        if system_a not in outcomes_by_system or system_b not in outcomes_by_system:
            raise ValueError("Every comparison must reference known systems.")
        if system_a == system_b:
            raise ValueError("A system cannot be compared with itself.")
        comparison_key = frozenset((system_a, system_b))
        if comparison_key in seen_comparisons:
            raise ValueError("Pairwise comparisons must be unique.")
        seen_comparisons.add(comparison_key)

        result = exact_mcnemar(
            outcome_lists[system_a],
            outcome_lists[system_b],
        )
        result.update(
            {
                "system_a": system_a,
                "system_b": system_b,
                "involves_primary": (
                    primary_system is not None
                    and primary_system in (system_a, system_b)
                ),
            }
        )
        pairwise_results.append(result)

    adjusted_p_values = holm_adjust(
        [result["raw_p_value"] for result in pairwise_results]
    )
    omnibus = cochrans_q(outcome_rows, alpha=alpha)

    for result, adjusted_p_value in zip(pairwise_results, adjusted_p_values):
        result["holm_adjusted_p_value"] = adjusted_p_value
        result["reject_null_holm"] = adjusted_p_value <= alpha
        result["reject_after_omnibus_gate"] = (
            omnibus["reject_null"] and result["reject_null_holm"]
        )

    return {
        "analysis_role": analysis_role,
        "unit_of_analysis": "question",
        "document_clustering_applied": False,
        "alpha": alpha,
        "confidence_level": confidence_level,
        "questions_run": len(question_ids),
        "systems_compared": len(system_ids),
        "primary_system": primary_system,
        "systems": system_summaries,
        "omnibus": omnibus,
        "pairwise": {
            "test": "exact_mcnemar",
            "scope": pairwise_scope,
            "multiple_comparison_correction": "holm",
            "family_size": len(pairwise_results),
            "comparisons": pairwise_results,
        },
        "interpretation_note": (
            "Statistics describe paired question-level outcomes and do not "
            "change the configured parameter-selection rule. Questions from "
            "the same document are not cluster-adjusted."
        ),
    }
