RETRIEVAL_SCORE_WEIGHTS = {
    "avg_all_evidence_hit_at_k": 0.50,
    "avg_evidence_recall_at_k": 0.30,
    "avg_reciprocal_rank": 0.10,
    "avg_precision_at_k": 0.10,
}


def is_complete_parameter_sweep(
    top_k_values,
    strategies,
    retrieval_methods,
    chunk_sizes,
    reranker_configs,
    expected_top_k_values,
    expected_strategies,
    expected_retrieval_methods,
    expected_chunk_sizes,
    expected_reranker_configs,
):
    """Checks that each configured parameter is tested exactly once."""

    def matches(values, expected_values):
        return (
            len(values) == len(expected_values)
            and set(values) == set(expected_values)
        )

    return (
        matches(top_k_values, expected_top_k_values)
        and matches(strategies, expected_strategies)
        and matches(retrieval_methods, expected_retrieval_methods)
        and matches(chunk_sizes, expected_chunk_sizes)
        and matches(reranker_configs, expected_reranker_configs)
    )


def build_reranker_configs(reranker_enabled_values, reranker_pool_sizes):
    """Builds one disabled config and one config per enabled pool size."""

    configs = []
    if False in reranker_enabled_values:
        configs.append((False, None))
    if True in reranker_enabled_values:
        configs.extend(
            (True, pool_size)
            for pool_size in reranker_pool_sizes
        )
    return configs


def _new_parameter_score():
    return {
        "questions_run": 0,
        "correct_count": 0,
        "within_one_percent_count": 0,
        "metric_totals": {},
        "metric_counts": {},
    }


def _record_retrieval_metrics(score, retrieval_metrics):
    for metric_name in (
        "all_evidence_hit_at_k",
        "evidence_recall_at_k",
        "reciprocal_rank",
        "precision_at_k",
    ):
        value = retrieval_metrics.get(metric_name)
        if value is None:
            continue

        score["metric_totals"][metric_name] = (
            score["metric_totals"].get(metric_name, 0.0) + float(value)
        )
        score["metric_counts"][metric_name] = (
            score["metric_counts"].get(metric_name, 0) + 1
        )


def record_retrieval_result(
    parameter_scores,
    config_key,
    retrieval_metrics,
):
    """Records one retrieval-only result for later weighted ranking."""

    score = parameter_scores.setdefault(config_key, _new_parameter_score())
    score["questions_run"] += 1
    _record_retrieval_metrics(score, retrieval_metrics)


def record_parameter_result(
    parameter_scores,
    config_key,
    is_correct,
    retrieval_metrics,
    generation_metrics,
):
    score = parameter_scores.setdefault(config_key, _new_parameter_score())
    score["questions_run"] += 1
    score["correct_count"] += int(bool(is_correct))
    score["within_one_percent_count"] += int(
        generation_metrics.get("within_one_percent") is True
    )
    _record_retrieval_metrics(score, retrieval_metrics)


def _build_parameter_candidates(parameter_scores, include_generation_metrics):
    candidates = []

    for config_key, score in parameter_scores.items():
        (
            chunk_size,
            strategy,
            retrieval_method,
            top_k,
            reranker_enabled,
            reranker_pool_size,
        ) = config_key
        questions_run = score["questions_run"]

        def average_metric(metric_name):
            count = score["metric_counts"].get(metric_name, 0)
            if count == 0:
                return None
            return score["metric_totals"][metric_name] / count

        metrics = {
            "questions_run": questions_run,
            "retrieval_metrics_complete": all(
                score["metric_counts"].get(metric_name, 0) == questions_run
                for metric_name in (
                    "all_evidence_hit_at_k",
                    "evidence_recall_at_k",
                    "reciprocal_rank",
                    "precision_at_k",
                )
            ),
            "avg_all_evidence_hit_at_k": average_metric(
                "all_evidence_hit_at_k"
            ),
            "avg_evidence_recall_at_k": average_metric(
                "evidence_recall_at_k"
            ),
            "avg_reciprocal_rank": average_metric("reciprocal_rank"),
            "avg_precision_at_k": average_metric("precision_at_k"),
        }
        if include_generation_metrics:
            metrics.update(
                {
                    "accuracy": (
                        score["correct_count"] / questions_run
                        if questions_run
                        else None
                    ),
                    "within_one_percent_rate": (
                        score["within_one_percent_count"] / questions_run
                        if questions_run
                        else None
                    ),
                }
            )

        candidates.append(
            {
                "retrieval_method": retrieval_method,
                "strategy": strategy,
                "chunk_size": chunk_size,
                "top_k": top_k,
                "reranker_enabled": reranker_enabled,
                "reranker_pool_size": reranker_pool_size,
                "selection_metrics": metrics,
            }
        )

    return candidates


def _descending_metric(candidate, metric_name):
    value = candidate["selection_metrics"].get(metric_name)
    return -(float(value) if value is not None else -1.0)


def rank_retrieval_parameter_configs(parameter_scores, limit=None):
    """Ranks retrieval configs with the pre-registered weighted score."""

    candidates = _build_parameter_candidates(
        parameter_scores,
        include_generation_metrics=False,
    )

    for candidate in candidates:
        metrics = candidate["selection_metrics"]
        metrics["retrieval_score"] = sum(
            (metrics.get(metric_name) or 0.0) * weight
            for metric_name, weight in RETRIEVAL_SCORE_WEIGHTS.items()
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            _descending_metric(candidate, "retrieval_score"),
            _descending_metric(candidate, "avg_all_evidence_hit_at_k"),
            _descending_metric(candidate, "avg_evidence_recall_at_k"),
            _descending_metric(candidate, "avg_reciprocal_rank"),
            _descending_metric(candidate, "avg_precision_at_k"),
            candidate["reranker_enabled"],
            candidate["reranker_pool_size"] or 0,
            candidate["top_k"],
            candidate["chunk_size"],
            candidate["retrieval_method"],
            candidate["strategy"],
        ),
    )

    return ranked if limit is None else ranked[:limit]


def rank_parameter_configs(parameter_scores):
    """Ranks generated-answer configs with accuracy as the primary metric."""

    candidates = _build_parameter_candidates(
        parameter_scores,
        include_generation_metrics=True,
    )

    return sorted(
        candidates,
        key=lambda candidate: (
            _descending_metric(candidate, "accuracy"),
            _descending_metric(candidate, "within_one_percent_rate"),
            _descending_metric(candidate, "avg_all_evidence_hit_at_k"),
            _descending_metric(candidate, "avg_evidence_recall_at_k"),
            _descending_metric(candidate, "avg_reciprocal_rank"),
            _descending_metric(candidate, "avg_precision_at_k"),
            candidate["reranker_enabled"],
            candidate["reranker_pool_size"] or 0,
            candidate["top_k"],
            candidate["chunk_size"],
            candidate["retrieval_method"],
            candidate["strategy"],
        ),
    )


def select_best_parameter_config(parameter_scores):
    """Selects the most accurate config with deterministic retrieval tie-breaks."""

    ranked = rank_parameter_configs(parameter_scores)
    return ranked[0] if ranked else None
