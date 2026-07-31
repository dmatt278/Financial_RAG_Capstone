import os
from typing import Any
from app.data.data_loader import (
    BASELINE_SPLIT,
    DEV_SPLIT,
    TRAIN_SPLIT,
    TUNING_SPLIT,
    get_finqa_gold_evidence,
    iter_docfinqa_examples,
)
from app.evaluation import (
    align_docfinqa_evidence,
    evaluate_docfinqa_answer_metrics,
    evaluate_retrieval,
)
from app.rag.generator import (
    DEFAULT_OPENAI_MODEL,
    generate_answer,
    generate_no_context_answer,
    limit_chunks_to_prompt_budget,
)
from app.rag.math_agent import build_math_program_prompt, math_agent
from app.rag.parameter_selection import (
    RETRIEVAL_SCORE_WEIGHTS,
    build_reranker_configs,
    is_complete_parameter_sweep,
    record_parameter_result,
    record_retrieval_result,
    rank_parameter_configs,
    rank_retrieval_parameter_configs,
)
from app.rag.retriever import get_all_chunks, get_top_k_chunks
from app.rag.vector_store import DEFAULT_CHUNK_SIZES
from app.statistical_analysis import analyze_paired_binary_outcomes, holm_adjust
from app.results_logger import (
    complete_experiment_run,
    create_results_table,
    get_best_rag_parameters,
    get_retrieval_shortlist,
    log_question_result,
    log_question_results,
    save_best_rag_parameters,
    save_retrieval_shortlist,
    save_statistical_analysis,
    start_experiment_run,
)


OPTIMIZED_RAG_CONFIG = {
    "retrieval_method": os.getenv("OPTIMIZED_RETRIEVAL_METHOD", "hybrid"),
    "reranker_enabled": (
        os.getenv("OPTIMIZED_RERANKER_ENABLED", "true").lower() == "true"
    ),
    "strategy": os.getenv("OPTIMIZED_CHUNK_STRATEGY", "section"),
    "chunk_size": int(os.getenv("OPTIMIZED_CHUNK_SIZE", "512")),
    "top_k": int(os.getenv("OPTIMIZED_TOP_K", "5")),
    "reranker_pool_size": int(
        os.getenv("OPTIMIZED_RERANKER_POOL_SIZE", "20")
    ),
}

RESULT_LOG_BATCH_SIZE = 500
FULL_RAG_RESULT_LOG_BATCH_SIZE = 10
FULL_RAG_TOP_K_VALUES = (3, 5, 10)
FULL_RAG_STRATEGIES = ("fixed", "sentence", "section")
FULL_RAG_RETRIEVAL_METHODS = ("keyword", "semantic", "hybrid")
FULL_RAG_RERANKER_ENABLED_VALUES = (False, True)
FULL_RAG_RERANKER_POOL_SIZES = (10, 20, 40)
FULL_RAG_RERANKER_CONFIGS = (
    (False, None),
    (True, 10),
    (True, 20),
    (True, 40),
)
RETRIEVAL_SWEEP_EXPERIMENT = "top_chunks_evidence_sweep"
DEV_SHORTLIST_EXPERIMENT = "full_rag_dev_shortlist_sweep"
RETRIEVAL_SHORTLIST_SIZE = 12
DEV_FINALIST_COUNT = 3
STAGE_2_ANALYSIS_NAME = "stage2_parameter_comparison"
STAGE_3_ANALYSIS_NAME = "stage3_baseline_comparison"
MATH_AGENT_EXPERIMENT = "full_rag_math_tool_agent"
MATH_AGENT_ANALYSIS_NAME = "math_agent_vs_direct_llm"


def _parameter_config_key(config):
    reranker_enabled = bool(config["reranker_enabled"])
    return (
        int(config["chunk_size"]),
        str(config["strategy"]),
        str(config["retrieval_method"]),
        int(config["top_k"]),
        reranker_enabled,
        (
            int(config["reranker_pool_size"])
            if reranker_enabled
            else None
        ),
    )


def _parameter_config_from_candidate(candidate):
    return {
        "retrieval_method": candidate["retrieval_method"],
        "strategy": candidate["strategy"],
        "chunk_size": candidate["chunk_size"],
        "top_k": candidate["top_k"],
        "reranker_enabled": candidate["reranker_enabled"],
        "reranker_pool_size": candidate.get("reranker_pool_size"),
    }


def _parameter_config_id(config):
    """Returns a stable readable id for one complete parameter configuration."""

    reranker_label = (
        f"pool_{config['reranker_pool_size']}"
        if config["reranker_enabled"]
        else "off"
    )
    return "__".join(
        (
            str(config["retrieval_method"]),
            str(config["strategy"]),
            f"chunk_{config['chunk_size']}",
            f"top_{config['top_k']}",
            f"reranker_{reranker_label}",
        )
    )


def _get_docfinqa_evidence_alignment(example, split, where):
    gold_evidence = get_finqa_gold_evidence(
        split=split,
        question=example["question"],
        gold_answer=example["gold_answer"],
    )
    all_question_chunks = get_all_chunks(where=where)
    return align_docfinqa_evidence(
        gold_evidence=gold_evidence,
        all_question_chunks=all_question_chunks,
    )


def _compact_retrieved_chunks(chunks):
    """Keeps retrieval diagnostics without duplicating Chroma text in PostgreSQL."""

    return [
        {
            key: value
            for key, value in chunk.items()
            if key != "text"
        }
        for chunk in chunks
    ]


def _context_was_truncated(source_chunks, generation_chunks):
    return (
        len(source_chunks) != len(generation_chunks)
        or any(
            str(source.get("text", ""))
            != str(generated.get("text", ""))
            for source, generated in zip(source_chunks, generation_chunks)
        )
    )


def full_rag(
    split: str = TUNING_SPLIT,
    index: int = 0,
    top_k: int = 3,
    strategy: str = "fixed",
    chunk_size: int = 512,
    retrieval_method: str = "semantic",
    log_result: bool = True,
) -> dict[str, Any]:
    """
    Runs full pipeline without math agent. Did not add financebench yet.
    Gets chunks, generates an answer, evaluates it, and logs the result.
    """

    example = next(
        iter_docfinqa_examples(split=split, start_index=index, limit=1),
        None,
    )

    if example is None:
        raise IndexError(f"Index {index} is out of range for split '{split}'.")

    where = {
        "$and": [
            {"document_id": {"$eq": example["document_id"]}},
            {"strategy": {"$eq": strategy}},
            {"chunk_size": {"$eq": chunk_size}},
        ]
    }

    retrieved_chunks = get_top_k_chunks(
        question=example["question"],
        top_k=top_k,
        where=where,
        retrieval_method=retrieval_method,
    )
    evidence_alignment = _get_docfinqa_evidence_alignment(
        example=example,
        split=split,
        where=where,
    )
    retrieval_metrics = evaluate_retrieval(
        chunks=retrieved_chunks,
        k=top_k,
        evidence_alignment=evidence_alignment,
    )
    answer = generate_answer(
        question=example["question"],
        retrieved_chunks=retrieved_chunks,
    )
    generation_metrics = evaluate_docfinqa_answer_metrics(
        generated_answer=answer,
        gold_answer=example["gold_answer"],
    )
    generation_metrics.update(
        {
            "answer_source": "openai",
            "model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        }
    )
    is_correct = generation_metrics["docfinqa_answer_correct"]

    result = {
        "experiment_name": "full_rag",
        "dataset": "docfinqa",
        "split": split,
        "question_id": example["question_id"],
        "question": example["question"],
        "gold_answer": example["gold_answer"],
        "generated_answer": answer,
        "is_correct": is_correct,
        "retrieval_method": retrieval_method,
        "chunk_strategy": strategy,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "reranker_used": False,
        "retrieved_chunk_ids": [chunk["id"] for chunk in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "sources": [
            {
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "preview": chunk["text"][:500],
                "metadata": chunk["metadata"],
            }
            for chunk in retrieved_chunks
        ],
    }

    if log_result:
        result["result_id"] = log_question_result(result)

    return result


def full_rag_parameter_sweep(
    split: str = TUNING_SPLIT,
    start_index: int = 0,
    limit: int | None = None,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    reranker_enabled_values: list[bool] | None = None,
    reranker_pool_sizes: list[int] | None = None,
    chunk_size: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
    candidate_configs: list[dict[str, Any]] | None = None,
    experiment_name: str = "full_rag_parameter_sweep",
    winner_source_split: str | None = None,
):
    """Runs answer generation and evaluation for every selected configuration."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required before starting the full RAG sweep."
        )

    candidate_config_keys = None
    if candidate_configs is not None:
        if not candidate_configs:
            raise ValueError("At least one candidate configuration is required.")
        candidate_config_keys = {
            _parameter_config_key(config)
            for config in candidate_configs
        }
        if len(candidate_config_keys) != len(candidate_configs):
            raise ValueError("Candidate configurations must be unique.")

        chunk_sizes = sorted({key[0] for key in candidate_config_keys})
        strategies = sorted({key[1] for key in candidate_config_keys})
        retrieval_methods = sorted({key[2] for key in candidate_config_keys})
        top_k_values = sorted({key[3] for key in candidate_config_keys})
        reranker_enabled_values = sorted(
            {key[4] for key in candidate_config_keys}
        )
        reranker_pool_sizes = sorted(
            {
                key[5]
                for key in candidate_config_keys
                if key[4] and key[5] is not None
            }
        )
    else:
        top_k_values = top_k_values or list(FULL_RAG_TOP_K_VALUES)
        strategies = strategies or list(FULL_RAG_STRATEGIES)
        retrieval_methods = retrieval_methods or list(
            FULL_RAG_RETRIEVAL_METHODS
        )
        reranker_enabled_values = reranker_enabled_values or list(
            FULL_RAG_RERANKER_ENABLED_VALUES
        )
        reranker_pool_sizes = reranker_pool_sizes or list(
            FULL_RAG_RERANKER_POOL_SIZES
        )
        chunk_sizes = (
            [chunk_size]
            if chunk_size is not None
            else list(DEFAULT_CHUNK_SIZES)
        )

    if any(pool_size <= 0 for pool_size in reranker_pool_sizes):
        raise ValueError("Reranker pool sizes must be positive integers.")
    if (
        True in reranker_enabled_values
        and max(top_k_values) > min(reranker_pool_sizes)
    ):
        raise ValueError(
            "Every reranker pool size must be greater than or equal to top_k."
        )

    reranker_configs = build_reranker_configs(
        reranker_enabled_values=reranker_enabled_values,
        reranker_pool_sizes=reranker_pool_sizes,
    )
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    complete_parameter_sweep = (
        candidate_config_keys is None
        and is_complete_parameter_sweep(
            top_k_values=top_k_values,
            strategies=strategies,
            retrieval_methods=retrieval_methods,
            chunk_sizes=chunk_sizes,
            reranker_configs=reranker_configs,
            expected_top_k_values=FULL_RAG_TOP_K_VALUES,
            expected_strategies=FULL_RAG_STRATEGIES,
            expected_retrieval_methods=FULL_RAG_RETRIEVAL_METHODS,
            expected_chunk_sizes=DEFAULT_CHUNK_SIZES,
            expected_reranker_configs=FULL_RAG_RERANKER_CONFIGS,
        )
    )

    results = []
    pending_results = []
    parameter_scores = {}
    parameter_outcomes = {}
    questions_processed = 0
    rows_processed = 0
    rows_saved = 0
    run_id = None

    if log_results:
        create_results_table()
        run_id = start_experiment_run(
            dataset="docfinqa",
            experiment_name=experiment_name,
            split=split,
            model=model,
        )

    def flush_pending_results():
        nonlocal rows_saved

        if not pending_results:
            return

        result_ids = log_question_results(
            pending_results,
            ensure_table=False,
        )
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id

        rows_saved += len(result_ids)
        pending_results.clear()

    for example in iter_docfinqa_examples(
        split=split,
        start_index=start_index,
        limit=limit,
    ):
        questions_processed += 1

        for current_chunk_size in chunk_sizes:
            for strategy in strategies:
                where = {
                    "$and": [
                        {"document_id": {"$eq": example["document_id"]}},
                        {"strategy": {"$eq": strategy}},
                        {"chunk_size": {"$eq": current_chunk_size}},
                    ]
                }
                evidence_alignment = _get_docfinqa_evidence_alignment(
                    example=example,
                    split=split,
                    where=where,
                )

                for retrieval_method in retrieval_methods:
                    for (
                        reranker_enabled,
                        reranker_pool_size,
                    ) in reranker_configs:
                        selected_top_k_values = [
                            top_k
                            for top_k in top_k_values
                            if (
                                candidate_config_keys is None
                                or (
                                    current_chunk_size,
                                    strategy,
                                    retrieval_method,
                                    top_k,
                                    reranker_enabled,
                                    reranker_pool_size,
                                )
                                in candidate_config_keys
                            )
                        ]
                        if not selected_top_k_values:
                            continue

                        ranked_chunks = get_top_k_chunks(
                            question=example["question"],
                            top_k=max(selected_top_k_values),
                            where=where,
                            retrieval_method=retrieval_method,
                            reranker_enabled=reranker_enabled,
                            reranker_pool_size=(
                                reranker_pool_size
                                or OPTIMIZED_RAG_CONFIG[
                                    "reranker_pool_size"
                                ]
                            ),
                        )

                        for top_k in selected_top_k_values:
                            chunks = [
                                dict(chunk)
                                for chunk in ranked_chunks[:top_k]
                            ]
                            retrieval_metrics = evaluate_retrieval(
                                chunks=chunks,
                                k=top_k,
                                evidence_alignment=evidence_alignment,
                            )
                            retrieval_metrics["reranker_pool_size"] = (
                                reranker_pool_size
                                if reranker_enabled
                                else None
                            )
                            generation_chunks = limit_chunks_to_prompt_budget(
                                question=example["question"],
                                retrieved_chunks=chunks,
                                model=model,
                            )
                            answer = generate_answer(
                                question=example["question"],
                                retrieved_chunks=generation_chunks,
                            )
                            generation_metrics = (
                                evaluate_docfinqa_answer_metrics(
                                    generated_answer=answer,
                                    gold_answer=example["gold_answer"],
                                )
                            )
                            generation_metrics.update(
                                {
                                    "answer_source": "openai",
                                    "model": model,
                                    "generation_context_chunk_count": len(
                                        generation_chunks
                                    ),
                                    "generation_context_chunk_ids": [
                                        chunk["id"]
                                        for chunk in generation_chunks
                                    ],
                                    "prompt_was_truncated": (
                                        len(generation_chunks) < len(chunks)
                                        or any(
                                            len(
                                                str(
                                                    limited.get("text", "")
                                                )
                                            )
                                            < len(
                                                str(
                                                    retrieved.get("text", "")
                                                )
                                            )
                                            for limited, retrieved in zip(
                                                generation_chunks,
                                                chunks,
                                            )
                                        )
                                    ),
                                }
                            )
                            is_correct = generation_metrics[
                                "docfinqa_answer_correct"
                            ]
                            if type(is_correct) is not bool:
                                raise TypeError(
                                    "docfinqa_answer_correct must be Boolean."
                                )
                            config_key = (
                                current_chunk_size,
                                strategy,
                                retrieval_method,
                                top_k,
                                reranker_enabled,
                                reranker_pool_size,
                            )
                            record_parameter_result(
                                parameter_scores=parameter_scores,
                                config_key=config_key,
                                is_correct=is_correct,
                                retrieval_metrics=retrieval_metrics,
                                generation_metrics=generation_metrics,
                            )
                            question_id = str(example["question_id"])
                            config_outcomes = parameter_outcomes.setdefault(
                                config_key,
                                {},
                            )
                            if question_id in config_outcomes:
                                raise RuntimeError(
                                    "Duplicate question outcome encountered for "
                                    f"configuration {config_key}: {question_id}"
                                )
                            config_outcomes[question_id] = is_correct

                            result = {
                                "run_id": run_id,
                                "experiment_name": experiment_name,
                                "dataset": "docfinqa",
                                "split": split,
                                "question_id": example["question_id"],
                                "question": example["question"],
                                "gold_answer": example["gold_answer"],
                                "generated_answer": answer,
                                "is_correct": is_correct,
                                "retrieval_method": retrieval_method,
                                "chunk_strategy": strategy,
                                "chunk_size": current_chunk_size,
                                "top_k": top_k,
                                "reranker_used": reranker_enabled,
                                "reranker_pool_size": (
                                    reranker_pool_size
                                    if reranker_enabled
                                    else None
                                ),
                                "retrieved_chunk_ids": [
                                    chunk["id"]
                                    for chunk in chunks
                                ],
                                "retrieved_chunks": (
                                    _compact_retrieved_chunks(chunks)
                                ),
                                "retrieval_metrics": retrieval_metrics,
                                "generation_metrics": generation_metrics,
                            }
                            rows_processed += 1

                            if return_results:
                                results.append(result)

                            if log_results:
                                pending_results.append(result)
                                if (
                                    len(pending_results)
                                    >= FULL_RAG_RESULT_LOG_BATCH_SIZE
                                ):
                                    flush_pending_results()

    if log_results:
        flush_pending_results()

    ranked_candidates = rank_parameter_configs(parameter_scores)
    best_candidate = ranked_candidates[0] if ranked_candidates else None
    statistical_analysis = None
    if candidate_config_keys is not None and len(ranked_candidates) >= 2:
        outcomes_by_system = {}
        system_metadata = {}

        for rank, candidate in enumerate(ranked_candidates, start=1):
            config = _parameter_config_from_candidate(candidate)
            config_key = _parameter_config_key(config)
            system_id = _parameter_config_id(config)
            outcomes_by_system[system_id] = parameter_outcomes[config_key]
            system_metadata[system_id] = {
                "dev_accuracy_rank": rank,
                **config,
            }

        primary_system = _parameter_config_id(
            _parameter_config_from_candidate(best_candidate)
        )
        statistical_analysis = analyze_paired_binary_outcomes(
            outcomes_by_system,
            system_metadata=system_metadata,
            primary_system=primary_system,
            analysis_role="exploratory_parameter_diagnostics",
        )
        statistical_analysis.update(
            {
                "parameter_selection_metric": (
                    "docfinqa_answer_correct_accuracy"
                ),
                "statistics_used_to_select_winner": False,
                "selection_note": (
                    "The winner is selected by the pre-registered accuracy "
                    "rule and deterministic tie-breakers. These dev-set "
                    "tests are supporting post-selection diagnostics."
                ),
            }
        )

    complete_candidate_outcomes = (
        candidate_config_keys is not None
        and len(ranked_candidates) == len(candidate_config_keys)
        and len(parameter_outcomes) == len(candidate_config_keys)
        and all(
            len(outcomes) == questions_processed
            for outcomes in parameter_outcomes.values()
        )
    )

    best_parameters = None
    best_parameter_metrics = None
    best_parameters_saved = False
    best_parameters_save_reason = None
    statistical_analysis_saved = False

    if best_candidate is None:
        best_parameters_save_reason = "no_results"
    else:
        best_parameter_metrics = best_candidate["selection_metrics"]
        best_parameters = {
            key: value
            for key, value in best_candidate.items()
            if key != "selection_metrics"
        }

        if not log_results:
            best_parameters_save_reason = "result_logging_disabled"
        elif start_index != 0 or limit is not None:
            best_parameters_save_reason = "partial_dataset_run"
        elif winner_source_split is not None:
            if split != winner_source_split:
                best_parameters_save_reason = "wrong_winner_source_split"
            elif (
                candidate_config_keys is None
                or len(candidate_config_keys) != RETRIEVAL_SHORTLIST_SIZE
            ):
                best_parameters_save_reason = "incomplete_shortlist"
            elif not complete_candidate_outcomes:
                best_parameters_save_reason = "incomplete_shortlist_outcomes"
            elif statistical_analysis is None:
                best_parameters_save_reason = "statistical_analysis_unavailable"
            else:
                save_statistical_analysis(
                    run_id=run_id,
                    dataset="docfinqa",
                    model=model,
                    source_experiment=experiment_name,
                    source_split=split,
                    analysis_name=STAGE_2_ANALYSIS_NAME,
                    analysis=statistical_analysis,
                )
                statistical_analysis_saved = True
                save_best_rag_parameters(
                    run_id=run_id,
                    dataset="docfinqa",
                    model=model,
                    source_experiment=experiment_name,
                    source_split=split,
                    config=best_parameters,
                    selection_metrics=best_parameter_metrics,
                )
                best_parameters_saved = True
        elif split != TUNING_SPLIT:
            best_parameters_save_reason = "not_tuning_split"
        elif not complete_parameter_sweep:
            best_parameters_save_reason = "partial_parameter_sweep"
        else:
            save_best_rag_parameters(
                run_id=run_id,
                dataset="docfinqa",
                model=model,
                source_experiment=experiment_name,
                source_split=split,
                config=best_parameters,
                selection_metrics=best_parameter_metrics,
            )
            best_parameters_saved = True

    statistical_analysis_save_reason = None
    if statistical_analysis is None:
        statistical_analysis_save_reason = "not_available"
    elif not statistical_analysis_saved:
        statistical_analysis_save_reason = best_parameters_save_reason

    is_full_run = bool(
        best_parameters_saved
        and (
            statistical_analysis_saved
            if winner_source_split is not None
            else complete_parameter_sweep
        )
    )
    if log_results:
        complete_experiment_run(
            run_id=run_id,
            questions_processed=questions_processed,
            rows_saved=rows_saved,
            is_full_run=is_full_run,
        )

    summary = {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "experiment_name": experiment_name,
        "dataset": "docfinqa",
        "split": split,
        "model": model,
        "start_index": start_index,
        "requested_limit": limit,
        "questions_processed": questions_processed,
        "chunk_sizes": chunk_sizes,
        "strategies": strategies,
        "retrieval_methods": retrieval_methods,
        "top_k_values": top_k_values,
        "reranker_enabled_values": reranker_enabled_values,
        "reranker_pool_sizes": reranker_pool_sizes,
        "reranker_configurations": [
            {
                "reranker_enabled": enabled,
                "reranker_pool_size": pool_size,
            }
            for enabled, pool_size in reranker_configs
        ],
        "parameter_combinations_per_question": (
            len(candidate_config_keys)
            if candidate_config_keys is not None
            else (
                len(chunk_sizes)
                * len(strategies)
                * len(retrieval_methods)
                * len(top_k_values)
                * len(reranker_configs)
            )
        ),
        "uses_candidate_shortlist": candidate_config_keys is not None,
        "complete_candidate_outcomes": complete_candidate_outcomes,
        "complete_parameter_sweep": complete_parameter_sweep,
        "rows_processed": rows_processed,
        "rows_saved": rows_saved,
        "results_returned": len(results),
        "best_parameters": best_parameters,
        "best_parameter_metrics": best_parameter_metrics,
        "best_parameters_saved": best_parameters_saved,
        "best_parameters_save_reason": best_parameters_save_reason,
        "top_generation_candidates": ranked_candidates[
            :DEV_FINALIST_COUNT
        ],
        "statistical_analysis": statistical_analysis,
        "statistical_analysis_saved": statistical_analysis_saved,
        "statistical_analysis_save_reason": (
            statistical_analysis_save_reason
        ),
    }
    if return_results:
        summary["results"] = results

    return summary


def full_rag_shortlist_sweep(
    start_index: int = 0,
    limit: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
):
    """Runs the saved train retrieval finalists over the complete dev split."""

    saved_shortlist = get_retrieval_shortlist(
        dataset="docfinqa",
        source_experiment=RETRIEVAL_SWEEP_EXPERIMENT,
        source_split=TRAIN_SPLIT,
    )
    if saved_shortlist is None:
        raise RuntimeError(
            "No completed train retrieval shortlist was found. "
            "Run POST /run-chunk-rag first."
        )

    candidates = saved_shortlist.get("candidates") or []
    if len(candidates) != RETRIEVAL_SHORTLIST_SIZE:
        raise RuntimeError(
            "The saved retrieval shortlist is incomplete: expected "
            f"{RETRIEVAL_SHORTLIST_SIZE} candidates, found {len(candidates)}."
        )

    candidate_configs = [
        _parameter_config_from_candidate(candidate)
        for candidate in candidates
    ]
    candidate_keys = {
        _parameter_config_key(config)
        for config in candidate_configs
    }
    if len(candidate_keys) != RETRIEVAL_SHORTLIST_SIZE:
        raise RuntimeError(
            "The saved retrieval shortlist contains duplicate configurations."
        )

    summary = full_rag_parameter_sweep(
        split=DEV_SPLIT,
        start_index=start_index,
        limit=limit,
        log_results=log_results,
        return_results=return_results,
        candidate_configs=candidate_configs,
        experiment_name=DEV_SHORTLIST_EXPERIMENT,
        winner_source_split=DEV_SPLIT,
    )
    summary["retrieval_shortlist_source"] = {
        "experiment_name": saved_shortlist["source_experiment"],
        "split": saved_shortlist["source_split"],
        "questions_run": saved_shortlist["questions_run"],
        "updated_at": saved_shortlist["updated_at"],
    }
    return summary


def top_chunks(
    split: str = TRAIN_SPLIT,
    start_index: int = 0,
    limit: int | None = None,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    reranker_enabled_values: list[bool] | None = None,
    reranker_pool_sizes: list[int] | None = None,
    chunk_size: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
):
    """
    Runs the retrieval-only train sweep and saves its top 12 configurations.
    """

    top_k_values = top_k_values or list(FULL_RAG_TOP_K_VALUES)
    strategies = strategies or list(FULL_RAG_STRATEGIES)
    retrieval_methods = retrieval_methods or list(
        FULL_RAG_RETRIEVAL_METHODS
    )
    reranker_enabled_values = reranker_enabled_values or list(
        FULL_RAG_RERANKER_ENABLED_VALUES
    )
    reranker_pool_sizes = reranker_pool_sizes or list(
        FULL_RAG_RERANKER_POOL_SIZES
    )
    if any(pool_size <= 0 for pool_size in reranker_pool_sizes):
        raise ValueError("Reranker pool sizes must be positive integers.")
    if (
        True in reranker_enabled_values
        and max(top_k_values) > min(reranker_pool_sizes)
    ):
        raise ValueError(
            "Every reranker pool size must be greater than or equal to top_k."
        )

    reranker_configs = build_reranker_configs(
        reranker_enabled_values=reranker_enabled_values,
        reranker_pool_sizes=reranker_pool_sizes,
    )
    chunk_sizes = (
        [chunk_size]
        if chunk_size is not None
        else list(DEFAULT_CHUNK_SIZES)
    )
    complete_parameter_sweep = is_complete_parameter_sweep(
        top_k_values=top_k_values,
        strategies=strategies,
        retrieval_methods=retrieval_methods,
        chunk_sizes=chunk_sizes,
        reranker_configs=reranker_configs,
        expected_top_k_values=FULL_RAG_TOP_K_VALUES,
        expected_strategies=FULL_RAG_STRATEGIES,
        expected_retrieval_methods=FULL_RAG_RETRIEVAL_METHODS,
        expected_chunk_sizes=DEFAULT_CHUNK_SIZES,
        expected_reranker_configs=FULL_RAG_RERANKER_CONFIGS,
    )

    results = []
    pending_results = []
    parameter_scores = {}
    questions_processed = 0
    rows_processed = 0
    rows_saved = 0
    run_id = None

    if log_results:
        create_results_table()
        run_id = start_experiment_run(
            dataset="docfinqa",
            experiment_name=RETRIEVAL_SWEEP_EXPERIMENT,
            split=split,
            model=None,
        )

    def flush_pending_results():
        nonlocal rows_saved

        if not pending_results:
            return

        result_ids = log_question_results(
            pending_results,
            ensure_table=False,
        )
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id

        rows_saved += len(result_ids)
        pending_results.clear()

    for example in iter_docfinqa_examples(split=split, start_index=start_index, limit=limit):
        questions_processed += 1

        for current_chunk_size in chunk_sizes:
            for strategy in strategies:
                where = {
                    "$and": [
                        {"document_id": {"$eq": example["document_id"]}},
                        {"strategy": {"$eq": strategy}},
                        {"chunk_size": {"$eq": current_chunk_size}},
                    ]
                }
                evidence_alignment = _get_docfinqa_evidence_alignment(
                    example=example,
                    split=split,
                    where=where,
                )

                for retrieval_method in retrieval_methods:
                    for (
                        reranker_enabled,
                        reranker_pool_size,
                    ) in reranker_configs:
                        ranked_chunks = get_top_k_chunks(
                            question=example["question"],
                            top_k=max(top_k_values),
                            where=where,
                            retrieval_method=retrieval_method,
                            reranker_enabled=reranker_enabled,
                            reranker_pool_size=(
                                reranker_pool_size
                                or OPTIMIZED_RAG_CONFIG[
                                    "reranker_pool_size"
                                ]
                            ),
                        )

                        for top_k in top_k_values:
                            chunks = [
                                dict(chunk)
                                for chunk in ranked_chunks[:top_k]
                            ]
                            retrieval_metrics = evaluate_retrieval(
                                chunks=chunks,
                                k=top_k,
                                evidence_alignment=evidence_alignment,
                            )
                            retrieval_metrics["reranker_pool_size"] = (
                                reranker_pool_size
                                if reranker_enabled
                                else None
                            )
                            record_retrieval_result(
                                parameter_scores=parameter_scores,
                                config_key=(
                                    current_chunk_size,
                                    strategy,
                                    retrieval_method,
                                    top_k,
                                    reranker_enabled,
                                    reranker_pool_size,
                                ),
                                retrieval_metrics=retrieval_metrics,
                            )

                            result = {
                                "run_id": run_id,
                                "experiment_name": (
                                    RETRIEVAL_SWEEP_EXPERIMENT
                                ),
                                "dataset": "docfinqa",
                                "split": split,
                                "question_id": example["question_id"],
                                "question": example["question"],
                                "gold_answer": example["gold_answer"],
                                "generated_answer": None,
                                "is_correct": None,
                                "retrieval_method": retrieval_method,
                                "chunk_strategy": strategy,
                                "chunk_size": current_chunk_size,
                                "top_k": top_k,
                                "reranker_used": reranker_enabled,
                                "reranker_pool_size": (
                                    reranker_pool_size
                                    if reranker_enabled
                                    else None
                                ),
                                "retrieved_chunk_ids": [
                                    chunk["id"]
                                    for chunk in chunks
                                ],
                                "retrieved_chunks": (
                                    _compact_retrieved_chunks(chunks)
                                ),
                                "retrieval_metrics": retrieval_metrics,
                                "generation_metrics": {},
                            }
                            rows_processed += 1

                            if return_results:
                                results.append(result)

                            if log_results:
                                pending_results.append(result)
                                if (
                                    len(pending_results)
                                    >= RESULT_LOG_BATCH_SIZE
                                ):
                                    flush_pending_results()

    if log_results:
        flush_pending_results()

    top_candidates = rank_retrieval_parameter_configs(
        parameter_scores,
        limit=RETRIEVAL_SHORTLIST_SIZE,
    )
    shortlist_saved = False
    shortlist_save_reason = None

    if not top_candidates:
        shortlist_save_reason = "no_results"
    elif not log_results:
        shortlist_save_reason = "result_logging_disabled"
    elif split != TRAIN_SPLIT:
        shortlist_save_reason = "not_train_split"
    elif start_index != 0 or limit is not None:
        shortlist_save_reason = "partial_dataset_run"
    elif not complete_parameter_sweep:
        shortlist_save_reason = "partial_parameter_sweep"
    elif len(top_candidates) != RETRIEVAL_SHORTLIST_SIZE:
        shortlist_save_reason = "incomplete_shortlist"
    elif not all(
        candidate["selection_metrics"]["retrieval_metrics_complete"]
        for candidate in top_candidates
    ):
        shortlist_save_reason = "incomplete_retrieval_metrics"
    else:
        save_retrieval_shortlist(
            run_id=run_id,
            dataset="docfinqa",
            source_experiment=RETRIEVAL_SWEEP_EXPERIMENT,
            source_split=split,
            questions_run=questions_processed,
            ranking_rule={
                "shortlist_size": RETRIEVAL_SHORTLIST_SIZE,
                "metric_weights": RETRIEVAL_SCORE_WEIGHTS,
                "tie_breakers": [
                    "avg_all_evidence_hit_at_k",
                    "avg_evidence_recall_at_k",
                    "avg_reciprocal_rank",
                    "avg_precision_at_k",
                    "prefer_no_reranker",
                    "smaller_reranker_pool_size",
                    "smaller_top_k",
                    "smaller_chunk_size",
                ],
            },
            candidates=top_candidates,
        )
        shortlist_saved = True

    is_full_run = shortlist_saved
    if log_results:
        complete_experiment_run(
            run_id=run_id,
            questions_processed=questions_processed,
            rows_saved=rows_saved,
            is_full_run=is_full_run,
        )

    summary = {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "experiment_name": RETRIEVAL_SWEEP_EXPERIMENT,
        "dataset": "docfinqa",
        "split": split,
        "start_index": start_index,
        "requested_limit": limit,
        "questions_processed": questions_processed,
        "chunk_sizes": chunk_sizes,
        "strategies": strategies,
        "retrieval_methods": retrieval_methods,
        "top_k_values": top_k_values,
        "reranker_enabled_values": reranker_enabled_values,
        "reranker_pool_sizes": reranker_pool_sizes,
        "reranker_configurations": [
            {
                "reranker_enabled": enabled,
                "reranker_pool_size": pool_size,
            }
            for enabled, pool_size in reranker_configs
        ],
        "parameter_combinations_per_question": (
            len(chunk_sizes)
            * len(strategies)
            * len(retrieval_methods)
            * len(top_k_values)
            * len(reranker_configs)
        ),
        "complete_parameter_sweep": complete_parameter_sweep,
        "rows_processed": rows_processed,
        "rows_saved": rows_saved,
        "results_returned": len(results),
        "retrieval_score_weights": RETRIEVAL_SCORE_WEIGHTS,
        "top_retrieval_candidates": top_candidates,
        "shortlist_saved": shortlist_saved,
        "shortlist_save_reason": shortlist_save_reason,
    }
    if return_results:
        summary["results"] = results

    return summary


def full_rag_with_math_agent(
    split: str = TUNING_SPLIT,
    start_index: int = 0,
    limit: int | None = 15,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    chunk_size: int = 512,
    log_results: bool = True,
    return_results: bool = False,
):
    """
    Compares direct LLM math with an LLM-planned, Python-executed math agent.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required before starting the math-agent sweep."
        )

    top_k_values = top_k_values or [3, 5, 10]
    strategies = strategies or ["fixed", "sentence", "section"]
    retrieval_methods = retrieval_methods or ["keyword", "semantic", "hybrid"]
    if start_index < 0:
        raise ValueError("start_index cannot be negative.")
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("Every top_k value must be greater than zero.")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    results = []
    pending_results = []
    outcomes_by_config = {}
    questions_processed = 0
    rows_processed = 0
    rows_saved = 0
    run_id = None

    def flush_pending_results():
        nonlocal rows_saved
        if not pending_results:
            return
        result_ids = log_question_results(
            pending_results,
            ensure_table=False,
        )
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id
        rows_saved += len(pending_results)
        pending_results.clear()

    if log_results:
        create_results_table()
        run_id = start_experiment_run(
            dataset="docfinqa",
            experiment_name=MATH_AGENT_EXPERIMENT,
            split=split,
            model=model,
            parameters={
                "start_index": start_index,
                "limit": limit,
                "top_k_values": top_k_values,
                "strategies": strategies,
                "retrieval_methods": retrieval_methods,
                "chunk_size": chunk_size,
                "comparison_methods": ["direct_llm", "math_agent"],
                "reranker_enabled": False,
            },
        )

    for example in iter_docfinqa_examples(
        split=split,
        start_index=start_index,
        limit=limit,
    ):
        questions_processed += 1
        for strategy in strategies:
            where = {
                "$and": [
                    {"document_id": {"$eq": example["document_id"]}},
                    {"strategy": {"$eq": strategy}},
                    {"chunk_size": {"$eq": chunk_size}},
                ]
            }
            evidence_alignment = _get_docfinqa_evidence_alignment(
                example=example,
                split=split,
                where=where,
            )

            for retrieval_method in retrieval_methods:
                for top_k in top_k_values:
                    chunks = get_top_k_chunks(
                        question=example["question"],
                        top_k=top_k,
                        where=where,
                        retrieval_method=retrieval_method,
                    )
                    retrieval_metrics = evaluate_retrieval(
                        chunks=chunks,
                        k=top_k,
                        evidence_alignment=evidence_alignment,
                    )

                    # Use one context that fits both prompts so the only
                    # experimental difference is how the math is performed.
                    shared_chunks = limit_chunks_to_prompt_budget(
                        question=example["question"],
                        retrieved_chunks=chunks,
                        model=model,
                        prompt_builder=build_math_program_prompt,
                    )
                    shared_chunks = limit_chunks_to_prompt_budget(
                        question=example["question"],
                        retrieved_chunks=shared_chunks,
                        model=model,
                    )
                    shared_context_was_truncated = _context_was_truncated(
                        chunks,
                        shared_chunks,
                    )
                    shared_context_chunk_ids = [
                        chunk["id"] for chunk in shared_chunks
                    ]

                    direct_context_metrics = {}
                    direct_answer = generate_answer(
                        question=example["question"],
                        retrieved_chunks=shared_chunks,
                        generation_context_metrics=direct_context_metrics,
                    )
                    direct_metrics = evaluate_docfinqa_answer_metrics(
                        generated_answer=direct_answer,
                        gold_answer=example["gold_answer"],
                    )
                    direct_metrics.update(
                        {
                            "answer_source": "direct_llm_calculation",
                            "comparison_method": "direct_llm",
                            "model": model,
                            **direct_context_metrics,
                            "shared_context_chunk_ids": (
                                shared_context_chunk_ids
                            ),
                            "source_context_chunk_count": len(chunks),
                            "source_context_character_count": sum(
                                len(str(chunk.get("text", "")))
                                for chunk in chunks
                            ),
                            "prompt_was_truncated": (
                                shared_context_was_truncated
                                or direct_context_metrics.get(
                                    "prompt_was_truncated",
                                    False,
                                )
                            ),
                        }
                    )
                    direct_is_correct = direct_metrics[
                        "docfinqa_answer_correct"
                    ]
                    if type(direct_is_correct) is not bool:
                        raise TypeError(
                            "docfinqa_answer_correct must be Boolean."
                        )

                    agent_result = math_agent(
                        question=example["question"],
                        chunks=shared_chunks,
                        dataset="docfinqa",
                    )
                    agent_answer = agent_result["answer"]
                    agent_metrics = evaluate_docfinqa_answer_metrics(
                        generated_answer=agent_answer,
                        gold_answer=example["gold_answer"],
                    )
                    agent_metrics.update(
                        {
                            "answer_source": "llm_program_python_executor",
                            "comparison_method": "math_agent",
                            "model": agent_result["model"],
                            "shared_context_chunk_ids": (
                                shared_context_chunk_ids
                            ),
                            "math_prompt_version": agent_result[
                                "prompt_version"
                            ],
                            "math_agent_status": agent_result["status"],
                            "math_program": agent_result["program"],
                            "raw_math_answer": agent_result["raw_answer"],
                            "raw_program_response": agent_result[
                                "raw_model_output"
                            ],
                            "program_parse_succeeded": agent_result[
                                "program_parse_succeeded"
                            ],
                            "execution_succeeded": agent_result[
                                "execution_succeeded"
                            ],
                            "execution_error": agent_result["error"],
                            "execution_steps": agent_result[
                                "execution_steps"
                            ],
                            "literal_operand_count": agent_result[
                                "literal_operand_count"
                            ],
                            "grounded_operand_count": agent_result[
                                "grounded_operand_count"
                            ],
                            "operand_grounding_rate": agent_result[
                                "operand_grounding_rate"
                            ],
                            "ungrounded_operands": agent_result[
                                "ungrounded_operands"
                            ],
                            "generation_context_chunk_ids": agent_result[
                                "generation_context_chunk_ids"
                            ],
                            "generation_context_chunk_count": agent_result[
                                "generation_context_chunk_count"
                            ],
                            "prompt_was_truncated": agent_result[
                                "prompt_was_truncated"
                            ] or shared_context_was_truncated,
                            "source_context_chunk_count": len(chunks),
                            "source_context_character_count": sum(
                                len(str(chunk.get("text", "")))
                                for chunk in chunks
                            ),
                            "generation_context_character_count": sum(
                                len(str(chunk.get("text", "")))
                                for chunk in shared_chunks
                            ),
                        }
                    )
                    agent_is_correct = agent_metrics[
                        "docfinqa_answer_correct"
                    ]
                    if type(agent_is_correct) is not bool:
                        raise TypeError(
                            "docfinqa_answer_correct must be Boolean."
                        )

                    config = {
                        "retrieval_method": retrieval_method,
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "top_k": top_k,
                        "reranker_enabled": False,
                        "reranker_pool_size": None,
                    }
                    config_id = _parameter_config_id(config)
                    config_outcomes = outcomes_by_config.setdefault(
                        config_id,
                        {
                            "config": config,
                            "direct_llm": {},
                            "math_agent": {},
                        },
                    )
                    question_id = str(example["question_id"])
                    for method, is_correct in (
                        ("direct_llm", direct_is_correct),
                        ("math_agent", agent_is_correct),
                    ):
                        if question_id in config_outcomes[method]:
                            raise RuntimeError(
                                "Duplicate math comparison outcome for "
                                f"{config_id}, {method}, question "
                                f"{question_id}."
                            )
                        config_outcomes[method][question_id] = is_correct

                    common_result = {
                        "run_id": run_id,
                        "experiment_name": MATH_AGENT_EXPERIMENT,
                        "dataset": "docfinqa",
                        "split": split,
                        "question_id": example["question_id"],
                        "question": example["question"],
                        "gold_answer": example["gold_answer"],
                        "retrieval_method": retrieval_method,
                        "chunk_strategy": strategy,
                        "chunk_size": chunk_size,
                        "top_k": top_k,
                        "reranker_used": False,
                        "retrieved_chunk_ids": [chunk["id"] for chunk in chunks],
                        "retrieved_chunks": _compact_retrieved_chunks(chunks),
                        "retrieval_metrics": retrieval_metrics,
                    }
                    comparison_results = [
                        {
                            **common_result,
                            "generated_answer": direct_answer,
                            "is_correct": direct_is_correct,
                            "generation_metrics": direct_metrics,
                        },
                        {
                            **common_result,
                            "generated_answer": agent_answer,
                            "is_correct": agent_is_correct,
                            "generation_metrics": agent_metrics,
                        },
                    ]
                    rows_processed += len(comparison_results)

                    if return_results:
                        results.extend(comparison_results)
                    if log_results:
                        pending_results.extend(comparison_results)
                        if (
                            len(pending_results)
                            >= FULL_RAG_RESULT_LOG_BATCH_SIZE
                        ):
                            flush_pending_results()

    if log_results:
        flush_pending_results()

    statistical_analyses = []
    for config_id, config_outcomes in outcomes_by_config.items():
        direct_outcomes = config_outcomes["direct_llm"]
        agent_outcomes = config_outcomes["math_agent"]
        if (
            not direct_outcomes
            or set(direct_outcomes) != set(agent_outcomes)
        ):
            continue

        analysis = analyze_paired_binary_outcomes(
            {
                "direct_llm": direct_outcomes,
                "math_agent": agent_outcomes,
            },
            system_metadata={
                "direct_llm": {
                    "method": "direct_llm_calculation",
                    **config_outcomes["config"],
                },
                "math_agent": {
                    "method": "llm_program_python_executor",
                    **config_outcomes["config"],
                },
            },
            primary_system="math_agent",
            comparisons=[("math_agent", "direct_llm")],
            analysis_role="paired_math_execution_comparison",
        )
        analysis.update(
            {
                "config_id": config_id,
                "config": config_outcomes["config"],
                "statistics_used_to_select_rag_parameters": False,
                "comparison_note": (
                    "Both methods received the same effective chunks. This "
                    "experiment is separate from baseline parameter selection."
                ),
            }
        )
        statistical_analyses.append(analysis)

    raw_p_values = [
        analysis["pairwise"]["comparisons"][0]["raw_p_value"]
        for analysis in statistical_analyses
    ]
    for analysis, adjusted_p_value in zip(
        statistical_analyses,
        holm_adjust(raw_p_values),
    ):
        comparison = analysis["pairwise"]["comparisons"][0]
        comparison["holm_adjusted_across_configurations"] = (
            adjusted_p_value
        )
        comparison["reject_null_across_configurations"] = (
            adjusted_p_value <= analysis["alpha"]
        )
        analysis["across_configuration_correction"] = {
            "method": "holm",
            "family_size": len(statistical_analyses),
        }

    complete_paired_outcomes = bool(outcomes_by_config) and all(
        len(config_outcomes["direct_llm"]) == questions_processed
        and len(config_outcomes["math_agent"]) == questions_processed
        for config_outcomes in outcomes_by_config.values()
    )
    is_full_run = bool(
        start_index == 0
        and limit is None
        and questions_processed > 0
        and complete_paired_outcomes
    )
    statistical_analyses_saved = 0
    if log_results and is_full_run:
        for analysis in statistical_analyses:
            save_statistical_analysis(
                run_id=run_id,
                dataset="docfinqa",
                model=model,
                source_experiment=MATH_AGENT_EXPERIMENT,
                source_split=split,
                analysis_name=(
                    f"{MATH_AGENT_ANALYSIS_NAME}__{analysis['config_id']}"
                ),
                analysis=analysis,
            )
            statistical_analyses_saved += 1

    if log_results:
        complete_experiment_run(
            run_id=run_id,
            questions_processed=questions_processed,
            rows_saved=rows_saved,
            is_full_run=is_full_run,
        )

    summary = {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "experiment_name": MATH_AGENT_EXPERIMENT,
        "dataset": "docfinqa",
        "split": split,
        "model": model,
        "start_index": start_index,
        "requested_limit": limit,
        "questions_processed": questions_processed,
        "configuration_count": len(outcomes_by_config),
        "comparison_methods": ["direct_llm", "math_agent"],
        "rows_processed": rows_processed,
        "rows_saved": rows_saved,
        "statistical_analyses": statistical_analyses,
        "statistical_analyses_saved": statistical_analyses_saved,
        "statistical_analysis_save_reason": (
            None
            if statistical_analyses_saved
            else (
                "result_logging_disabled"
                if not log_results
                else "full_dataset_run_required"
            )
        ),
        "results_returned": len(results),
    }
    if return_results:
        summary["results"] = results

    return summary


def get_baseline_results():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required before starting the baseline runs."
        )

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    saved_optimized_parameters = get_best_rag_parameters(
        dataset="docfinqa",
        model=model,
        source_split=DEV_SPLIT,
    )
    if saved_optimized_parameters is None:
        raise RuntimeError(
            "No completed dev shortlist winner was found for "
            f"model '{model}'. Run POST /run-full-rag first."
        )
    if (
        saved_optimized_parameters.get("source_experiment")
        != DEV_SHORTLIST_EXPERIMENT
    ):
        raise RuntimeError(
            "The saved dev parameters did not come from the completed "
            "12-candidate dev shortlist sweep. Run POST /run-full-rag first."
        )

    create_results_table()
    run_id = start_experiment_run(
        dataset="docfinqa",
        experiment_name="baseline",
        split=BASELINE_SPLIT,
        model=model,
    )
    rows_saved = 0

    reranker_enabled = saved_optimized_parameters["reranker_used"]
    config = {
        "retrieval_method": saved_optimized_parameters[
            "retrieval_method"
        ],
        "reranker_enabled": reranker_enabled,
        "strategy": saved_optimized_parameters["chunk_strategy"],
        "chunk_size": saved_optimized_parameters["chunk_size"],
        "top_k": saved_optimized_parameters["top_k"],
        "reranker_pool_size": (
            (
                saved_optimized_parameters["reranker_pool_size"]
                or OPTIMIZED_RAG_CONFIG["reranker_pool_size"]
            )
            if reranker_enabled
            else None
        ),
    }
    optimized_config_source = "saved_dev_shortlist_winner"

    method_outcomes = {
        "no_context": {},
        "full_document": {},
        "naive_rag": {},
        "optimized_rag": {},
    }

    def record_method_outcome(method_name, question_id, is_correct):
        if type(is_correct) is not bool:
            raise TypeError("docfinqa_answer_correct must be Boolean.")
        question_id = str(question_id)
        outcomes = method_outcomes[method_name]
        if question_id in outcomes:
            raise RuntimeError(
                f"Duplicate {method_name} outcome for question {question_id}."
            )
        outcomes[question_id] = is_correct

    def get_generation_metrics(answer, gold_answer, baseline_name):
        metrics = evaluate_docfinqa_answer_metrics(
            generated_answer=answer,
            gold_answer=gold_answer,
        )
        metrics.update(
            {
                "baseline": baseline_name,
                "answer_source": "openai",
                "model": model,
            }
        )
        return metrics

    #get metrics on each of these
    #no context just question
    for example in iter_docfinqa_examples(split=BASELINE_SPLIT):
        answer = generate_no_context_answer(example["question"])
        generation_metrics = get_generation_metrics(
            answer,
            example["gold_answer"],
            "no_context",
        )
        is_correct = generation_metrics["docfinqa_answer_correct"]
        record_method_outcome(
            "no_context",
            example["question_id"],
            is_correct,
        )

        log_question_result({
            "run_id": run_id,
            "experiment_name": "baseline",
            "dataset": "docfinqa",
            "split": BASELINE_SPLIT,
            "question_id": example["question_id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": None,
            "chunk_strategy": None,
            "chunk_size": None,
            "top_k": None,
            "reranker_used": False,
            "retrieved_chunk_ids": [],
            "retrieved_chunks": [],
            "retrieval_metrics": {},
            "generation_metrics": generation_metrics,
        })
        rows_saved += 1

    #full document and question
    for example in iter_docfinqa_examples(split=BASELINE_SPLIT):
        full_document_context = [{
            "chunk_id": "full_document",
            "text": example["document_text"],
            "metadata": {"question_id": example["question_id"]},
        }]
        full_document_context_metrics = {}
        answer = generate_answer(
            example["question"],
            full_document_context,
            generation_context_metrics=full_document_context_metrics,
        )
        generation_metrics = get_generation_metrics(
            answer,
            example["gold_answer"],
            "full_document",
        )
        generation_metrics.update(full_document_context_metrics)
        is_correct = generation_metrics["docfinqa_answer_correct"]
        record_method_outcome(
            "full_document",
            example["question_id"],
            is_correct,
        )

        log_question_result({
            "run_id": run_id,
            "experiment_name": "baseline",
            "dataset": "docfinqa",
            "split": BASELINE_SPLIT,
            "question_id": example["question_id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": None,
            "chunk_strategy": None,
            "chunk_size": None,
            "top_k": None,
            "reranker_used": False,
            "retrieved_chunk_ids": [],
            "retrieved_chunks": [],
            "retrieval_metrics": {},
            "generation_metrics": generation_metrics,
        })
        rows_saved += 1
    
    #navie RAG
    for example in iter_docfinqa_examples(split=BASELINE_SPLIT):
        where = {
            "$and": [
                {"document_id": {"$eq": example["document_id"]}},
                {"strategy": {"$eq": "fixed"}},
                {"chunk_size": {"$eq": 512}},
            ]
        }

        best_chunks = get_top_k_chunks(
            question=example["question"],
            top_k=3,
            where=where,
            retrieval_method="semantic"
        )
        evidence_alignment = _get_docfinqa_evidence_alignment(
            example=example,
            split=BASELINE_SPLIT,
            where=where,
        )
        retrieval_metrics = evaluate_retrieval(
            chunks=best_chunks,
            k=3,
            evidence_alignment=evidence_alignment,
        )
        answer = generate_answer(example["question"], best_chunks)
        generation_metrics = get_generation_metrics(
            answer,
            example["gold_answer"],
            "naive_rag",
        )
        is_correct = generation_metrics["docfinqa_answer_correct"]
        record_method_outcome(
            "naive_rag",
            example["question_id"],
            is_correct,
        )

        log_question_result({
            "run_id": run_id,
            "experiment_name": "baseline",
            "dataset": "docfinqa",
            "split": BASELINE_SPLIT,
            "question_id": example["question_id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": "semantic",
            "chunk_strategy": "fixed",
            "chunk_size": 512,
            "top_k": 3,
            "reranker_used": False,
            "retrieved_chunk_ids": [chunk["id"] for chunk in best_chunks],
            "retrieved_chunks": best_chunks,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
        })
        rows_saved += 1

    #optimized RAG
    for example in iter_docfinqa_examples(split=BASELINE_SPLIT):
        where = {
            "$and": [
                {"document_id": {"$eq": example["document_id"]}},
                {"strategy": {"$eq": config["strategy"]}},
                {"chunk_size": {"$eq": config["chunk_size"]}},
            ]
        }

        best_chunks = get_top_k_chunks(
            question=example["question"],
            top_k=config["top_k"],
            where=where,
            retrieval_method=config["retrieval_method"],
            reranker_enabled=config["reranker_enabled"],
            reranker_pool_size=config["reranker_pool_size"],
        )
        evidence_alignment = _get_docfinqa_evidence_alignment(
            example=example,
            split=BASELINE_SPLIT,
            where=where,
        )
        retrieval_metrics = evaluate_retrieval(
            chunks=best_chunks,
            k=config["top_k"],
            evidence_alignment=evidence_alignment,
        )
        retrieval_metrics["reranker_pool_size"] = (
            config["reranker_pool_size"]
            if config["reranker_enabled"]
            else None
        )
        answer = generate_answer(example["question"], best_chunks)
        generation_metrics = get_generation_metrics(
            answer,
            example["gold_answer"],
            "optimized_rag",
        )
        generation_metrics.update(
            {
                "optimized_config_source": optimized_config_source,
                "optimized_config_source_split": (
                    saved_optimized_parameters["source_split"]
                    if saved_optimized_parameters
                    else None
                ),
                "optimized_config_selection_metrics": (
                    saved_optimized_parameters["selection_metrics"]
                    if saved_optimized_parameters
                    else None
                ),
            }
        )
        is_correct = generation_metrics["docfinqa_answer_correct"]
        record_method_outcome(
            "optimized_rag",
            example["question_id"],
            is_correct,
        )

        log_question_result({
            "run_id": run_id,
            "experiment_name": "baseline",
            "dataset": "docfinqa",
            "split": BASELINE_SPLIT,
            "question_id": example["question_id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": config["retrieval_method"],
            "chunk_strategy": config["strategy"],
            "chunk_size": config["chunk_size"],
            "top_k": config["top_k"],
            "reranker_used": config["reranker_enabled"],
            "reranker_pool_size": (
                config["reranker_pool_size"]
                if config["reranker_enabled"]
                else None
            ),
            "retrieved_chunk_ids": [chunk["id"] for chunk in best_chunks],
            "retrieved_chunks": best_chunks,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
        })
        rows_saved += 1

    statistical_analysis = analyze_paired_binary_outcomes(
        method_outcomes,
        system_metadata={
            "no_context": {
                "method": "question_only_openai_baseline",
            },
            "full_document": {
                "method": "full_document_openai_baseline",
            },
            "naive_rag": {
                "method": "naive_rag_baseline",
                "retrieval_method": "semantic",
                "strategy": "fixed",
                "chunk_size": 512,
                "top_k": 3,
                "reranker_enabled": False,
                "reranker_pool_size": None,
            },
            "optimized_rag": {
                "method": "optimized_rag",
                **config,
                "parameters_selected_on": DEV_SPLIT,
            },
        },
        primary_system="optimized_rag",
        comparisons=[
            ("optimized_rag", "no_context"),
            ("optimized_rag", "full_document"),
            ("optimized_rag", "naive_rag"),
        ],
        analysis_role="confirmatory_method_comparison",
    )
    statistical_analysis.update(
        {
            "optimized_parameters_frozen_before_test": True,
            "optimized_config_source": optimized_config_source,
            "optimized_config_source_split": (
                saved_optimized_parameters["source_split"]
            ),
        }
    )
    save_statistical_analysis(
        run_id=run_id,
        dataset="docfinqa",
        model=model,
        source_experiment="baseline",
        source_split=BASELINE_SPLIT,
        analysis_name=STAGE_3_ANALYSIS_NAME,
        analysis=statistical_analysis,
    )

    questions_processed = len(method_outcomes["optimized_rag"])
    is_full_run = True
    complete_experiment_run(
        run_id=run_id,
        questions_processed=questions_processed,
        rows_saved=rows_saved,
        is_full_run=is_full_run,
    )

    return {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "experiment_name": "baseline",
        "dataset": "docfinqa",
        "split": BASELINE_SPLIT,
        "model": model,
        "optimized_parameters": config,
        "optimized_config_source": optimized_config_source,
        "statistical_analysis": statistical_analysis,
        "statistical_analysis_saved": True,
    }
