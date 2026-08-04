import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import perf_counter
from typing import Any
from app.data.data_loader import (
    BASELINE_SPLIT,
    DEFAULT_SAMPLE_SEED,
    DEV_SPLIT,
    TRAIN_SPLIT,
    TUNING_SPLIT,
    get_finqa_gold_evidence,
    iter_docfinqa_examples_by_question_ids,
    iter_docfinqa_examples,
    iter_sampled_docfinqa_examples,
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
from app.rag.retriever import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    SEMANTIC_SEARCH_BACKEND,
    get_all_chunks,
    get_parameter_sweep_rankings,
    get_top_k_chunks,
)
from app.rag.vector_store import DEFAULT_CHUNK_SIZES
from app.statistical_analysis import analyze_paired_binary_outcomes, holm_adjust
from app.results_logger import (
    complete_experiment_run,
    create_results_table,
    get_best_rag_parameters,
    get_experiment_run,
    get_run_results_for_resume,
    get_retrieval_shortlist,
    log_question_result,
    log_question_results,
    resume_experiment_run,
    save_best_rag_parameters,
    save_retrieval_shortlist,
    save_statistical_analysis,
    start_experiment_run,
    update_experiment_run_progress,
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
FULL_RAG_RESULT_LOG_BATCH_SIZE = 100
DEFAULT_OPENAI_CONCURRENCY = 4
TRAIN_SAMPLE_SIZE = 50
DEV_SAMPLE_SIZE = 100
TEST_SAMPLE_SIZE = 100
MATH_SAMPLE_SIZE = 50
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
RETRIEVAL_SHORTLIST_SIZE = 3
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


def get_openai_concurrency() -> int:
    """Returns the bounded number of simultaneous generation requests."""

    raw_value = os.getenv(
        "OPENAI_CONCURRENCY",
        str(DEFAULT_OPENAI_CONCURRENCY),
    )
    try:
        concurrency = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "OPENAI_CONCURRENCY must be a positive integer."
        ) from exc
    if concurrency <= 0:
        raise RuntimeError(
            "OPENAI_CONCURRENCY must be a positive integer."
        )
    return concurrency


def _result_key(question_id, config=None, method="result"):
    config_id = _parameter_config_id(config) if config is not None else "none"
    return f"{question_id}::{config_id}::{method}"


def _config_from_result_row(row):
    return {
        "retrieval_method": row["retrieval_method"],
        "strategy": row["chunk_strategy"],
        "chunk_size": row["chunk_size"],
        "top_k": row["top_k"],
        "reranker_enabled": bool(row["reranker_used"]),
        "reranker_pool_size": (
            row["reranker_pool_size"] if row["reranker_used"] else None
        ),
    }


def _materialize_protocol_examples(
    *,
    split,
    sample_size,
    sample_seed,
    start_index,
    limit,
    resume_run_id,
):
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")
    if start_index < 0:
        raise ValueError("start_index cannot be negative.")
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative.")
    if resume_run_id and (start_index != 0 or limit is not None):
        raise ValueError(
            "A resumed run must use start_index=0 and limit=None."
        )

    if resume_run_id:
        saved_run = get_experiment_run(resume_run_id)
        if saved_run is None:
            raise ValueError(
                f"Experiment run '{resume_run_id}' was not found."
            )
        question_ids = saved_run.get("parameters", {}).get("question_ids")
        if not question_ids:
            raise ValueError(
                "The saved run does not contain a question checkpoint manifest."
            )
        examples = list(
            iter_docfinqa_examples_by_question_ids(
                split=split,
                question_ids=question_ids,
            )
        )
        return examples, [str(value) for value in question_ids]

    sampled_examples = list(
        iter_sampled_docfinqa_examples(
            split=split,
            sample_size=sample_size,
            seed=sample_seed,
        )
    )
    end_index = None if limit is None else start_index + limit
    examples = sampled_examples[start_index:end_index]
    return examples, [str(example["question_id"]) for example in examples]


def _run_bounded_generation(tasks, worker, handle_result, max_workers):
    """Runs a lazy task stream with bounded OpenAI concurrency."""

    pending = {}
    first_error = None

    def consume(return_when):
        nonlocal first_error
        if not pending:
            return
        completed, _ = wait(tuple(pending), return_when=return_when)
        for future in completed:
            task = pending.pop(future)
            try:
                output = future.result()
                handle_result(task, output)
            except Exception as exc:  # preserve other paid completions first
                if first_error is None:
                    first_error = exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        task_iterator = iter(tasks)
        while first_error is None:
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            except Exception as exc:
                first_error = exc
                break
            try:
                pending[executor.submit(worker, task)] = task
            except Exception as exc:
                first_error = exc
                break
            if len(pending) >= max_workers:
                consume(FIRST_COMPLETED)
        while pending:
            consume(FIRST_COMPLETED)

    if first_error is not None:
        raise first_error


def _get_docfinqa_evidence_alignment(example, split, where):
    gold_evidence = get_finqa_gold_evidence(
        split=split,
        question=example["question"],
        gold_answer=example["gold_answer"],
        gold_program=example.get("program"),
        document_text=example.get("document_text"),
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
            if key not in {"text", "embedding"}
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
    sample_size: int | None = DEV_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """Runs answer generation and evaluation for every selected configuration."""

    if resume_run_id and not log_results:
        raise ValueError("resume_run_id requires log_results=True.")
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
    openai_concurrency = get_openai_concurrency()
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

    examples, question_ids = _materialize_protocol_examples(
        split=split,
        sample_size=sample_size,
        sample_seed=sample_seed,
        start_index=start_index,
        limit=limit,
        resume_run_id=resume_run_id,
    )
    combinations_per_question = (
        len(candidate_config_keys)
        if candidate_config_keys is not None
        else (
            len(chunk_sizes)
            * len(strategies)
            * len(retrieval_methods)
            * len(top_k_values)
            * len(reranker_configs)
        )
    )
    expected_rows = len(examples) * combinations_per_question
    manifest_configs = (
        sorted(candidate_configs, key=_parameter_config_id)
        if candidate_configs is not None
        else None
    )
    protocol_parameters = {
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "sampling_method": "uniform_without_replacement_reservoir",
        "question_ids": question_ids,
        "candidate_configs": manifest_configs,
        "top_k_values": list(top_k_values),
        "strategies": list(strategies),
        "retrieval_methods": list(retrieval_methods),
        "chunk_sizes": list(chunk_sizes),
        "reranker_configurations": [list(value) for value in reranker_configs],
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "reranker_model_revision": DEFAULT_RERANKER_REVISION,
        "semantic_search_backend": SEMANTIC_SEARCH_BACKEND,
    }

    results = []
    pending_results = []
    parameter_scores = {}
    parameter_outcomes = {}
    rows_processed = 0
    rows_added_this_attempt = 0
    run_id = None
    existing_rows = []
    existing_keys = set()
    persisted_question_counts = {}

    if log_results:
        create_results_table()
        if resume_run_id:
            resume_experiment_run(
                run_id=resume_run_id,
                dataset="docfinqa",
                experiment_name=experiment_name,
                split=split,
                model=model,
                parameters=protocol_parameters,
                expected_rows=expected_rows,
            )
            run_id = resume_run_id
            existing_rows = get_run_results_for_resume(run_id)
        else:
            run_id = start_experiment_run(
                dataset="docfinqa",
                experiment_name=experiment_name,
                split=split,
                model=model,
                parameters=protocol_parameters,
                expected_rows=expected_rows,
            )

    for row in existing_rows:
        if not row.get("result_key"):
            raise RuntimeError(
                "This checkpoint predates resumable result keys and cannot "
                "be resumed safely."
            )
        config = _config_from_result_row(row)
        config_key = _parameter_config_key(config)
        is_correct = row["is_correct"]
        if type(is_correct) is not bool:
            raise TypeError("Saved docfinqa_answer_correct must be Boolean.")
        existing_keys.add(row["result_key"])
        record_parameter_result(
            parameter_scores=parameter_scores,
            config_key=config_key,
            is_correct=is_correct,
            retrieval_metrics=row["retrieval_metrics"],
            generation_metrics=row["generation_metrics"],
        )
        parameter_outcomes.setdefault(config_key, {})[
            str(row["question_id"])
        ] = is_correct
        question_id = str(row["question_id"])
        persisted_question_counts[question_id] = (
            persisted_question_counts.get(question_id, 0) + 1
        )

    rows_saved = len(existing_keys)

    def flush_pending_results():
        nonlocal rows_saved, rows_added_this_attempt
        if not pending_results:
            return
        result_ids = log_question_results(pending_results, ensure_table=False)
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id
            existing_keys.add(result["result_key"])
            question_id = str(result["question_id"])
            persisted_question_counts[question_id] = (
                persisted_question_counts.get(question_id, 0) + 1
            )
        rows_saved += len(result_ids)
        rows_added_this_attempt += len(result_ids)
        pending_results.clear()
        checkpointed_questions = sum(
            count == combinations_per_question
            for count in persisted_question_counts.values()
        )
        update_experiment_run_progress(
            run_id=run_id,
            questions_processed=checkpointed_questions,
            rows_saved=rows_saved,
        )
        print(
            f"{experiment_name}: {checkpointed_questions}/{len(examples)} "
            "questions checkpointed",
            flush=True,
        )

    def task_stream():
        for example in examples:
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
                        for reranker_enabled, reranker_pool_size in reranker_configs:
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
                            missing_configs = []
                            for top_k in selected_top_k_values:
                                config = {
                                    "chunk_size": current_chunk_size,
                                    "strategy": strategy,
                                    "retrieval_method": retrieval_method,
                                    "top_k": top_k,
                                    "reranker_enabled": reranker_enabled,
                                    "reranker_pool_size": (
                                        reranker_pool_size
                                        if reranker_enabled
                                        else None
                                    ),
                                }
                                result_key = _result_key(
                                    example["question_id"],
                                    config,
                                    "generated_answer",
                                )
                                if result_key not in existing_keys:
                                    missing_configs.append((config, result_key))
                            if not missing_configs:
                                continue

                            ranked_chunks = get_top_k_chunks(
                                question=example["question"],
                                top_k=max(
                                    config["top_k"]
                                    for config, _ in missing_configs
                                ),
                                where=where,
                                retrieval_method=retrieval_method,
                                reranker_enabled=reranker_enabled,
                                reranker_pool_size=(
                                    reranker_pool_size
                                    or OPTIMIZED_RAG_CONFIG["reranker_pool_size"]
                                ),
                            )
                            for config, result_key in missing_configs:
                                chunks = [
                                    dict(chunk)
                                    for chunk in ranked_chunks[: config["top_k"]]
                                ]
                                retrieval_metrics = evaluate_retrieval(
                                    chunks=chunks,
                                    k=config["top_k"],
                                    evidence_alignment=evidence_alignment,
                                )
                                retrieval_metrics["reranker_pool_size"] = (
                                    config["reranker_pool_size"]
                                )
                                retrieval_metrics["reranker_model"] = (
                                    DEFAULT_RERANKER_MODEL
                                    if config["reranker_enabled"]
                                    else None
                                )
                                retrieval_metrics[
                                    "reranker_model_revision"
                                ] = (
                                    DEFAULT_RERANKER_REVISION
                                    if config["reranker_enabled"]
                                    else None
                                )
                                retrieval_metrics[
                                    "semantic_search_backend"
                                ] = (
                                    SEMANTIC_SEARCH_BACKEND
                                    if retrieval_method
                                    in {"semantic", "hybrid"}
                                    else None
                                )
                                generation_chunks = limit_chunks_to_prompt_budget(
                                    question=example["question"],
                                    retrieved_chunks=chunks,
                                    model=model,
                                )
                                yield {
                                    "example": example,
                                    "config": config,
                                    "result_key": result_key,
                                    "chunks": chunks,
                                    "generation_chunks": generation_chunks,
                                    "retrieval_metrics": retrieval_metrics,
                                }

    def generate_task(task):
        answer = generate_answer(
            question=task["example"]["question"],
            retrieved_chunks=task["generation_chunks"],
        )
        generation_metrics = evaluate_docfinqa_answer_metrics(
            generated_answer=answer,
            gold_answer=task["example"]["gold_answer"],
        )
        generation_metrics.update(
            {
                "answer_source": "openai",
                "model": model,
                "generation_context_chunk_count": len(
                    task["generation_chunks"]
                ),
                "generation_context_chunk_ids": [
                    chunk["id"] for chunk in task["generation_chunks"]
                ],
                "prompt_was_truncated": _context_was_truncated(
                    task["chunks"], task["generation_chunks"]
                ),
            }
        )
        return answer, generation_metrics

    def handle_generated_result(task, output):
        nonlocal rows_processed
        answer, generation_metrics = output
        is_correct = generation_metrics["docfinqa_answer_correct"]
        if type(is_correct) is not bool:
            raise TypeError("docfinqa_answer_correct must be Boolean.")
        config = task["config"]
        config_key = _parameter_config_key(config)
        record_parameter_result(
            parameter_scores=parameter_scores,
            config_key=config_key,
            is_correct=is_correct,
            retrieval_metrics=task["retrieval_metrics"],
            generation_metrics=generation_metrics,
        )
        question_id = str(task["example"]["question_id"])
        config_outcomes = parameter_outcomes.setdefault(config_key, {})
        if question_id in config_outcomes:
            raise RuntimeError(
                "Duplicate question outcome encountered for "
                f"configuration {config_key}: {question_id}"
            )
        config_outcomes[question_id] = is_correct
        result = {
            "run_id": run_id,
            "result_key": task["result_key"],
            "experiment_name": experiment_name,
            "dataset": "docfinqa",
            "split": split,
            "question_id": task["example"]["question_id"],
            "question": task["example"]["question"],
            "gold_answer": task["example"]["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": config["retrieval_method"],
            "chunk_strategy": config["strategy"],
            "chunk_size": config["chunk_size"],
            "top_k": config["top_k"],
            "reranker_used": config["reranker_enabled"],
            "reranker_pool_size": config["reranker_pool_size"],
            "retrieved_chunk_ids": [
                chunk["id"] for chunk in task["chunks"]
            ],
            "retrieved_chunks": _compact_retrieved_chunks(task["chunks"]),
            "retrieval_metrics": task["retrieval_metrics"],
            "generation_metrics": generation_metrics,
        }
        rows_processed += 1
        if return_results:
            results.append(result)
        if log_results:
            pending_results.append(result)
            if len(pending_results) >= FULL_RAG_RESULT_LOG_BATCH_SIZE:
                flush_pending_results()

    try:
        _run_bounded_generation(
            task_stream(),
            generate_task,
            handle_generated_result,
            max_workers=openai_concurrency,
        )
    finally:
        if log_results:
            flush_pending_results()

    questions_processed = len(examples)

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
    sampled_dev_protocol_complete = bool(
        split == DEV_SPLIT
        and sample_size == DEV_SAMPLE_SIZE
        and sample_seed == DEFAULT_SAMPLE_SEED
        and start_index == 0
        and limit is None
        and questions_processed == DEV_SAMPLE_SIZE
        and rows_saved == expected_rows
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
        elif winner_source_split is not None and not sampled_dev_protocol_complete:
            best_parameters_save_reason = "incomplete_sampled_protocol"
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

    is_authoritative = bool(
        best_parameters_saved
        and (
            statistical_analysis_saved
            if winner_source_split is not None
            else complete_parameter_sweep
        )
    )
    is_full_run = bool(
        is_authoritative
        and sample_size is None
        and start_index == 0
        and limit is None
    )
    if log_results:
        complete_experiment_run(
            run_id=run_id,
            questions_processed=questions_processed,
            rows_saved=rows_saved,
            is_full_run=is_full_run,
            is_authoritative=is_authoritative,
        )

    summary = {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "is_authoritative": is_authoritative,
        "experiment_name": experiment_name,
        "dataset": "docfinqa",
        "split": split,
        "model": model,
        "start_index": start_index,
        "requested_limit": limit,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "openai_concurrency": openai_concurrency,
        "question_ids": question_ids,
        "questions_processed": questions_processed,
        "chunk_sizes": chunk_sizes,
        "strategies": strategies,
        "retrieval_methods": retrieval_methods,
        "top_k_values": top_k_values,
        "reranker_enabled_values": reranker_enabled_values,
        "reranker_pool_sizes": reranker_pool_sizes,
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "reranker_model_revision": DEFAULT_RERANKER_REVISION,
        "semantic_search_backend": SEMANTIC_SEARCH_BACKEND,
        "reranker_configurations": [
            {
                "reranker_enabled": enabled,
                "reranker_pool_size": pool_size,
            }
            for enabled, pool_size in reranker_configs
        ],
        "parameter_combinations_per_question": combinations_per_question,
        "uses_candidate_shortlist": candidate_config_keys is not None,
        "complete_candidate_outcomes": complete_candidate_outcomes,
        "complete_parameter_sweep": complete_parameter_sweep,
        "rows_processed": rows_processed,
        "rows_saved": rows_saved,
        "rows_already_saved": len(existing_rows),
        "rows_added_this_attempt": rows_added_this_attempt,
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
    sample_size: int | None = DEV_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """Runs the saved train finalists over the seeded dev sample."""

    if resume_run_id:
        saved_run = get_experiment_run(resume_run_id)
        if (
            saved_run is None
            or saved_run.get("experiment_name") != DEV_SHORTLIST_EXPERIMENT
        ):
            raise RuntimeError(
                "The requested run is not a saved dev shortlist checkpoint."
            )
        candidate_configs = (
            saved_run.get("parameters", {}).get("candidate_configs") or []
        )
        shortlist_source = {
            "experiment_name": RETRIEVAL_SWEEP_EXPERIMENT,
            "split": TRAIN_SPLIT,
            "questions_run": None,
            "updated_at": None,
            "resume_source": "saved_run_manifest",
        }
    else:
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
        ranking_rule = saved_shortlist.get("ranking_rule") or {}
        expected_retrieval_provenance = {
            "reranker_model": DEFAULT_RERANKER_MODEL,
            "reranker_model_revision": DEFAULT_RERANKER_REVISION,
            "semantic_search_backend": SEMANTIC_SEARCH_BACKEND,
        }
        if any(
            ranking_rule.get(key) != value
            for key, value in expected_retrieval_provenance.items()
        ):
            raise RuntimeError(
                "The saved retrieval shortlist uses a different reranker "
                "or semantic search backend. Run POST /run-chunk-rag again."
            )
        candidates = saved_shortlist.get("candidates") or []
        candidate_configs = [
            _parameter_config_from_candidate(candidate)
            for candidate in candidates
        ]
        shortlist_source = {
            "experiment_name": saved_shortlist["source_experiment"],
            "split": saved_shortlist["source_split"],
            "questions_run": saved_shortlist["questions_run"],
            "updated_at": saved_shortlist["updated_at"],
            "resume_source": "current_authoritative_shortlist",
        }

    if len(candidate_configs) != RETRIEVAL_SHORTLIST_SIZE:
        raise RuntimeError(
            "The saved retrieval shortlist is incomplete: expected "
            f"{RETRIEVAL_SHORTLIST_SIZE} candidates, found "
            f"{len(candidate_configs)}."
        )
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
        sample_size=sample_size,
        sample_seed=sample_seed,
        resume_run_id=resume_run_id,
    )
    summary["retrieval_shortlist_source"] = shortlist_source
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
    sample_size: int | None = TRAIN_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """
    Runs the seeded retrieval-only train sweep and saves its top three.
    """

    if resume_run_id and not log_results:
        raise ValueError("resume_run_id requires log_results=True.")

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

    examples, question_ids = _materialize_protocol_examples(
        split=split,
        sample_size=sample_size,
        sample_seed=sample_seed,
        start_index=start_index,
        limit=limit,
        resume_run_id=resume_run_id,
    )
    combinations_per_question = (
        len(chunk_sizes)
        * len(strategies)
        * len(retrieval_methods)
        * len(top_k_values)
        * len(reranker_configs)
    )
    expected_rows = len(examples) * combinations_per_question
    protocol_parameters = {
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "sampling_method": "uniform_without_replacement_reservoir",
        "question_ids": question_ids,
        "top_k_values": list(top_k_values),
        "strategies": list(strategies),
        "retrieval_methods": list(retrieval_methods),
        "chunk_sizes": list(chunk_sizes),
        "reranker_configurations": [list(value) for value in reranker_configs],
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "reranker_model_revision": DEFAULT_RERANKER_REVISION,
        "semantic_search_backend": SEMANTIC_SEARCH_BACKEND,
        "ranking_weights": RETRIEVAL_SCORE_WEIGHTS,
        "shortlist_size": RETRIEVAL_SHORTLIST_SIZE,
    }

    results = []
    pending_results = []
    parameter_scores = {}
    rows_processed = 0
    rows_added_this_attempt = 0
    run_id = None
    existing_rows = []
    existing_keys = set()
    phase_timings = {
        "corpus_load_seconds": 0.0,
        "keyword_ranking_seconds": 0.0,
        "semantic_ranking_seconds": 0.0,
        "hybrid_fusion_seconds": 0.0,
        "reranking_seconds": 0.0,
        "evidence_alignment_seconds": 0.0,
        "metric_evaluation_seconds": 0.0,
        "database_seconds": 0.0,
    }
    working_question_durations = []
    sweep_started = perf_counter()

    if log_results:
        create_results_table()
        if resume_run_id:
            resume_experiment_run(
                run_id=resume_run_id,
                dataset="docfinqa",
                experiment_name=RETRIEVAL_SWEEP_EXPERIMENT,
                split=split,
                model=None,
                parameters=protocol_parameters,
                expected_rows=expected_rows,
            )
            run_id = resume_run_id
            existing_rows = get_run_results_for_resume(run_id)
        else:
            run_id = start_experiment_run(
                dataset="docfinqa",
                experiment_name=RETRIEVAL_SWEEP_EXPERIMENT,
                split=split,
                model=None,
                parameters=protocol_parameters,
                expected_rows=expected_rows,
            )

    for row in existing_rows:
        if not row.get("result_key"):
            raise RuntimeError(
                "This checkpoint predates resumable result keys and cannot "
                "be resumed safely."
            )
        existing_keys.add(row["result_key"])
        record_retrieval_result(
            parameter_scores=parameter_scores,
            config_key=_parameter_config_key(_config_from_result_row(row)),
            retrieval_metrics=row["retrieval_metrics"],
        )

    rows_saved = len(existing_keys)

    def flush_pending_results():
        nonlocal rows_saved, rows_added_this_attempt

        if not pending_results:
            return

        database_started = perf_counter()
        result_ids = log_question_results(
            pending_results,
            ensure_table=False,
        )
        phase_timings["database_seconds"] += (
            perf_counter() - database_started
        )
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id

        rows_saved += len(result_ids)
        rows_added_this_attempt += len(result_ids)
        existing_keys.update(
            result["result_key"] for result in pending_results
        )
        pending_results.clear()

    max_top_k = max(top_k_values)
    for question_number, example in enumerate(examples, start=1):
        question_started = perf_counter()
        filter_work = []

        for current_chunk_size in chunk_sizes:
            for strategy in strategies:
                missing_by_retrieval_config = {}
                for retrieval_method in retrieval_methods:
                    for reranker_enabled, reranker_pool_size in reranker_configs:
                        retrieval_config = (
                            retrieval_method,
                            reranker_enabled,
                            reranker_pool_size,
                        )
                        missing_results = []
                        for top_k in top_k_values:
                            config = {
                                "chunk_size": current_chunk_size,
                                "strategy": strategy,
                                "retrieval_method": retrieval_method,
                                "top_k": top_k,
                                "reranker_enabled": reranker_enabled,
                                "reranker_pool_size": (
                                    reranker_pool_size
                                    if reranker_enabled
                                    else None
                                ),
                            }
                            result_key = _result_key(
                                example["question_id"],
                                config,
                                "retrieval",
                            )
                            if result_key not in existing_keys:
                                missing_results.append((config, result_key))

                        if missing_results:
                            missing_by_retrieval_config[retrieval_config] = (
                                missing_results
                            )

                if missing_by_retrieval_config:
                    filter_work.append(
                        (
                            current_chunk_size,
                            strategy,
                            {
                                "$and": [
                                    {
                                        "document_id": {
                                            "$eq": example["document_id"]
                                        }
                                    },
                                    {"strategy": {"$eq": strategy}},
                                    {
                                        "chunk_size": {
                                            "$eq": current_chunk_size
                                        }
                                    },
                                ]
                            },
                            missing_by_retrieval_config,
                        )
                    )

        if filter_work:
            gold_evidence = get_finqa_gold_evidence(
                split=split,
                question=example["question"],
                gold_answer=example["gold_answer"],
                gold_program=example.get("program"),
                document_text=example.get("document_text"),
            )

        for filter_number, (
            current_chunk_size,
            strategy,
            where,
            missing_by_retrieval_config,
        ) in enumerate(filter_work, start=1):
            filter_started = perf_counter()
            sweep_rankings = get_parameter_sweep_rankings(
                question=example["question"],
                max_top_k=max_top_k,
                where=where,
                retrieval_configs=list(missing_by_retrieval_config),
            )
            for timing_name, seconds in sweep_rankings["timings"].items():
                phase_timings[timing_name] += seconds

            evidence_started = perf_counter()
            evidence_alignment = align_docfinqa_evidence(
                gold_evidence=gold_evidence,
                all_question_chunks=sweep_rankings["all_chunks"],
            )
            phase_timings["evidence_alignment_seconds"] += (
                perf_counter() - evidence_started
            )

            for retrieval_config, missing_results in (
                missing_by_retrieval_config.items()
            ):
                retrieval_method, reranker_enabled, reranker_pool_size = (
                    retrieval_config
                )
                ranked_chunks = sweep_rankings["rankings"][retrieval_config]

                for config, result_key in missing_results:
                    top_k = config["top_k"]
                    chunks = [
                        dict(chunk)
                        for chunk in ranked_chunks[:top_k]
                    ]
                    evaluation_started = perf_counter()
                    retrieval_metrics = evaluate_retrieval(
                        chunks=chunks,
                        k=top_k,
                        evidence_alignment=evidence_alignment,
                    )
                    phase_timings["metric_evaluation_seconds"] += (
                        perf_counter() - evaluation_started
                    )
                    retrieval_metrics["reranker_pool_size"] = (
                        reranker_pool_size
                        if reranker_enabled
                        else None
                    )
                    retrieval_metrics["reranker_model"] = (
                        DEFAULT_RERANKER_MODEL
                        if reranker_enabled
                        else None
                    )
                    retrieval_metrics["reranker_model_revision"] = (
                        DEFAULT_RERANKER_REVISION
                        if reranker_enabled
                        else None
                    )
                    retrieval_metrics["semantic_search_backend"] = (
                        SEMANTIC_SEARCH_BACKEND
                        if retrieval_method in {"semantic", "hybrid"}
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
                        "result_key": result_key,
                        "experiment_name": RETRIEVAL_SWEEP_EXPERIMENT,
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
                            chunk["id"] for chunk in chunks
                        ],
                        "retrieved_chunks": _compact_retrieved_chunks(chunks),
                        "retrieval_metrics": retrieval_metrics,
                        "generation_metrics": {},
                    }
                    rows_processed += 1

                    if return_results:
                        results.append(result)

                    if log_results:
                        pending_results.append(result)
                        if len(pending_results) >= RESULT_LOG_BATCH_SIZE:
                            flush_pending_results()

            print(
                f"{RETRIEVAL_SWEEP_EXPERIMENT}: "
                f"question {question_number}/{len(examples)}, "
                f"corpus {filter_number}/{len(filter_work)} complete "
                f"({strategy}, chunk_size={current_chunk_size}); "
                f"corpus_seconds={perf_counter() - filter_started:.2f}",
                flush=True,
            )

        if log_results:
            flush_pending_results()
            database_started = perf_counter()
            update_experiment_run_progress(
                run_id=run_id,
                questions_processed=question_number,
                rows_saved=rows_saved,
            )
            phase_timings["database_seconds"] += (
                perf_counter() - database_started
            )

        question_seconds = perf_counter() - question_started
        if filter_work:
            working_question_durations.append(question_seconds)
        average_seconds = (
            sum(working_question_durations) / len(working_question_durations)
            if working_question_durations
            else 0.0
        )
        estimated_hours = (
            average_seconds * (len(examples) - question_number) / 3600
        )
        print(
            f"{RETRIEVAL_SWEEP_EXPERIMENT}: "
            f"{question_number}/{len(examples)} questions checkpointed; "
            f"question_seconds={question_seconds:.2f}; "
            f"estimated_remaining_hours={estimated_hours:.2f}",
            flush=True,
        )

    questions_processed = len(examples)
    top_candidates = rank_retrieval_parameter_configs(
        parameter_scores,
        limit=RETRIEVAL_SHORTLIST_SIZE,
    )
    shortlist_saved = False
    shortlist_save_reason = None

    protocol_complete = bool(
        split == TRAIN_SPLIT
        and sample_size == TRAIN_SAMPLE_SIZE
        and sample_seed == DEFAULT_SAMPLE_SEED
        and start_index == 0
        and limit is None
        and questions_processed == TRAIN_SAMPLE_SIZE
        and rows_saved == expected_rows
    )

    if not top_candidates:
        shortlist_save_reason = "no_results"
    elif not log_results:
        shortlist_save_reason = "result_logging_disabled"
    elif not protocol_complete:
        shortlist_save_reason = "incomplete_sampled_protocol"
    elif split != TRAIN_SPLIT:
        shortlist_save_reason = "not_train_split"
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
                "sample_size": sample_size,
                "sample_seed": sample_seed,
                "question_ids": question_ids,
                "reranker_model": DEFAULT_RERANKER_MODEL,
                "reranker_model_revision": DEFAULT_RERANKER_REVISION,
                "semantic_search_backend": SEMANTIC_SEARCH_BACKEND,
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

    is_full_run = bool(
        shortlist_saved
        and sample_size is None
        and start_index == 0
        and limit is None
    )
    if log_results:
        complete_experiment_run(
            run_id=run_id,
            questions_processed=questions_processed,
            rows_saved=rows_saved,
            is_full_run=is_full_run,
            is_authoritative=shortlist_saved,
        )

    total_elapsed_seconds = perf_counter() - sweep_started
    timing_summary = {
        key: round(value, 3)
        for key, value in phase_timings.items()
    }
    timing_summary["total_elapsed_seconds"] = round(
        total_elapsed_seconds,
        3,
    )
    timing_summary["average_working_question_seconds"] = round(
        (
            sum(working_question_durations) / len(working_question_durations)
            if working_question_durations
            else 0.0
        ),
        3,
    )

    summary = {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "is_authoritative": shortlist_saved,
        "experiment_name": RETRIEVAL_SWEEP_EXPERIMENT,
        "dataset": "docfinqa",
        "split": split,
        "start_index": start_index,
        "requested_limit": limit,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "question_ids": question_ids,
        "questions_processed": questions_processed,
        "chunk_sizes": chunk_sizes,
        "strategies": strategies,
        "retrieval_methods": retrieval_methods,
        "top_k_values": top_k_values,
        "reranker_enabled_values": reranker_enabled_values,
        "reranker_pool_sizes": reranker_pool_sizes,
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "reranker_model_revision": DEFAULT_RERANKER_REVISION,
        "semantic_search_backend": SEMANTIC_SEARCH_BACKEND,
        "reranker_configurations": [
            {
                "reranker_enabled": enabled,
                "reranker_pool_size": pool_size,
            }
            for enabled, pool_size in reranker_configs
        ],
        "parameter_combinations_per_question": combinations_per_question,
        "complete_parameter_sweep": complete_parameter_sweep,
        "rows_processed": rows_processed,
        "rows_saved": rows_saved,
        "rows_already_saved": len(existing_rows),
        "rows_added_this_attempt": rows_added_this_attempt,
        "results_returned": len(results),
        "timings": timing_summary,
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
    limit: int | None = None,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    chunk_size: int = 512,
    log_results: bool = True,
    return_results: bool = False,
    sample_size: int | None = MATH_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """
    Compares direct LLM math with an LLM-planned, Python-executed math agent.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required before starting the math-agent sweep."
        )

    top_k_values = top_k_values or [5]
    strategies = strategies or ["section"]
    retrieval_methods = retrieval_methods or ["hybrid"]
    if resume_run_id and not log_results:
        raise ValueError("A resumed run requires result logging.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("Every top_k value must be greater than zero.")
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("Every top_k value must be unique.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("Every chunk strategy must be unique.")
    if len(set(retrieval_methods)) != len(retrieval_methods):
        raise ValueError("Every retrieval method must be unique.")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    openai_concurrency = get_openai_concurrency()

    examples, question_ids = _materialize_protocol_examples(
        split=split,
        sample_size=sample_size,
        sample_seed=sample_seed,
        start_index=start_index,
        limit=limit,
        resume_run_id=resume_run_id,
    )
    comparison_methods = ("direct_llm", "math_agent")
    protocol_configs = [
        {
            "retrieval_method": retrieval_method,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "reranker_enabled": False,
            "reranker_pool_size": None,
        }
        for strategy in strategies
        for retrieval_method in retrieval_methods
        for top_k in top_k_values
    ]
    configurations_per_question = len(protocol_configs)
    rows_per_question = configurations_per_question * len(comparison_methods)
    expected_result_keys = {
        _result_key(question_id, config, method)
        for question_id in question_ids
        for config in protocol_configs
        for method in comparison_methods
    }
    expected_rows = len(expected_result_keys)
    expected_config_ids = {
        _parameter_config_id(config) for config in protocol_configs
    }
    protocol_parameters = {
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "sampling_method": "uniform_without_replacement_reservoir",
        "question_ids": question_ids,
        "top_k_values": list(top_k_values),
        "strategies": list(strategies),
        "retrieval_methods": list(retrieval_methods),
        "chunk_size": chunk_size,
        "comparison_methods": list(comparison_methods),
        "reranker_enabled": False,
    }

    results = []
    pending_results = []
    outcomes_by_config = {}
    rows_processed = 0
    rows_added_this_attempt = 0
    run_id = None
    existing_rows = []
    existing_keys = set()
    persisted_question_counts = {}

    if log_results:
        create_results_table()
        if resume_run_id:
            resume_experiment_run(
                run_id=resume_run_id,
                dataset="docfinqa",
                experiment_name=MATH_AGENT_EXPERIMENT,
                split=split,
                model=model,
                parameters=protocol_parameters,
                expected_rows=expected_rows,
            )
            run_id = resume_run_id
            existing_rows = get_run_results_for_resume(run_id)
        else:
            run_id = start_experiment_run(
                dataset="docfinqa",
                experiment_name=MATH_AGENT_EXPERIMENT,
                split=split,
                model=model,
                parameters=protocol_parameters,
                expected_rows=expected_rows,
            )

    for row in existing_rows:
        result_key = row.get("result_key")
        if not result_key:
            raise RuntimeError(
                "This checkpoint predates resumable result keys and cannot "
                "be resumed safely."
            )
        if result_key not in expected_result_keys:
            raise RuntimeError(
                "The saved checkpoint contains a result outside the math "
                f"protocol: {result_key}"
            )
        config = _config_from_result_row(row)
        config_id = _parameter_config_id(config)
        if config_id not in expected_config_ids:
            raise RuntimeError(
                "The saved checkpoint contains an unexpected math-agent "
                f"configuration: {config_id}"
            )
        generation_metrics = row.get("generation_metrics") or {}
        method = generation_metrics.get("comparison_method")
        if method not in comparison_methods:
            raise RuntimeError(
                "The saved math-agent checkpoint contains an unknown "
                f"comparison method: {method!r}"
            )
        is_correct = row["is_correct"]
        if type(is_correct) is not bool:
            raise TypeError("Saved docfinqa_answer_correct must be Boolean.")

        config_outcomes = outcomes_by_config.setdefault(
            config_id,
            {
                "config": config,
                "direct_llm": {},
                "math_agent": {},
            },
        )
        question_id = str(row["question_id"])
        if question_id in config_outcomes[method]:
            raise RuntimeError(
                "Duplicate saved math comparison outcome for "
                f"{config_id}, {method}, question {question_id}."
            )
        config_outcomes[method][question_id] = is_correct
        existing_keys.add(result_key)
        persisted_question_counts[question_id] = (
            persisted_question_counts.get(question_id, 0) + 1
        )

    rows_saved = len(existing_keys)

    def flush_pending_results():
        nonlocal rows_saved, rows_added_this_attempt
        if not pending_results:
            return
        result_ids = log_question_results(
            pending_results,
            ensure_table=False,
        )
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id
            existing_keys.add(result["result_key"])
            question_id = str(result["question_id"])
            persisted_question_counts[question_id] = (
                persisted_question_counts.get(question_id, 0) + 1
            )
        rows_added_this_attempt += len(result_ids)
        rows_saved = len(existing_keys)
        pending_results.clear()
        checkpointed_questions = sum(
            count == rows_per_question
            for count in persisted_question_counts.values()
        )
        update_experiment_run_progress(
            run_id=run_id,
            questions_processed=checkpointed_questions,
            rows_saved=rows_saved,
        )
        print(
            f"{MATH_AGENT_EXPERIMENT}: "
            f"{checkpointed_questions}/{len(examples)} questions checkpointed",
            flush=True,
        )

    generation_tasks = []
    for example in examples:
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
                    config = {
                        "retrieval_method": retrieval_method,
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "top_k": top_k,
                        "reranker_enabled": False,
                        "reranker_pool_size": None,
                    }
                    config_id = _parameter_config_id(config)
                    missing_methods = [
                        method
                        for method in comparison_methods
                        if _result_key(
                            example["question_id"],
                            config,
                            method,
                        )
                        not in existing_keys
                    ]
                    if not missing_methods:
                        continue
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

                    for method in missing_methods:
                        generation_tasks.append(
                            {
                                "example": example,
                                "config": config,
                                "config_id": config_id,
                                "method": method,
                                "result_key": _result_key(
                                    example["question_id"],
                                    config,
                                    method,
                                ),
                                "chunks": chunks,
                                "shared_chunks": shared_chunks,
                                "shared_context_chunk_ids": (
                                    shared_context_chunk_ids
                                ),
                                "shared_context_was_truncated": (
                                    shared_context_was_truncated
                                ),
                                "retrieval_metrics": retrieval_metrics,
                            }
                        )


    def generate_task(task):
        example = task["example"]
        chunks = task["chunks"]
        shared_chunks = task["shared_chunks"]
        source_context_character_count = sum(
            len(str(chunk.get("text", ""))) for chunk in chunks
        )

        if task["method"] == "direct_llm":
            direct_context_metrics = {}
            answer = generate_answer(
                question=example["question"],
                retrieved_chunks=shared_chunks,
                generation_context_metrics=direct_context_metrics,
            )
            metrics = evaluate_docfinqa_answer_metrics(
                generated_answer=answer,
                gold_answer=example["gold_answer"],
            )
            metrics.update(
                {
                    "answer_source": "direct_llm_calculation",
                    "comparison_method": "direct_llm",
                    "model": model,
                    **direct_context_metrics,
                    "shared_context_chunk_ids": task[
                        "shared_context_chunk_ids"
                    ],
                    "source_context_chunk_count": len(chunks),
                    "source_context_character_count": (
                        source_context_character_count
                    ),
                    "prompt_was_truncated": (
                        task["shared_context_was_truncated"]
                        or direct_context_metrics.get(
                            "prompt_was_truncated",
                            False,
                        )
                    ),
                }
            )
            return answer, metrics

        agent_result = math_agent(
            question=example["question"],
            chunks=shared_chunks,
            dataset="docfinqa",
        )
        answer = agent_result["answer"]
        metrics = evaluate_docfinqa_answer_metrics(
            generated_answer=answer,
            gold_answer=example["gold_answer"],
        )
        metrics.update(
            {
                "answer_source": "llm_program_python_executor",
                "comparison_method": "math_agent",
                "model": agent_result["model"],
                "shared_context_chunk_ids": task[
                    "shared_context_chunk_ids"
                ],
                "math_prompt_version": agent_result["prompt_version"],
                "math_agent_status": agent_result["status"],
                "math_program": agent_result["program"],
                "raw_math_answer": agent_result["raw_answer"],
                "raw_program_response": agent_result["raw_model_output"],
                "program_parse_succeeded": agent_result[
                    "program_parse_succeeded"
                ],
                "execution_succeeded": agent_result["execution_succeeded"],
                "execution_error": agent_result["error"],
                "execution_steps": agent_result["execution_steps"],
                "literal_operand_count": agent_result[
                    "literal_operand_count"
                ],
                "grounded_operand_count": agent_result[
                    "grounded_operand_count"
                ],
                "operand_grounding_rate": agent_result[
                    "operand_grounding_rate"
                ],
                "ungrounded_operands": agent_result["ungrounded_operands"],
                "generation_context_chunk_ids": agent_result[
                    "generation_context_chunk_ids"
                ],
                "generation_context_chunk_count": agent_result[
                    "generation_context_chunk_count"
                ],
                "prompt_was_truncated": (
                    agent_result["prompt_was_truncated"]
                    or task["shared_context_was_truncated"]
                ),
                "source_context_chunk_count": len(chunks),
                "source_context_character_count": (
                    source_context_character_count
                ),
                "generation_context_character_count": sum(
                    len(str(chunk.get("text", "")))
                    for chunk in shared_chunks
                ),
            }
        )
        return answer, metrics

    def handle_generated_result(task, output):
        nonlocal rows_processed
        answer, generation_metrics = output
        method = task["method"]
        is_correct = generation_metrics["docfinqa_answer_correct"]
        if type(is_correct) is not bool:
            raise TypeError("docfinqa_answer_correct must be Boolean.")

        config_outcomes = outcomes_by_config.setdefault(
            task["config_id"],
            {
                "config": task["config"],
                "direct_llm": {},
                "math_agent": {},
            },
        )
        question_id = str(task["example"]["question_id"])
        if question_id in config_outcomes[method]:
            raise RuntimeError(
                "Duplicate math comparison outcome for "
                f"{task['config_id']}, {method}, question {question_id}."
            )
        config_outcomes[method][question_id] = is_correct

        result = {
            "run_id": run_id,
            "result_key": task["result_key"],
            "experiment_name": MATH_AGENT_EXPERIMENT,
            "dataset": "docfinqa",
            "split": split,
            "question_id": task["example"]["question_id"],
            "question": task["example"]["question"],
            "gold_answer": task["example"]["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": task["config"]["retrieval_method"],
            "chunk_strategy": task["config"]["strategy"],
            "chunk_size": task["config"]["chunk_size"],
            "top_k": task["config"]["top_k"],
            "reranker_used": False,
            "reranker_pool_size": None,
            "retrieved_chunk_ids": [
                chunk["id"] for chunk in task["chunks"]
            ],
            "retrieved_chunks": _compact_retrieved_chunks(task["chunks"]),
            "retrieval_metrics": task["retrieval_metrics"],
            "generation_metrics": generation_metrics,
        }
        rows_processed += 1
        if return_results:
            results.append(result)
        if log_results:
            pending_results.append(result)
            if len(pending_results) >= FULL_RAG_RESULT_LOG_BATCH_SIZE:
                flush_pending_results()

    try:
        _run_bounded_generation(
            generation_tasks,
            generate_task,
            handle_generated_result,
            max_workers=openai_concurrency,
        )
    finally:
        if log_results:
            flush_pending_results()

    questions_processed = len(examples)

    statistical_analyses = []
    for config_id in sorted(outcomes_by_config):
        config_outcomes = outcomes_by_config[config_id]
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

    expected_question_ids = set(question_ids)
    complete_paired_outcomes = (
        set(outcomes_by_config) == expected_config_ids
        and all(
            set(config_outcomes["direct_llm"]) == expected_question_ids
            and set(config_outcomes["math_agent"]) == expected_question_ids
            for config_outcomes in outcomes_by_config.values()
        )
    )
    exact_protocol_config = protocol_configs == [
        {
            "retrieval_method": "hybrid",
            "strategy": "section",
            "chunk_size": 512,
            "top_k": 5,
            "reranker_enabled": False,
            "reranker_pool_size": None,
        }
    ]
    protocol_complete = bool(
        split == TUNING_SPLIT
        and sample_size == MATH_SAMPLE_SIZE
        and sample_seed == DEFAULT_SAMPLE_SEED
        and start_index == 0
        and limit is None
        and questions_processed == MATH_SAMPLE_SIZE
        and exact_protocol_config
        and complete_paired_outcomes
        and existing_keys == expected_result_keys
        and rows_saved == expected_rows
        and len(statistical_analyses) == 1
    )
    is_full_run = False
    statistical_analyses_saved = 0
    if log_results and protocol_complete:
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

    is_authoritative = bool(
        protocol_complete
        and statistical_analyses_saved == len(statistical_analyses)
        and statistical_analyses_saved > 0
    )

    if log_results:
        complete_experiment_run(
            run_id=run_id,
            questions_processed=questions_processed,
            rows_saved=rows_saved,
            is_full_run=is_full_run,
            is_authoritative=is_authoritative,
        )

    summary = {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "is_authoritative": is_authoritative,
        "experiment_name": MATH_AGENT_EXPERIMENT,
        "dataset": "docfinqa",
        "split": split,
        "model": model,
        "start_index": start_index,
        "requested_limit": limit,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "openai_concurrency": openai_concurrency,
        "question_ids": question_ids,
        "questions_processed": questions_processed,
        "configuration_count": len(outcomes_by_config),
        "comparison_methods": list(comparison_methods),
        "expected_rows": expected_rows,
        "rows_processed": rows_processed,
        "rows_saved": rows_saved,
        "rows_already_saved": len(existing_rows),
        "rows_added_this_attempt": rows_added_this_attempt,
        "statistical_analyses": statistical_analyses,
        "statistical_analyses_saved": statistical_analyses_saved,
        "statistical_analysis_save_reason": (
            None
            if statistical_analyses_saved
            else "complete_50_question_protocol_required"
        ),
        "results_returned": len(results),
    }
    if return_results:
        summary["results"] = results

    return summary


def get_baseline_results(
    sample_size: int | None = TEST_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required before starting the baseline runs."
        )

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    optimized_config_source = "saved_dev_shortlist_winner"
    if resume_run_id:
        saved_run = get_experiment_run(resume_run_id)
        if (
            saved_run is None
            or saved_run.get("experiment_name") != "baseline"
        ):
            raise RuntimeError(
                "The requested run is not a saved baseline checkpoint."
            )
        saved_parameters = saved_run.get("parameters") or {}
        config = saved_parameters.get("optimized_parameters")
        if not config:
            raise RuntimeError(
                "The baseline checkpoint does not contain its frozen "
                "optimized configuration."
            )
        saved_optimized_parameters = {
            "source_experiment": saved_parameters.get(
                "optimized_config_source_experiment"
            ),
            "source_split": saved_parameters.get(
                "optimized_config_source_split"
            ),
            "selection_metrics": saved_parameters.get(
                "optimized_config_selection_metrics"
            ),
        }
    else:
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
                "dev shortlist sweep. Run POST /run-full-rag first."
            )
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

    if (
        saved_optimized_parameters.get("source_experiment")
        != DEV_SHORTLIST_EXPERIMENT
    ):
        raise RuntimeError(
            "The saved dev parameters did not come from the completed "
            "dev shortlist sweep. Run POST /run-full-rag first."
        )
    baseline_methods = (
        "no_context",
        "full_document",
        "naive_rag",
        "optimized_rag",
    )
    naive_config = {
        "retrieval_method": "semantic",
        "reranker_enabled": False,
        "strategy": "fixed",
        "chunk_size": 512,
        "top_k": 3,
        "reranker_pool_size": None,
    }

    examples, question_ids = _materialize_protocol_examples(
        split=BASELINE_SPLIT,
        sample_size=sample_size,
        sample_seed=sample_seed,
        start_index=0,
        limit=None,
        resume_run_id=resume_run_id,
    )
    expected_rows = len(examples) * len(baseline_methods)
    openai_concurrency = get_openai_concurrency()
    protocol_parameters = {
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "sampling_method": "uniform_without_replacement_reservoir",
        "question_ids": question_ids,
        "baseline_methods": list(baseline_methods),
        "optimized_parameters": config,
        "optimized_config_source": optimized_config_source,
        "optimized_config_source_experiment": (
            saved_optimized_parameters["source_experiment"]
        ),
        "optimized_config_source_split": saved_optimized_parameters[
            "source_split"
        ],
        "optimized_config_selection_metrics": (
            saved_optimized_parameters["selection_metrics"]
        ),
    }

    # Construct the shared SDK client before creating the durable run or
    # allowing worker threads to access the cached client concurrently.
    from app.rag.generator import get_openai_client

    get_openai_client()
    create_results_table()
    if resume_run_id:
        resume_experiment_run(
            run_id=resume_run_id,
            dataset="docfinqa",
            experiment_name="baseline",
            split=BASELINE_SPLIT,
            model=model,
            parameters=protocol_parameters,
            expected_rows=expected_rows,
        )
        run_id = resume_run_id
        existing_rows = get_run_results_for_resume(run_id)
    else:
        run_id = start_experiment_run(
            dataset="docfinqa",
            experiment_name="baseline",
            split=BASELINE_SPLIT,
            model=model,
            parameters=protocol_parameters,
            expected_rows=expected_rows,
        )
        existing_rows = []

    method_outcomes = {method: {} for method in baseline_methods}
    persisted_methods_by_question = {}
    existing_keys = set()
    question_id_set = set(question_ids)

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

    for row in existing_rows:
        result_key = row.get("result_key")
        generation_metrics = row.get("generation_metrics") or {}
        method_name = generation_metrics.get("baseline")
        question_id = str(row["question_id"])
        if not result_key:
            raise RuntimeError(
                "This checkpoint predates resumable result keys and cannot "
                "be resumed safely."
            )
        if method_name not in baseline_methods:
            raise RuntimeError(
                f"Saved baseline method is invalid: {method_name!r}."
            )
        if question_id not in question_id_set:
            raise RuntimeError(
                f"Saved question {question_id!r} is not in the run manifest."
            )
        expected_key = _result_key(question_id, None, method_name)
        if result_key != expected_key:
            raise RuntimeError(
                f"Saved baseline result key does not match {expected_key!r}."
            )
        is_correct = row["is_correct"]
        record_method_outcome(method_name, question_id, is_correct)
        existing_keys.add(result_key)
        persisted_methods_by_question.setdefault(question_id, set()).add(
            method_name
        )

    rows_saved = len(existing_keys)
    rows_added_this_attempt = 0
    pending_results = []

    def flush_pending_results():
        nonlocal rows_saved, rows_added_this_attempt
        if not pending_results:
            return

        result_ids = log_question_results(
            pending_results,
            ensure_table=False,
        )
        keys_before_flush = len(existing_keys)
        for result, result_id in zip(pending_results, result_ids):
            result["result_id"] = result_id
            existing_keys.add(result["result_key"])
            question_id = str(result["question_id"])
            method_name = result["generation_metrics"]["baseline"]
            persisted_methods_by_question.setdefault(
                question_id,
                set(),
            ).add(method_name)

        rows_saved = len(existing_keys)
        rows_added_this_attempt += rows_saved - keys_before_flush
        pending_results.clear()
        checkpointed_questions = sum(
            set(methods) == set(baseline_methods)
            for methods in persisted_methods_by_question.values()
        )
        update_experiment_run_progress(
            run_id=run_id,
            questions_processed=checkpointed_questions,
            rows_saved=rows_saved,
        )
        print(
            f"baseline: {checkpointed_questions}/{len(examples)} "
            "questions checkpointed",
            flush=True,
        )

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

    def build_task(
        *,
        example,
        method_name,
        generation_chunks,
        retrieved_chunks=None,
        retrieval_metrics=None,
        method_config=None,
    ):
        return {
            "example": example,
            "method_name": method_name,
            "result_key": _result_key(
                example["question_id"],
                None,
                method_name,
            ),
            "generation_chunks": generation_chunks,
            "retrieved_chunks": retrieved_chunks or [],
            "retrieval_metrics": retrieval_metrics or {},
            "method_config": method_config,
        }

    def task_stream():
        for example in examples:
            question_id = str(example["question_id"])

            no_context_key = _result_key(
                question_id,
                None,
                "no_context",
            )
            if no_context_key not in existing_keys:
                yield build_task(
                    example=example,
                    method_name="no_context",
                    generation_chunks=None,
                )

            full_document_key = _result_key(
                question_id,
                None,
                "full_document",
            )
            if full_document_key not in existing_keys:
                full_document_context = [
                    {
                        "chunk_id": "full_document",
                        "text": example["document_text"],
                        "metadata": {
                            "question_id": example["question_id"],
                        },
                    }
                ]
                yield build_task(
                    example=example,
                    method_name="full_document",
                    generation_chunks=full_document_context,
                )

            naive_key = _result_key(
                question_id,
                None,
                "naive_rag",
            )
            if naive_key not in existing_keys:
                naive_where = {
                    "$and": [
                        {"document_id": {"$eq": example["document_id"]}},
                        {"strategy": {"$eq": naive_config["strategy"]}},
                        {"chunk_size": {"$eq": naive_config["chunk_size"]}},
                    ]
                }
                naive_chunks = get_top_k_chunks(
                    question=example["question"],
                    top_k=naive_config["top_k"],
                    where=naive_where,
                    retrieval_method=naive_config["retrieval_method"],
                )
                naive_alignment = _get_docfinqa_evidence_alignment(
                    example=example,
                    split=BASELINE_SPLIT,
                    where=naive_where,
                )
                naive_metrics = evaluate_retrieval(
                    chunks=naive_chunks,
                    k=naive_config["top_k"],
                    evidence_alignment=naive_alignment,
                )
                naive_metrics["reranker_pool_size"] = None
                yield build_task(
                    example=example,
                    method_name="naive_rag",
                    generation_chunks=naive_chunks,
                    retrieved_chunks=naive_chunks,
                    retrieval_metrics=naive_metrics,
                    method_config=naive_config,
                )

            optimized_key = _result_key(
                question_id,
                None,
                "optimized_rag",
            )
            if optimized_key not in existing_keys:
                optimized_where = {
                    "$and": [
                        {"document_id": {"$eq": example["document_id"]}},
                        {"strategy": {"$eq": config["strategy"]}},
                        {"chunk_size": {"$eq": config["chunk_size"]}},
                    ]
                }
                optimized_chunks = get_top_k_chunks(
                    question=example["question"],
                    top_k=config["top_k"],
                    where=optimized_where,
                    retrieval_method=config["retrieval_method"],
                    reranker_enabled=config["reranker_enabled"],
                    reranker_pool_size=config["reranker_pool_size"],
                )
                optimized_alignment = _get_docfinqa_evidence_alignment(
                    example=example,
                    split=BASELINE_SPLIT,
                    where=optimized_where,
                )
                optimized_metrics = evaluate_retrieval(
                    chunks=optimized_chunks,
                    k=config["top_k"],
                    evidence_alignment=optimized_alignment,
                )
                optimized_metrics["reranker_pool_size"] = (
                    config["reranker_pool_size"]
                    if config["reranker_enabled"]
                    else None
                )
                yield build_task(
                    example=example,
                    method_name="optimized_rag",
                    generation_chunks=optimized_chunks,
                    retrieved_chunks=optimized_chunks,
                    retrieval_metrics=optimized_metrics,
                    method_config=config,
                )

    def generate_task(task):
        context_metrics = {}
        if task["method_name"] == "no_context":
            answer = generate_no_context_answer(task["example"]["question"])
        else:
            answer = generate_answer(
                task["example"]["question"],
                task["generation_chunks"],
                generation_context_metrics=context_metrics,
            )
        generation_metrics = get_generation_metrics(
            answer,
            task["example"]["gold_answer"],
            task["method_name"],
        )
        generation_metrics.update(context_metrics)
        if task["method_name"] == "optimized_rag":
            generation_metrics.update(
                {
                    "optimized_config_source": optimized_config_source,
                    "optimized_config_source_split": (
                        saved_optimized_parameters["source_split"]
                    ),
                    "optimized_config_selection_metrics": (
                        saved_optimized_parameters["selection_metrics"]
                    ),
                }
            )
        return answer, generation_metrics

    def handle_generated_result(task, output):
        answer, generation_metrics = output
        is_correct = generation_metrics["docfinqa_answer_correct"]
        method_name = task["method_name"]
        example = task["example"]
        record_method_outcome(
            method_name,
            example["question_id"],
            is_correct,
        )
        method_config = task["method_config"]
        result = {
            "run_id": run_id,
            "result_key": task["result_key"],
            "experiment_name": "baseline",
            "dataset": "docfinqa",
            "split": BASELINE_SPLIT,
            "question_id": example["question_id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "retrieval_method": (
                method_config["retrieval_method"] if method_config else None
            ),
            "chunk_strategy": (
                method_config["strategy"] if method_config else None
            ),
            "chunk_size": (
                method_config["chunk_size"] if method_config else None
            ),
            "top_k": method_config["top_k"] if method_config else None,
            "reranker_used": (
                method_config["reranker_enabled"]
                if method_config
                else False
            ),
            "reranker_pool_size": (
                method_config["reranker_pool_size"]
                if method_config
                and method_config["reranker_enabled"]
                else None
            ),
            "retrieved_chunk_ids": [
                chunk["id"] for chunk in task["retrieved_chunks"]
            ],
            "retrieved_chunks": _compact_retrieved_chunks(
                task["retrieved_chunks"]
            ),
            "retrieval_metrics": task["retrieval_metrics"],
            "generation_metrics": generation_metrics,
        }
        pending_results.append(result)
        if len(pending_results) >= FULL_RAG_RESULT_LOG_BATCH_SIZE:
            flush_pending_results()

    try:
        _run_bounded_generation(
            task_stream(),
            generate_task,
            handle_generated_result,
            max_workers=openai_concurrency,
        )
    finally:
        flush_pending_results()

    manifest_question_ids = set(question_ids)
    outcomes_complete = all(
        set(method_outcomes[method_name]) == manifest_question_ids
        for method_name in baseline_methods
    )
    durable_results_complete = rows_saved == expected_rows
    if not outcomes_complete or not durable_results_complete:
        raise RuntimeError(
            "The baseline run ended without all expected paired outcomes and "
            "remains available for resume."
        )

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
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "question_ids": question_ids,
        }
    )

    questions_processed = len(examples)
    is_full_run = sample_size is None
    is_authoritative = bool(
        sample_size == TEST_SAMPLE_SIZE
        and sample_seed == DEFAULT_SAMPLE_SEED
        and questions_processed == TEST_SAMPLE_SIZE
        and outcomes_complete
        and durable_results_complete
    )
    statistical_analysis_saved = False
    if is_authoritative:
        save_statistical_analysis(
            run_id=run_id,
            dataset="docfinqa",
            model=model,
            source_experiment="baseline",
            source_split=BASELINE_SPLIT,
            analysis_name=STAGE_3_ANALYSIS_NAME,
            analysis=statistical_analysis,
        )
        statistical_analysis_saved = True

    complete_experiment_run(
        run_id=run_id,
        questions_processed=questions_processed,
        rows_saved=rows_saved,
        is_full_run=is_full_run,
        is_authoritative=is_authoritative,
    )

    return {
        "run_id": run_id,
        "is_full_run": is_full_run,
        "is_authoritative": is_authoritative,
        "experiment_name": "baseline",
        "dataset": "docfinqa",
        "split": BASELINE_SPLIT,
        "model": model,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "openai_concurrency": openai_concurrency,
        "question_ids": question_ids,
        "questions_processed": questions_processed,
        "rows_saved": rows_saved,
        "rows_already_saved": len(existing_rows),
        "rows_added_this_attempt": rows_added_this_attempt,
        "optimized_parameters": config,
        "optimized_config_source": optimized_config_source,
        "statistical_analysis": statistical_analysis,
        "statistical_analysis_saved": statistical_analysis_saved,
    }
