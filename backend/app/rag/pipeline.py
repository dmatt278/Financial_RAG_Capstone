import os
from typing import Any
from app.data.data_loader import BASELINE_SPLIT, TUNING_SPLIT, iter_docfinqa_examples
from app.evaluation import evaluate_docfinqa_answer, evaluate_retrieval
from app.rag.generator import generate_answer
from app.rag.math_agent import math_agent
from app.rag.retriever import get_top_k_chunks
from app.results_logger import log_question_result


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
    retrieval_metrics = evaluate_retrieval(
        chunks=retrieved_chunks,
        k=top_k,
        program=example["program"],
    )
    answer = generate_answer(
        question=example["question"],
        retrieved_chunks=retrieved_chunks,
    )
    is_correct = evaluate_docfinqa_answer(
        generated_answer=answer,
        gold_answer=example["gold_answer"],
    )

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
        "generation_metrics": {
            "docfinqa_answer_correct": is_correct,
        },
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


def top_chunks(
    split: str = TUNING_SPLIT,
    start_index: int = 0,
    limit: int = 15,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    chunk_size: int = 512,
    log_results: bool = True,
):
    """
    Runs a retrieval-only parameter sweep and logs per-question results.
    """

    top_k_values = top_k_values or [3, 5, 10]
    strategies = strategies or ["fixed", "sentence", "section"]
    retrieval_methods = retrieval_methods or ["keyword", "semantic", "hybrid"]

    results = []

    for example in iter_docfinqa_examples(split=split, start_index=start_index, limit=limit):
        for strategy in strategies:
            for retrieval_method in retrieval_methods:
                for top_k in top_k_values:
                    where = {
                        "$and": [
                            {"document_id": {"$eq": example["document_id"]}},
                            {"strategy": {"$eq": strategy}},
                            {"chunk_size": {"$eq": chunk_size}},
                        ]
                    }

                    chunks = get_top_k_chunks(
                        question=example["question"],
                        top_k=top_k,
                        where=where,
                        retrieval_method=retrieval_method,
                    )
                    retrieval_metrics = evaluate_retrieval(
                        chunks=chunks,
                        k=top_k,
                        program=example["program"],
                    )

                    result = {
                        "experiment_name": "top_chunks_parameter_sweep",
                        "dataset": "docfinqa",
                        "split": split,
                        "question_id": example["question_id"],
                        "question": example["question"],
                        "gold_answer": example["gold_answer"],
                        "generated_answer": None,
                        "is_correct": None,
                        "retrieval_method": retrieval_method,
                        "chunk_strategy": strategy,
                        "chunk_size": chunk_size,
                        "top_k": top_k,
                        "reranker_used": False,
                        "retrieved_chunk_ids": [chunk["id"] for chunk in chunks],
                        "retrieved_chunks": chunks,
                        "retrieval_metrics": retrieval_metrics,
                        "generation_metrics": {},
                    }

                    if log_results:
                        result["result_id"] = log_question_result(result)

                    results.append(result)

    return results


def full_rag_with_math_agent(
    split: str = TUNING_SPLIT,
    start_index: int = 0,
    limit: int = 15,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    chunk_size: int = 512,
    log_results: bool = True,
):
    """
    Runs full RAG with the DocFinQA math agent instead of GPT generation.
    """

    top_k_values = top_k_values or [3, 5, 10]
    strategies = strategies or ["fixed", "sentence", "section"]
    retrieval_methods = retrieval_methods or ["keyword", "semantic", "hybrid"]

    results = []

    for example in iter_docfinqa_examples(split=split, start_index=start_index, limit=limit):
        for strategy in strategies:
            for retrieval_method in retrieval_methods:
                for top_k in top_k_values:
                    where = {
                        "$and": [
                            {"document_id": {"$eq": example["document_id"]}},
                            {"strategy": {"$eq": strategy}},
                            {"chunk_size": {"$eq": chunk_size}},
                        ]
                    }

                    chunks = get_top_k_chunks(
                        question=example["question"],
                        top_k=top_k,
                        where=where,
                        retrieval_method=retrieval_method,
                    )
                    retrieval_metrics = evaluate_retrieval(
                        chunks=chunks,
                        k=top_k,
                        program=example["program"],
                    )
                    answer = math_agent(
                        question=example["question"],
                        chunks=chunks,
                        dataset="docfinqa",
                        program=example["program"],
                    )
                    is_correct = evaluate_docfinqa_answer(
                        generated_answer=answer,
                        gold_answer=example["gold_answer"],
                    )

                    result = {
                        "experiment_name": "full_rag_with_math_agent",
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
                        "retrieved_chunk_ids": [chunk["id"] for chunk in chunks],
                        "retrieved_chunks": chunks,
                        "retrieval_metrics": retrieval_metrics,
                        "generation_metrics": {
                            "docfinqa_answer_correct": is_correct,
                            "answer_source": "math_agent",
                        },
                    }

                    if log_results:
                        result["result_id"] = log_question_result(result)

                    results.append(result)

    return results


def get_baseline_results():
    #get metrics on each of these
    #no context just question
    for example in iter_docfinqa_examples(split=BASELINE_SPLIT):
        answer = generate_answer(example["question"], [])
        is_correct = evaluate_docfinqa_answer(answer, example["gold_answer"])

        log_question_result({
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
            "generation_metrics": {
                "baseline": "no_context",
                "docfinqa_answer_correct": is_correct,
            },
        })

    #full document and question
    for example in iter_docfinqa_examples(split=BASELINE_SPLIT):
        full_document_context = [{
            "chunk_id": "full_document",
            "text": example["document_text"],
            "metadata": {"question_id": example["question_id"]},
        }]
        answer = generate_answer(example["question"], full_document_context)
        is_correct = evaluate_docfinqa_answer(answer, example["gold_answer"])

        log_question_result({
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
            "generation_metrics": {
                "baseline": "full_document",
                "docfinqa_answer_correct": is_correct,
            },
        })
    
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
        retrieval_metrics = evaluate_retrieval(
            chunks=best_chunks,
            k=3,
            program=example["program"],
        )
        answer = generate_answer(example["question"], best_chunks)
        is_correct = evaluate_docfinqa_answer(answer, example["gold_answer"])

        log_question_result({
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
            "generation_metrics": {
                "baseline": "naive_rag",
                "docfinqa_answer_correct": is_correct,
            },
        })

    #optimized RAG
    config = OPTIMIZED_RAG_CONFIG
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
        retrieval_metrics = evaluate_retrieval(
            chunks=best_chunks,
            k=config["top_k"],
            program=example["program"],
        )
        retrieval_metrics["reranker_pool_size"] = (
            config["reranker_pool_size"]
            if config["reranker_enabled"]
            else None
        )
        answer = generate_answer(example["question"], best_chunks)
        is_correct = evaluate_docfinqa_answer(answer, example["gold_answer"])

        log_question_result({
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
            "retrieved_chunk_ids": [chunk["id"] for chunk in best_chunks],
            "retrieved_chunks": best_chunks,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": {
                "baseline": "optimized_rag",
                "docfinqa_answer_correct": is_correct,
            },
        })

    return []
