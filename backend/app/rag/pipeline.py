from typing import Any
from app.data.data_loader import iter_docfinqa_examples
from app.evaluation import evaluate_docfinqa_answer, evaluate_retrieval
from app.rag.generator import generate_answer
from app.rag.math_agent import math_agent
from app.rag.retriever import get_top_k_chunks
from app.rag.retriever import get_collection
from app.results_logger import log_question_result

def full_rag(
    split: str = "train",
    index: int = 0,
    top_k: int = 3,
    strategy: str = "fixed",
    chunk_size: int = 512,
    overlap: int = 50,
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
            {"split": {"$eq": split}},
            {"source_index": {"$eq": index}},
            {"strategy": {"$eq": strategy}},
            {"chunk_size": {"$eq": chunk_size}},
            {"chunk_overlap": {"$eq": overlap}},
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
        "chunk_overlap": overlap,
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
    split: str = "train",
    start_index: int = 0,
    limit: int = 15,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    chunk_size: int = 512,
    overlap: int = 50,
    log_results: bool = True,
):
    """
    Runs a retrieval-only parameter sweep and logs per-question results.
    """

    top_k_values = top_k_values or [3, 5, 10]
    strategies = strategies or ["fixed", "sentence", "section"]
    retrieval_methods = retrieval_methods or ["keyword", "semantic", "hybrid"]

    results = []

    for source_index, example in enumerate(
        iter_docfinqa_examples(split=split, start_index=start_index, limit=limit),
        start=start_index,
    ):
        for strategy in strategies:
            for retrieval_method in retrieval_methods:
                for top_k in top_k_values:
                    where = {
                        "$and": [
                            {"split": {"$eq": split}},
                            {"source_index": {"$eq": source_index}},
                            {"strategy": {"$eq": strategy}},
                            {"chunk_size": {"$eq": chunk_size}},
                            {"chunk_overlap": {"$eq": overlap}},
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
                        "chunk_overlap": overlap,
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
    split: str = "train",
    start_index: int = 0,
    limit: int = 15,
    top_k_values: list[int] | None = None,
    strategies: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    chunk_size: int = 512,
    overlap: int = 50,
    log_results: bool = True,
):
    """
    Runs full RAG with the DocFinQA math agent instead of GPT generation.
    """

    top_k_values = top_k_values or [3, 5, 10]
    strategies = strategies or ["fixed", "sentence", "section"]
    retrieval_methods = retrieval_methods or ["keyword", "semantic", "hybrid"]

    results = []

    for source_index, example in enumerate(
        iter_docfinqa_examples(split=split, start_index=start_index, limit=limit),
        start=start_index,
    ):
        for strategy in strategies:
            for retrieval_method in retrieval_methods:
                for top_k in top_k_values:
                    where = {
                        "$and": [
                            {"split": {"$eq": split}},
                            {"source_index": {"$eq": source_index}},
                            {"strategy": {"$eq": strategy}},
                            {"chunk_size": {"$eq": chunk_size}},
                            {"chunk_overlap": {"$eq": overlap}},
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
                        "chunk_overlap": overlap,
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
    chunks = get_collection()
    
    #get metrics on each of these
    #no context just question
    no_context = chunks
    for chunk in no_context:
        answer = generate_answer(chunk["question"], "")
        chunk["score"] = evaluate_docfinqa_answer(answer, chunk["Answer"])

    log_question_result(no_context)

    #full document and question
    full_doc = chunks
    for chunk in full_doc:
        answer = generate_answer(chunk["Question"], chunk["Context"])
        chunk["score"] = evaluate_docfinqa_answer(answer, chunk["Answer"])
    
    log_question_result(full_doc)
    
    #navie RAG
    naive_rag = chunks
    for chunk in naive_rag:
        #get chunks with parameters
        source_index = chunk.get("metadata", {}).get("source_index")
        where = {
            "$and": [
                {"split": {"$eq": "train"}},
                {"source_index": {"$eq": source_index}},
                {"strategy": {"$eq": "fixed"}},
                {"chunk_size": {"$eq": 512}},
                {"chunk_overlap": {"$eq": 50}},
            ]
        }

        best_chunks = get_top_k_chunks(
            question=chunk["metadata"]["question"],
            top_k=3,
            collection_name=chunks,
            where=where,
            retrieval_method="semantic"
        )
        #generate answer from question and top chunks
        answer = generate_answer(chunk["Question"], best_chunks)
        #evaluate the generated answer
        chunk["score"] = evaluate_docfinqa_answer(answer, chunk["Answer"])
    
    log_question_result(naive_rag)

    #TBD
    #optimized RAG
    best_rag = chunks
    for chunk in best_rag:
        #get chunks with parameters
        #generate answer from question and top chunks
        #evaluate the generated answer
        z = 0

    log_question_result(best_rag)

    return []
