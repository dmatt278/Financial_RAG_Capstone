from fastapi import FastAPI
from typing import Literal
from app.data.data_loader import TRAIN_SPLIT, TUNING_SPLIT, prepare_finqa_files
from app.rag.pipeline import (
    full_rag_shortlist_sweep,
    full_rag_with_math_agent,
    get_baseline_results,
    top_chunks,
)
from app.rag.vector_store import insert_docfinqa_chunk_sweep
from app.results_logger import get_final_result_tables

app = FastAPI(title='Financial RAG Capstone API')

#checks that the API is running
@app.get("/")
def root():
    """
    Returns a basic message confirming that the backend API is running.
    """

    return {
        "message": "Financial RAG backend is running"
    }

#health check
@app.get("/health")
def health_check():
    """
    Returns a lightweight health check status for the backend service.
    """

    return {
        "status": "ok"
    }

@app.post("/load-docfinqa")
def load_docfinqa(
    start_index: int = 0,
    limit: int | None = None,
    reset: bool = False,
):
    """
    Loads and chunks all unique DocFinQA documents into Chroma for vector retrieval.
    """

    return insert_docfinqa_chunk_sweep(
        start_index=start_index,
        limit=limit,
        reset=reset,
    )

@app.post("/load-finqa")
def load_finqa():
    """
    Downloads the original FinQA splits used for gold supporting evidence.
    """

    return prepare_finqa_files()


@app.post("/run-chunk-rag")
def chunk_rag(
    start_index: int = 0,
    limit: int | None = None,
    chunk_size: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
):
    """
    Runs all 324 retrieval configurations on train and saves the top 12.
    """

    return top_chunks(
        split=TRAIN_SPLIT,
        start_index=start_index,
        limit=limit,
        chunk_size=chunk_size,
        log_results=log_results,
        return_results=return_results,
    )


@app.post("/run-full-rag")
def full_rag_pipeline(
    start_index: int = 0,
    limit: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
):
    """
    Runs the saved 12 train retrieval finalists on dev, saves the winner, and
    stores exploratory paired statistical diagnostics.
    """

    return full_rag_shortlist_sweep(
        start_index=start_index,
        limit=limit,
        log_results=log_results,
        return_results=return_results,
    )


@app.post("/run-full-rag-math-agent")
def full_rag_math_agent_pipeline(
    start_index: int = 0,
    limit: int = 15,
    full_dataset: bool = False,
    top_k: int | None = None,
    strategy: Literal["fixed", "sentence", "section"] | None = None,
    chunk_size: int = 512,
    retrieval_method: Literal["keyword", "semantic", "hybrid"] | None = None,
    log_results: bool = True,
    return_results: bool = False,
):
    """
    Separately compares direct LLM math with LLM-planned calculations executed
    by the deterministic DocFinQA math tool. Set full_dataset=true explicitly
    to run beyond the default 15-question diagnostic.
    """

    return full_rag_with_math_agent(
        split=TUNING_SPLIT,
        start_index=start_index,
        limit=None if full_dataset else limit,
        top_k_values=[top_k] if top_k is not None else None,
        strategies=[strategy] if strategy is not None else None,
        retrieval_methods=(
            [retrieval_method]
            if retrieval_method is not None
            else None
        ),
        chunk_size=chunk_size,
        log_results=log_results,
        return_results=return_results,
    )
    

@app.post("/get-baselines")
def get_baselines():
    """
    Runs the test baselines against the frozen optimized pipeline and stores
    confirmatory paired statistical comparisons.
    """
    return get_baseline_results()


@app.get("/final-results")
def final_results(
    split: Literal["train", "dev", "test", "train_dev"] | None = None,
    model: str | None = None,
):
    """
    Returns summary tables for logged experiment results.
    """

    return get_final_result_tables(split=split, model=model)

