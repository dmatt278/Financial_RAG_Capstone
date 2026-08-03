from threading import Lock
from fastapi import FastAPI, HTTPException
from typing import Literal
from app.data.data_loader import (
    DEFAULT_SAMPLE_SEED,
    TRAIN_SPLIT,
    TUNING_SPLIT,
    prepare_finqa_files,
)
from app.rag.pipeline import (
    DEV_SAMPLE_SIZE,
    MATH_SAMPLE_SIZE,
    TEST_SAMPLE_SIZE,
    TRAIN_SAMPLE_SIZE,
    full_rag_shortlist_sweep,
    full_rag_with_math_agent,
    get_baseline_results,
    top_chunks,
)
from app.rag.vector_store import insert_docfinqa_chunk_sweep
from app.results_logger import get_experiment_run, get_final_result_tables

app = FastAPI(title='Financial RAG Capstone API')
_LONG_EXPERIMENT_LOCK = Lock()


def _run_long_experiment(callback):
    if not _LONG_EXPERIMENT_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another long-running experiment is already active.",
        )
    try:
        return callback()
    finally:
        _LONG_EXPERIMENT_LOCK.release()

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

    return _run_long_experiment(
        lambda: insert_docfinqa_chunk_sweep(
            start_index=start_index,
            limit=limit,
            reset=reset,
        )
    )

@app.post("/load-finqa")
def load_finqa():
    """
    Downloads the original FinQA splits used for gold supporting evidence.
    """

    return _run_long_experiment(prepare_finqa_files)


@app.post("/run-chunk-rag")
def chunk_rag(
    start_index: int = 0,
    limit: int | None = None,
    chunk_size: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
    sample_size: int | None = TRAIN_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """
    Runs all 324 configurations on 50 seeded train questions and saves 3.
    """

    return _run_long_experiment(
        lambda: top_chunks(
            split=TRAIN_SPLIT,
            start_index=start_index,
            limit=limit,
            chunk_size=chunk_size,
            log_results=log_results,
            return_results=return_results,
            sample_size=sample_size,
            sample_seed=sample_seed,
            resume_run_id=resume_run_id,
        )
    )


@app.post("/run-full-rag")
def full_rag_pipeline(
    start_index: int = 0,
    limit: int | None = None,
    log_results: bool = True,
    return_results: bool = False,
    sample_size: int | None = DEV_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """
    Runs the saved 3 train retrieval finalists on 100 dev questions and
    stores exploratory paired statistical diagnostics.
    """

    return _run_long_experiment(
        lambda: full_rag_shortlist_sweep(
            start_index=start_index,
            limit=limit,
            log_results=log_results,
            return_results=return_results,
            sample_size=sample_size,
            sample_seed=sample_seed,
            resume_run_id=resume_run_id,
        )
    )


@app.post("/run-full-rag-math-agent")
def full_rag_math_agent_pipeline(
    start_index: int = 0,
    limit: int | None = None,
    top_k: int = 5,
    strategy: Literal["fixed", "sentence", "section"] = "section",
    chunk_size: int = 512,
    retrieval_method: Literal["keyword", "semantic", "hybrid"] = "hybrid",
    log_results: bool = True,
    return_results: bool = False,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """
    Separately compares direct LLM math with LLM-planned calculations executed
    by the deterministic DocFinQA math tool on 50 seeded questions.
    """

    return _run_long_experiment(
        lambda: full_rag_with_math_agent(
            split=TUNING_SPLIT,
            start_index=start_index,
            limit=limit,
            top_k_values=[top_k],
            strategies=[strategy],
            retrieval_methods=[retrieval_method],
            chunk_size=chunk_size,
            log_results=log_results,
            return_results=return_results,
            sample_size=MATH_SAMPLE_SIZE,
            sample_seed=sample_seed,
            resume_run_id=resume_run_id,
        )
    )
    

@app.post("/get-baselines")
def get_baselines(
    sample_size: int | None = TEST_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume_run_id: str | None = None,
):
    """
    Runs the test baselines against the frozen optimized pipeline and stores
    confirmatory paired statistical comparisons.
    """
    return _run_long_experiment(
        lambda: get_baseline_results(
            sample_size=sample_size,
            sample_seed=sample_seed,
            resume_run_id=resume_run_id,
        )
    )


@app.get("/experiment-status/{run_id}")
def experiment_status(run_id: str):
    """Returns durable checkpoint progress for one experiment run."""

    run = get_experiment_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Experiment run not found.")
    return run


@app.get("/final-results")
def final_results(
    split: Literal["train", "dev", "test", "train_dev"] | None = None,
    model: str | None = None,
):
    """
    Returns summary tables for logged experiment results.
    """

    return get_final_result_tables(split=split, model=model)

