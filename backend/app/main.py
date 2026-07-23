from fastapi import FastAPI
from app.data.data_loader import TUNING_SPLIT
from app.rag.pipeline import (
    full_rag,
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


@app.get("/run-chunk-rag")
def chunk_rag(
    start_index: int = 0,
    limit: int = 15,
    chunk_size: int = 512,
    log_results: bool = True,
):
    '''
    Full parameter sweep for only chunk retrieval. Metrics will be based on the chunks retrieved.
    '''
    return top_chunks(
        split=TUNING_SPLIT,
        start_index=start_index,
        limit=limit,
        chunk_size=chunk_size,
        log_results=log_results,
    )


@app.get("/run-full-rag")
def full_rag_pipeline(
    index: int = 0,
    top_k: int = 3,
    strategy: str = "fixed",
    chunk_size: int = 512,
    retrieval_method: str = "semantic",
    log_result: bool = True,
):
    '''
    Runs the full RAG pipeline for one train/dev DocFinQA question.
    '''
    return full_rag(
        split=TUNING_SPLIT,
        index=index,
        top_k=top_k,
        strategy=strategy,
        chunk_size=chunk_size,
        retrieval_method=retrieval_method,
        log_result=log_result,
    )


@app.get("/run-full-rag-math-agent")
def full_rag_math_agent_pipeline(
    start_index: int = 0,
    limit: int = 15,
    chunk_size: int = 512,
    log_results: bool = True,
):
    '''
    Full parameter sweep on the RAG pipeline using the DocFinQA math agent.
    '''
    return full_rag_with_math_agent(
        split=TUNING_SPLIT,
        start_index=start_index,
        limit=limit,
        chunk_size=chunk_size,
        log_results=log_results,
    )
    

@app.get("/get-baselines")
def get_baselines():
    """
    Runs the baseline RAG configurations for comparison against the optimized pipeline.
    Returns retrieval and answer-generation metrics for each baseline setup.
    """
    return get_baseline_results()


@app.get("/final-results")
def final_results(split: str | None = None):
    """
    Returns summary tables for logged experiment results.
    """

    return get_final_result_tables(split=split)

