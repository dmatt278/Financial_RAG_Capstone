from fastapi import FastAPI
from app.rag.pipeline import (
    full_rag,
    full_rag_with_math_agent,
    get_baseline_results,
    top_chunks,
)
from app.rag.vector_store import insert_docfinqa_chunk_sweep, insert_docfinqa_examples
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
    split: str = "train",
    start_index: int = 0,
    limit: int = 10,
    strategy: str = "section",
    chunk_size: int = 512,
    overlap: int = 50,
    full_sweep: bool = True,
    reset: bool = False,
):
    """
    Loads a range of DocFinQA examples into Chroma for vector retrieval.
    """

    if full_sweep:
        return insert_docfinqa_chunk_sweep(
            split=split,
            start_index=start_index,
            limit=limit,
            overlap=overlap,
            reset=reset,
        )

    return insert_docfinqa_examples(
        split=split,
        start_index=start_index,
        limit=limit,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        reset=reset,
    )


@app.get("/run-chunk-rag")
def chunk_rag(
    split: str = "train",
    start_index: int = 0,
    limit: int = 15,
    chunk_size: int = 512,
    overlap: int = 50,
    log_results: bool = True,
):
    '''
    Full parameter sweep for only chunk retrieval. Metrics will be based on the chunks retrieved.
    '''
    return top_chunks(
        split=split,
        start_index=start_index,
        limit=limit,
        chunk_size=chunk_size,
        overlap=overlap,
        log_results=log_results,
    )


@app.get("/run-full-rag")
def full_rag_pipeline(
    split: str = "train",
    index: int = 0,
    top_k: int = 3,
    strategy: str = "fixed",
    chunk_size: int = 512,
    overlap: int = 50,
    retrieval_method: str = "semantic",
    log_result: bool = True,
):
    '''
    Runs the full RAG pipeline for one DocFinQA question. Answer will be generated as well.
    Metrics produced will be based on the generation.
    '''
    return full_rag(
        split=split,
        index=index,
        top_k=top_k,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        retrieval_method=retrieval_method,
        log_result=log_result,
    )


@app.get("/run-full-rag-math-agent")
def full_rag_math_agent_pipeline(
    split: str = "train",
    start_index: int = 0,
    limit: int = 15,
    chunk_size: int = 512,
    overlap: int = 50,
    log_results: bool = True,
):
    '''
    Full parameter sweep on the RAG pipeline using the DocFinQA math agent.
    '''
    return full_rag_with_math_agent(
        split=split,
        start_index=start_index,
        limit=limit,
        chunk_size=chunk_size,
        overlap=overlap,
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

