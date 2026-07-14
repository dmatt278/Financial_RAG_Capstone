
import json
import os
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor


DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@db:5432/financial_rag"


def get_connection():
    """
    Opens a connection to the PostgreSQL database used for experiment results.
    """

    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return psycopg2.connect(database_url)


def create_results_table():
    """
    Creates the rag_results table if it does not already exist.
    """

    query = """
    CREATE TABLE IF NOT EXISTS rag_results (
        id SERIAL PRIMARY KEY,
        experiment_name TEXT,
        dataset TEXT,
        split TEXT,
        question_id TEXT,
        question TEXT,
        gold_answer TEXT,
        generated_answer TEXT,
        is_correct BOOLEAN,
        retrieval_method TEXT,
        chunk_strategy TEXT,
        chunk_size INTEGER,
        chunk_overlap INTEGER,
        top_k INTEGER,
        reranker_used BOOLEAN,
        retrieved_chunk_ids JSONB,
        retrieved_chunks JSONB,
        retrieval_metrics JSONB,
        generation_metrics JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    alter_query = """
    ALTER TABLE rag_results
    ADD COLUMN IF NOT EXISTS split TEXT;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            cursor.execute(alter_query)


def log_question_result(result: dict[str, Any]) -> int:
    """
    Inserts one per-question RAG result into PostgreSQL and returns its row id.
    """

    create_results_table()

    query = """
    INSERT INTO rag_results (
        experiment_name,
        dataset,
        split,
        question_id,
        question,
        gold_answer,
        generated_answer,
        is_correct,
        retrieval_method,
        chunk_strategy,
        chunk_size,
        chunk_overlap,
        top_k,
        reranker_used,
        retrieved_chunk_ids,
        retrieved_chunks,
        retrieval_metrics,
        generation_metrics
    )
    VALUES (
        %(experiment_name)s,
        %(dataset)s,
        %(split)s,
        %(question_id)s,
        %(question)s,
        %(gold_answer)s,
        %(generated_answer)s,
        %(is_correct)s,
        %(retrieval_method)s,
        %(chunk_strategy)s,
        %(chunk_size)s,
        %(chunk_overlap)s,
        %(top_k)s,
        %(reranker_used)s,
        %(retrieved_chunk_ids)s,
        %(retrieved_chunks)s,
        %(retrieval_metrics)s,
        %(generation_metrics)s
    )
    RETURNING id;
    """

    params = {
        "experiment_name": result.get("experiment_name"),
        "dataset": result.get("dataset"),
        "split": result.get("split"),
        "question_id": str(result.get("question_id")),
        "question": result.get("question"),
        "gold_answer": str(result.get("gold_answer")),
        "generated_answer": result.get("generated_answer"),
        "is_correct": result.get("is_correct"),
        "retrieval_method": result.get("retrieval_method"),
        "chunk_strategy": result.get("chunk_strategy"),
        "chunk_size": result.get("chunk_size"),
        "chunk_overlap": result.get("chunk_overlap"),
        "top_k": result.get("top_k"),
        "reranker_used": result.get("reranker_used"),
        "retrieved_chunk_ids": Json(result.get("retrieved_chunk_ids", [])),
        "retrieved_chunks": Json(result.get("retrieved_chunks", [])),
        "retrieval_metrics": Json(result.get("retrieval_metrics", {})),
        "generation_metrics": Json(result.get("generation_metrics", {})),
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()[0]


def get_results(experiment_name: str | None = None, limit: int = 100):
    """
    Gets saved RAG result rows, optionally filtered by experiment name.
    """

    if experiment_name:
        query = """
        SELECT *
        FROM rag_results
        WHERE experiment_name = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """
        params = (experiment_name, limit)
    else:
        query = """
        SELECT *
        FROM rag_results
        ORDER BY created_at DESC
        LIMIT %s;
        """
        params = (limit,)

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    return json.loads(json.dumps(rows, default=str))


def get_experiment_counts(split: str | None = None):
    """
    Counts saved rows for each experiment type.
    """

    create_results_table()

    if split:
        query = """
        SELECT
            experiment_name,
            split,
            COUNT(*) AS rows_saved,
            MIN(created_at) AS first_run_at,
            MAX(created_at) AS last_run_at
        FROM rag_results
        WHERE split = %s
        GROUP BY experiment_name, split
        ORDER BY experiment_name, split;
        """
        params = (split,)
    else:
        query = """
    SELECT
        experiment_name,
        split,
        COUNT(*) AS rows_saved,
        MIN(created_at) AS first_run_at,
        MAX(created_at) AS last_run_at
    FROM rag_results
    GROUP BY experiment_name, split
    ORDER BY experiment_name, split;
    """
        params = ()

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    return json.loads(json.dumps(rows, default=str))


def get_parameter_summary(experiment_name: str, split: str | None = None):
    """
    Aggregates saved results by parameter combination.
    """

    create_results_table()

    if split:
        where_clause = "WHERE experiment_name = %s AND split = %s"
        params = (experiment_name, split)
    else:
        where_clause = "WHERE experiment_name = %s"
        params = (experiment_name,)

    query = f"""
    SELECT
        experiment_name,
        dataset,
        split,
        retrieval_method,
        chunk_strategy,
        chunk_size,
        chunk_overlap,
        top_k,
        reranker_used,
        COUNT(*) AS questions_run,
        AVG(CASE WHEN is_correct THEN 1.0 WHEN is_correct = FALSE THEN 0.0 ELSE NULL END) AS accuracy,
        AVG(NULLIF(retrieval_metrics->>'precision_at_k', '')::float) AS avg_precision_at_k,
        AVG(NULLIF(retrieval_metrics->>'recall_at_k', '')::float) AS avg_recall_at_k,
        AVG(NULLIF(retrieval_metrics->>'reciprocal_rank', '')::float) AS avg_reciprocal_rank,
        AVG(NULLIF(retrieval_metrics->>'relevant_retrieved_count', '')::float) AS avg_relevant_retrieved_count,
        MIN(created_at) AS first_run_at,
        MAX(created_at) AS last_run_at
    FROM rag_results
    {where_clause}
    GROUP BY
        experiment_name,
        dataset,
        split,
        retrieval_method,
        chunk_strategy,
        chunk_size,
        chunk_overlap,
        top_k,
        reranker_used
    ORDER BY
        accuracy DESC NULLS LAST,
        avg_precision_at_k DESC NULLS LAST,
        avg_reciprocal_rank DESC NULLS LAST,
        questions_run DESC;
    """

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    return json.loads(json.dumps(rows, default=str))


def get_final_result_tables(split: str | None = None):
    """
    Returns summary tables for the saved baseline, retrieval, full RAG, and math-agent runs.
    """

    return {
        "split_filter": split,
        "experiment_counts": get_experiment_counts(split=split),
        "baseline_summary": get_parameter_summary("baseline", split=split),
        "chunk_rag_summary": get_parameter_summary("top_chunks_parameter_sweep", split=split),
        "full_rag_summary": get_parameter_summary("full_rag", split=split),
        "full_rag_math_agent_summary": get_parameter_summary("full_rag_with_math_agent", split=split),
    }
