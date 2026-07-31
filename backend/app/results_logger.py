
import json
import os
from decimal import Decimal
from threading import Lock
from typing import Any
from uuid import uuid4
import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values


DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@db:5432/financial_rag"
MATH_AGENT_EXPERIMENT = "full_rag_math_tool_agent"
_SCHEMA_READY = False
_SCHEMA_LOCK = Lock()


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _json_compatible(value):
    return json.loads(json.dumps(value, default=_json_default))


def get_connection():
    """
    Opens a connection to the PostgreSQL database used for experiment results.
    """

    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return psycopg2.connect(database_url)


def create_results_table():
    """
    Creates the result and parameter-selection tables if they do not exist.
    """

    global _SCHEMA_READY

    if _SCHEMA_READY:
        return

    query = """
    CREATE TABLE IF NOT EXISTS rag_results (
        id SERIAL PRIMARY KEY,
        run_id TEXT,
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
        top_k INTEGER,
        reranker_used BOOLEAN,
        reranker_pool_size INTEGER,
        retrieved_chunk_ids JSONB,
        retrieved_chunks JSONB,
        retrieval_metrics JSONB,
        generation_metrics JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    alter_query = """
    ALTER TABLE rag_results
    ADD COLUMN IF NOT EXISTS split TEXT,
    ADD COLUMN IF NOT EXISTS reranker_pool_size INTEGER,
    ADD COLUMN IF NOT EXISTS run_id TEXT;
    """

    experiment_runs_query = """
    CREATE TABLE IF NOT EXISTS rag_experiment_runs (
        run_id TEXT PRIMARY KEY,
        experiment_name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        split TEXT NOT NULL,
        model TEXT,
        status TEXT NOT NULL CHECK (status IN ('running', 'completed')),
        is_full_run BOOLEAN NOT NULL DEFAULT FALSE,
        questions_processed INTEGER NOT NULL DEFAULT 0,
        rows_saved BIGINT NOT NULL DEFAULT 0,
        parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
        started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMPTZ
    );
    """

    best_parameters_query = """
    CREATE TABLE IF NOT EXISTS rag_best_parameters (
        run_id TEXT,
        dataset TEXT NOT NULL,
        model TEXT NOT NULL,
        source_experiment TEXT NOT NULL,
        source_split TEXT NOT NULL,
        retrieval_method TEXT NOT NULL,
        chunk_strategy TEXT NOT NULL,
        chunk_size INTEGER NOT NULL,
        top_k INTEGER NOT NULL,
        reranker_used BOOLEAN NOT NULL,
        reranker_pool_size INTEGER,
        questions_run INTEGER NOT NULL,
        selection_metrics JSONB NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (dataset, model, source_split)
    );
    """

    best_parameters_index_query = """
    CREATE UNIQUE INDEX IF NOT EXISTS
        rag_best_parameters_dataset_model_split_idx
    ON rag_best_parameters (dataset, model, source_split);
    """

    retrieval_shortlist_query = """
    CREATE TABLE IF NOT EXISTS rag_retrieval_shortlists (
        run_id TEXT,
        dataset TEXT NOT NULL,
        source_experiment TEXT NOT NULL,
        source_split TEXT NOT NULL,
        questions_run INTEGER NOT NULL,
        ranking_rule JSONB NOT NULL,
        candidates JSONB NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (dataset, source_experiment, source_split)
    );
    """

    statistical_analyses_query = """
    CREATE TABLE IF NOT EXISTS rag_statistical_analyses (
        run_id TEXT,
        dataset TEXT NOT NULL,
        model TEXT NOT NULL,
        source_experiment TEXT NOT NULL,
        source_split TEXT NOT NULL,
        analysis_name TEXT NOT NULL,
        questions_run INTEGER NOT NULL,
        analysis JSONB NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (
            dataset,
            model,
            source_experiment,
            source_split,
            analysis_name
        )
    );
    """

    artifact_alter_query = """
    ALTER TABLE rag_best_parameters
    ADD COLUMN IF NOT EXISTS run_id TEXT;

    ALTER TABLE rag_retrieval_shortlists
    ADD COLUMN IF NOT EXISTS run_id TEXT;

    ALTER TABLE rag_statistical_analyses
    ADD COLUMN IF NOT EXISTS run_id TEXT;
    """

    reporting_indexes_query = """
    CREATE INDEX IF NOT EXISTS rag_results_run_experiment_split_idx
    ON rag_results (run_id, experiment_name, split);

    CREATE INDEX IF NOT EXISTS rag_experiment_runs_latest_completed_idx
    ON rag_experiment_runs (
        experiment_name,
        split,
        model,
        completed_at DESC
    )
    WHERE status = 'completed' AND is_full_run = TRUE;
    """

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                cursor.execute(alter_query)
                cursor.execute(experiment_runs_query)
                cursor.execute(best_parameters_query)
                cursor.execute(best_parameters_index_query)
                cursor.execute(retrieval_shortlist_query)
                cursor.execute(statistical_analyses_query)
                cursor.execute(artifact_alter_query)
                cursor.execute(reporting_indexes_query)

        _SCHEMA_READY = True


def start_experiment_run(
    *,
    experiment_name: str,
    dataset: str,
    split: str,
    model: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Creates one run record before result rows are written."""

    create_results_table()
    run_id = str(uuid4())
    query = """
    INSERT INTO rag_experiment_runs (
        run_id,
        experiment_name,
        dataset,
        split,
        model,
        status,
        parameters
    )
    VALUES (
        %(run_id)s,
        %(experiment_name)s,
        %(dataset)s,
        %(split)s,
        %(model)s,
        'running',
        %(parameters)s
    );
    """
    params = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "dataset": dataset,
        "split": split,
        "model": model,
        "parameters": Json(parameters or {}),
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)

    return run_id


def complete_experiment_run(
    *,
    run_id: str,
    questions_processed: int,
    rows_saved: int,
    is_full_run: bool,
) -> None:
    """Marks a normally finished run complete for final-results reporting."""

    create_results_table()
    query = """
    UPDATE rag_experiment_runs
    SET
        status = 'completed',
        is_full_run = %(is_full_run)s,
        questions_processed = %(questions_processed)s,
        rows_saved = %(rows_saved)s,
        completed_at = CURRENT_TIMESTAMP
    WHERE run_id = %(run_id)s
      AND status = 'running';
    """
    params = {
        "run_id": run_id,
        "questions_processed": questions_processed,
        "rows_saved": rows_saved,
        "is_full_run": is_full_run,
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Experiment run '{run_id}' was not in the running state."
                )


def save_retrieval_shortlist(
    dataset: str,
    source_experiment: str,
    source_split: str,
    questions_run: int,
    ranking_rule: dict[str, Any],
    candidates: list[dict[str, Any]],
    run_id: str,
):
    """Upserts the completed retrieval sweep's ranked shortlist."""

    create_results_table()
    query = """
    INSERT INTO rag_retrieval_shortlists (
        run_id,
        dataset,
        source_experiment,
        source_split,
        questions_run,
        ranking_rule,
        candidates
    )
    VALUES (
        %(run_id)s,
        %(dataset)s,
        %(source_experiment)s,
        %(source_split)s,
        %(questions_run)s,
        %(ranking_rule)s,
        %(candidates)s
    )
    ON CONFLICT (dataset, source_experiment, source_split)
    DO UPDATE SET
        run_id = EXCLUDED.run_id,
        questions_run = EXCLUDED.questions_run,
        ranking_rule = EXCLUDED.ranking_rule,
        candidates = EXCLUDED.candidates,
        updated_at = CURRENT_TIMESTAMP
    RETURNING *;
    """
    params = {
        "run_id": run_id,
        "dataset": dataset,
        "source_experiment": source_experiment,
        "source_split": source_split,
        "questions_run": questions_run,
        "ranking_rule": Json(ranking_rule),
        "candidates": Json(candidates),
    }

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

    return _json_compatible(row)


def get_retrieval_shortlist(
    dataset: str,
    source_experiment: str,
    source_split: str,
):
    """Returns the saved complete retrieval shortlist, if one exists."""

    create_results_table()
    query = """
    SELECT shortlist.*
    FROM rag_retrieval_shortlists AS shortlist
    JOIN rag_experiment_runs AS run
      ON run.run_id = shortlist.run_id
     AND run.status = 'completed'
     AND run.is_full_run = TRUE
    WHERE shortlist.dataset = %s
      AND shortlist.source_experiment = %s
      AND shortlist.source_split = %s;
    """

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                query,
                (dataset, source_experiment, source_split),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    return _json_compatible(row)


def save_statistical_analysis(
    dataset: str,
    model: str,
    source_experiment: str,
    source_split: str,
    analysis_name: str,
    analysis: dict[str, Any],
    run_id: str,
):
    """Upserts one completed run's paired statistical analysis."""

    create_results_table()
    query = """
    INSERT INTO rag_statistical_analyses (
        run_id,
        dataset,
        model,
        source_experiment,
        source_split,
        analysis_name,
        questions_run,
        analysis
    )
    VALUES (
        %(run_id)s,
        %(dataset)s,
        %(model)s,
        %(source_experiment)s,
        %(source_split)s,
        %(analysis_name)s,
        %(questions_run)s,
        %(analysis)s
    )
    ON CONFLICT (
        dataset,
        model,
        source_experiment,
        source_split,
        analysis_name
    )
    DO UPDATE SET
        run_id = EXCLUDED.run_id,
        questions_run = EXCLUDED.questions_run,
        analysis = EXCLUDED.analysis,
        updated_at = CURRENT_TIMESTAMP
    RETURNING *;
    """
    params = {
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "source_experiment": source_experiment,
        "source_split": source_split,
        "analysis_name": analysis_name,
        "questions_run": analysis["questions_run"],
        "analysis": Json(analysis),
    }

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

    return _json_compatible(row)


def get_latest_completed_runs(
    split: str | None = None,
    model: str | None = None,
    ensure_table: bool = True,
):
    """Returns the newest completed full run per experiment, split, and model."""

    if ensure_table:
        create_results_table()

    filters = ["status = 'completed'", "is_full_run = TRUE"]
    params = []
    if split is not None:
        filters.append("split = %s")
        params.append(split)
    if model is not None:
        filters.append("(model = %s OR model IS NULL)")
        params.append(model)

    query = f"""
    SELECT DISTINCT ON (
        experiment_name,
        dataset,
        split,
        COALESCE(model, '')
    )
        run_id,
        experiment_name,
        dataset,
        split,
        model,
        status,
        is_full_run,
        questions_processed,
        rows_saved,
        parameters,
        started_at,
        completed_at
    FROM rag_experiment_runs
    WHERE {' AND '.join(filters)}
    ORDER BY
        experiment_name,
        dataset,
        split,
        COALESCE(model, ''),
        completed_at DESC,
        run_id DESC;
    """

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    return _json_compatible(rows)


def get_latest_completed_math_agent_run(
    split: str | None = None,
    model: str | None = None,
    ensure_table: bool = True,
):
    """Returns the latest completed math-agent run, including a partial run."""

    if ensure_table:
        create_results_table()

    filters = [
        "status = 'completed'",
        "experiment_name = %s",
    ]
    params = [MATH_AGENT_EXPERIMENT]
    if split is not None:
        filters.append("split = %s")
        params.append(split)
    if model is not None:
        filters.append("(model = %s OR model IS NULL)")
        params.append(model)

    query = f"""
    SELECT
        run_id,
        experiment_name,
        dataset,
        split,
        model,
        status,
        is_full_run,
        questions_processed,
        rows_saved,
        parameters,
        started_at,
        completed_at
    FROM rag_experiment_runs
    WHERE {' AND '.join(filters)}
    ORDER BY completed_at DESC, run_id DESC
    LIMIT 1;
    """
    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            row = cursor.fetchone()

    return _json_compatible(row) if row is not None else None


def get_statistical_analyses(
    split: str | None = None,
    model: str | None = None,
    run_ids: list[str] | None = None,
    ensure_table: bool = True,
):
    """Returns analyses attached to the selected completed full runs."""

    if ensure_table:
        create_results_table()
    if run_ids is None:
        run_ids = [
            run["run_id"]
            for run in get_latest_completed_runs(
                split=split,
                model=model,
                ensure_table=False,
            )
        ]
    if not run_ids:
        return []

    query = """
    SELECT statistical.*
    FROM rag_statistical_analyses AS statistical
    JOIN rag_experiment_runs AS run
      ON run.run_id = statistical.run_id
     AND run.status = 'completed'
     AND run.is_full_run = TRUE
    WHERE statistical.run_id = ANY(%s)
    ORDER BY
        statistical.source_split,
        statistical.analysis_name,
        statistical.updated_at DESC;
    """

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (run_ids,))
            rows = cursor.fetchall()

    return _json_compatible(rows)


def save_best_rag_parameters(
    dataset: str,
    model: str,
    source_experiment: str,
    source_split: str,
    config: dict[str, Any],
    selection_metrics: dict[str, Any],
    run_id: str,
):
    """Upserts the best completed-sweep configuration for a dataset and model."""

    create_results_table()
    query = """
    INSERT INTO rag_best_parameters (
        run_id,
        dataset,
        model,
        source_experiment,
        source_split,
        retrieval_method,
        chunk_strategy,
        chunk_size,
        top_k,
        reranker_used,
        reranker_pool_size,
        questions_run,
        selection_metrics
    )
    VALUES (
        %(run_id)s,
        %(dataset)s,
        %(model)s,
        %(source_experiment)s,
        %(source_split)s,
        %(retrieval_method)s,
        %(chunk_strategy)s,
        %(chunk_size)s,
        %(top_k)s,
        %(reranker_used)s,
        %(reranker_pool_size)s,
        %(questions_run)s,
        %(selection_metrics)s
    )
    ON CONFLICT (dataset, model, source_split)
    DO UPDATE SET
        run_id = EXCLUDED.run_id,
        source_experiment = EXCLUDED.source_experiment,
        source_split = EXCLUDED.source_split,
        retrieval_method = EXCLUDED.retrieval_method,
        chunk_strategy = EXCLUDED.chunk_strategy,
        chunk_size = EXCLUDED.chunk_size,
        top_k = EXCLUDED.top_k,
        reranker_used = EXCLUDED.reranker_used,
        reranker_pool_size = EXCLUDED.reranker_pool_size,
        questions_run = EXCLUDED.questions_run,
        selection_metrics = EXCLUDED.selection_metrics,
        updated_at = CURRENT_TIMESTAMP
    RETURNING *;
    """
    params = {
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "source_experiment": source_experiment,
        "source_split": source_split,
        "retrieval_method": config["retrieval_method"],
        "chunk_strategy": config["strategy"],
        "chunk_size": config["chunk_size"],
        "top_k": config["top_k"],
        "reranker_used": config["reranker_enabled"],
        "reranker_pool_size": config.get("reranker_pool_size"),
        "questions_run": selection_metrics["questions_run"],
        "selection_metrics": Json(selection_metrics),
    }

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

    return _json_compatible(row)


def get_best_rag_parameters(dataset: str, model: str, source_split: str):
    """Returns the saved best configuration for a dataset and generation model."""

    create_results_table()
    query = """
    SELECT parameters.*
    FROM rag_best_parameters AS parameters
    JOIN rag_experiment_runs AS run
      ON run.run_id = parameters.run_id
     AND run.status = 'completed'
     AND run.is_full_run = TRUE
    WHERE parameters.dataset = %s
      AND parameters.model = %s
      AND parameters.source_split = %s;
    """

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (dataset, model, source_split))
            row = cursor.fetchone()

    if row is None:
        return None

    return _json_compatible(row)


def get_retrieval_shortlists(
    *,
    split: str | None = None,
    run_ids: list[str] | None = None,
    ensure_table: bool = True,
):
    """Returns authoritative retrieval shortlists for selected runs."""

    if ensure_table:
        create_results_table()
    if run_ids is None:
        run_ids = [
            run["run_id"]
            for run in get_latest_completed_runs(
                split=split,
                ensure_table=False,
            )
        ]
    if not run_ids:
        return []

    query = """
    SELECT shortlist.*
    FROM rag_retrieval_shortlists AS shortlist
    JOIN rag_experiment_runs AS run
      ON run.run_id = shortlist.run_id
     AND run.status = 'completed'
     AND run.is_full_run = TRUE
    WHERE shortlist.run_id = ANY(%s)
    ORDER BY shortlist.source_split, shortlist.updated_at DESC;
    """
    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (run_ids,))
            rows = cursor.fetchall()

    return _json_compatible(rows)


def get_best_rag_parameter_rows(
    *,
    split: str | None = None,
    model: str | None = None,
    run_ids: list[str] | None = None,
    ensure_table: bool = True,
):
    """Returns authoritative saved winners for selected runs."""

    if ensure_table:
        create_results_table()
    if run_ids is None:
        run_ids = [
            run["run_id"]
            for run in get_latest_completed_runs(
                split=split,
                model=model,
                ensure_table=False,
            )
        ]
    if not run_ids:
        return []

    query = """
    SELECT parameters.*
    FROM rag_best_parameters AS parameters
    JOIN rag_experiment_runs AS run
      ON run.run_id = parameters.run_id
     AND run.status = 'completed'
     AND run.is_full_run = TRUE
    WHERE parameters.run_id = ANY(%s)
    ORDER BY parameters.source_split, parameters.model, parameters.updated_at DESC;
    """
    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (run_ids,))
            rows = cursor.fetchall()

    return _json_compatible(rows)


def log_question_result(result: dict[str, Any]) -> int:
    """
    Inserts one per-question RAG result into PostgreSQL and returns its row id.
    """

    create_results_table()

    query = """
    INSERT INTO rag_results (
        run_id,
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
        top_k,
        reranker_used,
        reranker_pool_size,
        retrieved_chunk_ids,
        retrieved_chunks,
        retrieval_metrics,
        generation_metrics
    )
    VALUES (
        %(run_id)s,
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
        %(top_k)s,
        %(reranker_used)s,
        %(reranker_pool_size)s,
        %(retrieved_chunk_ids)s,
        %(retrieved_chunks)s,
        %(retrieval_metrics)s,
        %(generation_metrics)s
    )
    RETURNING id;
    """

    params = {
        "run_id": result.get("run_id"),
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
        "top_k": result.get("top_k"),
        "reranker_used": result.get("reranker_used"),
        "reranker_pool_size": result.get("reranker_pool_size"),
        "retrieved_chunk_ids": Json(result.get("retrieved_chunk_ids", [])),
        "retrieved_chunks": Json(result.get("retrieved_chunks", [])),
        "retrieval_metrics": Json(result.get("retrieval_metrics", {})),
        "generation_metrics": Json(result.get("generation_metrics", {})),
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()[0]


def log_question_results(
    results: list[dict[str, Any]],
    ensure_table: bool = True,
) -> list[int]:
    """
    Inserts a batch of result rows and returns their PostgreSQL ids.
    """

    if not results:
        return []

    if ensure_table:
        create_results_table()

    query = """
    INSERT INTO rag_results (
        run_id,
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
        top_k,
        reranker_used,
        reranker_pool_size,
        retrieved_chunk_ids,
        retrieved_chunks,
        retrieval_metrics,
        generation_metrics
    )
    VALUES %s
    RETURNING id;
    """

    values = [
        (
            result.get("run_id"),
            result.get("experiment_name"),
            result.get("dataset"),
            result.get("split"),
            str(result.get("question_id")),
            result.get("question"),
            str(result.get("gold_answer")),
            result.get("generated_answer"),
            result.get("is_correct"),
            result.get("retrieval_method"),
            result.get("chunk_strategy"),
            result.get("chunk_size"),
            result.get("top_k"),
            result.get("reranker_used"),
            result.get("reranker_pool_size"),
            Json(result.get("retrieved_chunk_ids", [])),
            Json(result.get("retrieved_chunks", [])),
            Json(result.get("retrieval_metrics", {})),
            Json(result.get("generation_metrics", {})),
        )
        for result in results
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(
                cursor,
                query,
                values,
                page_size=len(values),
            )
            return [row[0] for row in cursor.fetchall()]


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

    return _json_compatible(rows)


def get_experiment_counts(
    split: str | None = None,
    model: str | None = None,
    run_ids: list[str] | None = None,
    ensure_table: bool = True,
):
    """Returns row and question counts for the selected completed runs."""

    if ensure_table:
        create_results_table()
    if run_ids is None:
        run_ids = [
            run["run_id"]
            for run in get_latest_completed_runs(
                split=split,
                model=model,
                ensure_table=False,
            )
        ]
    if not run_ids:
        return []

    query = """
    SELECT
        run_id,
        experiment_name,
        dataset,
        split,
        model,
        questions_processed AS questions_run,
        rows_saved,
        started_at AS first_run_at,
        completed_at AS last_run_at
    FROM rag_experiment_runs
    WHERE run_id = ANY(%s)
    ORDER BY experiment_name, split, model NULLS FIRST;
    """
    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (run_ids,))
            rows = cursor.fetchall()

    return _json_compatible(rows)


def get_parameter_summary(
    experiment_name: str,
    split: str | None = None,
    model: str | None = None,
    run_ids: list[str] | None = None,
    ensure_table: bool = True,
    include_partial_runs: bool = False,
):
    """Aggregates one experiment from selected completed runs."""

    if ensure_table:
        create_results_table()
    if run_ids is None:
        run_ids = [
            run["run_id"]
            for run in get_latest_completed_runs(
                split=split,
                model=model,
                ensure_table=False,
            )
        ]
    if not run_ids:
        return []

    full_run_filter = (
        ""
        if include_partial_runs
        else "AND run.is_full_run = TRUE"
    )

    if experiment_name == "top_chunks_evidence_sweep":
        order_clause = """
        retrieval_selection_score DESC NULLS LAST,
        avg_all_evidence_hit_at_k DESC NULLS LAST,
        avg_evidence_recall_at_k DESC NULLS LAST,
        avg_reciprocal_rank DESC NULLS LAST,
        avg_precision_at_k DESC NULLS LAST,
        result.reranker_used ASC NULLS FIRST,
        COALESCE(result.reranker_pool_size, 0) ASC,
        result.top_k ASC,
        result.chunk_size ASC,
        result.retrieval_method ASC,
        result.chunk_strategy ASC,
        comparison_method ASC NULLS FIRST
        """
    else:
        order_clause = """
        accuracy DESC NULLS LAST,
        within_one_percent_rate DESC NULLS LAST,
        avg_all_evidence_hit_at_k DESC NULLS LAST,
        avg_evidence_recall_at_k DESC NULLS LAST,
        avg_reciprocal_rank DESC NULLS LAST,
        avg_precision_at_k DESC NULLS LAST,
        result.reranker_used ASC NULLS FIRST,
        COALESCE(result.reranker_pool_size, 0) ASC,
        result.top_k ASC NULLS FIRST,
        result.chunk_size ASC NULLS FIRST,
        result.retrieval_method ASC NULLS FIRST,
        result.chunk_strategy ASC NULLS FIRST,
        comparison_method ASC NULLS FIRST
        """

    query = f"""
    SELECT
        result.run_id,
        result.experiment_name,
        result.dataset,
        result.split,
        run.model,
        result.generation_metrics->>'baseline' AS baseline_type,
        result.generation_metrics->>'comparison_method' AS comparison_method,
        result.retrieval_method,
        result.chunk_strategy,
        result.chunk_size,
        result.top_k,
        result.reranker_used,
        result.reranker_pool_size,
        COUNT(*) AS questions_run,
        COUNT(*) AS rows_saved,
        AVG(
            CASE
                WHEN result.is_correct THEN 1.0
                WHEN result.is_correct = FALSE THEN 0.0
                ELSE NULL
            END
        )::double precision AS accuracy,
        AVG(NULLIF(result.retrieval_metrics->>'precision_at_k', '')::double precision) AS avg_precision_at_k,
        AVG(NULLIF(result.retrieval_metrics->>'recall_at_k', '')::double precision) AS avg_recall_at_k,
        AVG(NULLIF(result.retrieval_metrics->>'reciprocal_rank', '')::double precision) AS avg_reciprocal_rank,
        AVG(NULLIF(result.retrieval_metrics->>'relevant_retrieved_count', '')::double precision) AS avg_relevant_retrieved_count,
        AVG(NULLIF(result.retrieval_metrics->>'retrieved_count_at_k', '')::double precision) AS avg_retrieved_count_at_k,
        AVG(NULLIF(result.retrieval_metrics->>'hit_rate_at_k', '')::double precision) AS avg_hit_rate_at_k,
        AVG(NULLIF(result.retrieval_metrics->>'all_evidence_hit_at_k', '')::double precision) AS avg_all_evidence_hit_at_k,
        AVG(NULLIF(result.retrieval_metrics->>'evidence_recall_at_k', '')::double precision) AS avg_evidence_recall_at_k,
        AVG(NULLIF(result.retrieval_metrics->>'average_alignment_score', '')::double precision) AS avg_evidence_alignment_score,
        (
            0.50 * COALESCE(
                AVG(NULLIF(result.retrieval_metrics->>'all_evidence_hit_at_k', '')::double precision),
                0.0
            )
            + 0.30 * COALESCE(
                AVG(NULLIF(result.retrieval_metrics->>'evidence_recall_at_k', '')::double precision),
                0.0
            )
            + 0.10 * COALESCE(
                AVG(NULLIF(result.retrieval_metrics->>'reciprocal_rank', '')::double precision),
                0.0
            )
            + 0.10 * COALESCE(
                AVG(NULLIF(result.retrieval_metrics->>'precision_at_k', '')::double precision),
                0.0
            )
        )::double precision AS retrieval_selection_score,
        AVG(NULLIF(result.generation_metrics->>'absolute_error', '')::double precision) AS avg_absolute_error,
        AVG(NULLIF(result.generation_metrics->>'relative_error', '')::double precision) AS avg_relative_error,
        AVG(
            CASE
                WHEN result.generation_metrics->>'answer_parse_succeeded' = 'true' THEN 1.0
                WHEN result.generation_metrics->>'answer_parse_succeeded' = 'false' THEN 0.0
                ELSE NULL
            END
        )::double precision AS answer_parse_rate,
        AVG(
            CASE
                WHEN result.generation_metrics->>'within_one_percent' = 'true' THEN 1.0
                WHEN result.generation_metrics->>'within_one_percent' = 'false' THEN 0.0
                ELSE NULL
            END
        )::double precision AS within_one_percent_rate,
        AVG(
            CASE
                WHEN result.generation_metrics->>'exact_normalized_match' = 'true' THEN 1.0
                WHEN result.generation_metrics->>'exact_normalized_match' = 'false' THEN 0.0
                ELSE NULL
            END
        )::double precision AS exact_normalized_match_rate,
        AVG(
            NULLIF(
                result.generation_metrics->>'generation_context_chunk_count',
                ''
            )::double precision
        ) AS avg_generation_context_chunk_count,
        AVG(
            CASE
                WHEN result.generation_metrics->>'prompt_was_truncated' = 'true' THEN 1.0
                WHEN result.generation_metrics->>'prompt_was_truncated' = 'false' THEN 0.0
                ELSE NULL
            END
        )::double precision AS prompt_truncation_rate,
        AVG(
            CASE
                WHEN result.generation_metrics->>'program_parse_succeeded' = 'true' THEN 1.0
                WHEN result.generation_metrics->>'program_parse_succeeded' = 'false' THEN 0.0
                ELSE NULL
            END
        )::double precision AS math_program_parse_rate,
        AVG(
            CASE
                WHEN result.generation_metrics->>'execution_succeeded' = 'true' THEN 1.0
                WHEN result.generation_metrics->>'execution_succeeded' = 'false' THEN 0.0
                ELSE NULL
            END
        )::double precision AS math_execution_success_rate,
        AVG(
            NULLIF(
                result.generation_metrics->>'operand_grounding_rate',
                ''
            )::double precision
        ) AS avg_operand_grounding_rate,
        MIN(result.created_at) AS first_run_at,
        MAX(result.created_at) AS last_run_at,
        run.completed_at
    FROM rag_results AS result
    JOIN rag_experiment_runs AS run
      ON run.run_id = result.run_id
     AND run.status = 'completed'
     {full_run_filter}
    WHERE result.experiment_name = %s
      AND result.run_id = ANY(%s)
    GROUP BY
        result.run_id,
        result.experiment_name,
        result.dataset,
        result.split,
        run.model,
        result.generation_metrics->>'baseline',
        result.generation_metrics->>'comparison_method',
        result.retrieval_method,
        result.chunk_strategy,
        result.chunk_size,
        result.top_k,
        result.reranker_used,
        result.reranker_pool_size,
        run.completed_at
    ORDER BY {order_clause};
    """

    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (experiment_name, run_ids))
            rows = cursor.fetchall()

    return _json_compatible(rows)


def get_final_result_tables(
    split: str | None = None,
    model: str | None = None,
):
    """Returns authoritative summaries from the latest completed full runs."""

    create_results_table()
    selected_runs = get_latest_completed_runs(
        split=split,
        model=model,
        ensure_table=False,
    )
    run_ids = [run["run_id"] for run in selected_runs]
    latest_math_agent_run = get_latest_completed_math_agent_run(
        split=split,
        model=model,
        ensure_table=False,
    )
    exploratory_math_agent = None
    if (
        latest_math_agent_run is not None
        and not latest_math_agent_run["is_full_run"]
    ):
        exploratory_math_agent = {
            "reporting_role": "exploratory_partial_run",
            "authoritative": False,
            "warning": (
                "This run did not cover the full requested dataset and must "
                "not be reported as a final full-dataset result."
            ),
            "run": latest_math_agent_run,
            "summary": get_parameter_summary(
                MATH_AGENT_EXPERIMENT,
                split=split,
                model=model,
                run_ids=[latest_math_agent_run["run_id"]],
                ensure_table=False,
                include_partial_runs=True,
            ),
        }

    def summary(experiment_name):
        return get_parameter_summary(
            experiment_name,
            split=split,
            model=model,
            run_ids=run_ids,
            ensure_table=False,
        )

    return {
        "split_filter": split,
        "model_filter": model,
        "selected_runs": selected_runs,
        "experiment_counts": get_experiment_counts(
            split=split,
            model=model,
            run_ids=run_ids,
            ensure_table=False,
        ),
        "retrieval_shortlists": get_retrieval_shortlists(
            split=split,
            run_ids=run_ids,
            ensure_table=False,
        ),
        "best_parameters": get_best_rag_parameter_rows(
            split=split,
            model=model,
            run_ids=run_ids,
            ensure_table=False,
        ),
        "statistical_analyses": get_statistical_analyses(
            split=split,
            model=model,
            run_ids=run_ids,
            ensure_table=False,
        ),
        "baseline_summary": summary("baseline"),
        "chunk_rag_summary": summary("top_chunks_evidence_sweep"),
        "legacy_chunk_rag_summary": summary("top_chunks_parameter_sweep"),
        "dev_shortlist_rag_summary": summary(
            "full_rag_dev_shortlist_sweep"
        ),
        "full_rag_summary": summary("full_rag_parameter_sweep"),
        "legacy_full_rag_summary": summary("full_rag"),
        "full_rag_math_agent_summary": summary(
            MATH_AGENT_EXPERIMENT
        ),
        "exploratory_math_agent": exploratory_math_agent,
        "legacy_full_rag_math_agent_summary": summary(
            "full_rag_with_math_agent"
        ),
    }
