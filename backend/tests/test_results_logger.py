import importlib.util
from importlib.machinery import ModuleSpec
import sys
import unittest
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


if importlib.util.find_spec("psycopg2") is None:
    psycopg2_stub = ModuleType("psycopg2")
    psycopg2_stub.__spec__ = ModuleSpec("psycopg2", loader=None)
    psycopg2_stub.connect = lambda *_args, **_kwargs: None
    extras_stub = ModuleType("psycopg2.extras")
    extras_stub.__spec__ = ModuleSpec("psycopg2.extras", loader=None)
    extras_stub.Json = lambda value: value
    extras_stub.RealDictCursor = object
    extras_stub.execute_values = lambda *_args, **_kwargs: None
    psycopg2_stub.extras = extras_stub
    sys.modules["psycopg2"] = psycopg2_stub
    sys.modules["psycopg2.extras"] = extras_stub


from app import results_logger  # noqa: E402


class _FakeCursor:
    def __init__(self, *, rows=None, row=None, rowcount=1):
        self.rows = [] if rows is None else rows
        self.row = row
        self.rowcount = rowcount
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        self.executions.append((query, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self, **_kwargs):
        return self._cursor


class ResultsLoggerTests(unittest.TestCase):
    def test_final_results_uses_exact_selected_runs_and_returns_artifacts(self):
        selected_runs = [
            {"run_id": "retrieval-run", "experiment_name": "retrieval"},
            {"run_id": "generation-run", "experiment_name": "generation"},
        ]
        run_ids = ["retrieval-run", "generation-run"]

        with ExitStack() as stack:
            create_table = stack.enter_context(
                patch.object(results_logger, "create_results_table")
            )
            latest_runs = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_latest_completed_runs",
                    return_value=selected_runs,
                )
            )
            latest_math_run = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_latest_completed_math_agent_run",
                    return_value=None,
                )
            )
            counts = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_experiment_counts",
                    return_value=[{"run_id": "retrieval-run"}],
                )
            )
            shortlists = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_retrieval_shortlists",
                    return_value=[{"run_id": "retrieval-run", "candidates": []}],
                )
            )
            winners = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_best_rag_parameter_rows",
                    return_value=[{"run_id": "generation-run", "top_k": 5}],
                )
            )
            analyses = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_statistical_analyses",
                    return_value=[{"run_id": "generation-run", "analysis": {}}],
                )
            )
            summaries = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_parameter_summary",
                    side_effect=lambda experiment_name, **_kwargs: [
                        {"experiment_name": experiment_name}
                    ],
                )
            )

            result = results_logger.get_final_result_tables(
                split="dev",
                model="gpt-4o-mini",
            )

        create_table.assert_called_once_with()
        latest_runs.assert_called_once_with(
            split="dev",
            model="gpt-4o-mini",
            ensure_table=False,
        )
        latest_math_run.assert_called_once_with(
            split="dev",
            model="gpt-4o-mini",
            ensure_table=False,
        )
        counts.assert_called_once_with(
            split="dev",
            model="gpt-4o-mini",
            run_ids=run_ids,
            ensure_table=False,
        )
        shortlists.assert_called_once_with(
            split="dev",
            run_ids=run_ids,
            ensure_table=False,
        )
        winners.assert_called_once_with(
            split="dev",
            model="gpt-4o-mini",
            run_ids=run_ids,
            ensure_table=False,
        )
        analyses.assert_called_once_with(
            split="dev",
            model="gpt-4o-mini",
            run_ids=run_ids,
            ensure_table=False,
        )

        expected_experiments = [
            "baseline",
            "top_chunks_evidence_sweep",
            "top_chunks_parameter_sweep",
            "full_rag_dev_shortlist_sweep",
            "full_rag_parameter_sweep",
            "full_rag",
            "full_rag_math_tool_agent",
            "full_rag_with_math_agent",
        ]
        self.assertEqual(
            [call.args[0] for call in summaries.call_args_list],
            expected_experiments,
        )
        for call in summaries.call_args_list:
            self.assertEqual(call.kwargs["run_ids"], run_ids)
            self.assertEqual(call.kwargs["split"], "dev")
            self.assertEqual(call.kwargs["model"], "gpt-4o-mini")
            self.assertFalse(call.kwargs["ensure_table"])

        self.assertEqual(result["selected_runs"], selected_runs)
        self.assertIsNone(result["exploratory_math_agent"])
        self.assertEqual(result["retrieval_shortlists"][0]["run_id"], "retrieval-run")
        self.assertEqual(result["best_parameters"][0]["run_id"], "generation-run")
        self.assertEqual(result["statistical_analyses"][0]["run_id"], "generation-run")
        self.assertEqual(
            result["chunk_rag_summary"],
            [{"experiment_name": "top_chunks_evidence_sweep"}],
        )

    def test_latest_runs_query_requires_completed_authoritative_runs_and_filters_model(self):
        cursor = _FakeCursor(
            rows=[
                {
                    "run_id": "latest-dev-run",
                    "experiment_name": "full_rag_dev_shortlist_sweep",
                    "dataset": "docfinqa",
                    "split": "dev",
                    "model": "gpt-4o-mini",
                }
            ]
        )
        connection = _FakeConnection(cursor)

        with patch.object(results_logger, "get_connection", return_value=connection):
            rows = results_logger.get_latest_completed_runs(
                split="dev",
                model="gpt-4o-mini",
                ensure_table=False,
            )

        query, params = cursor.executions[0]
        normalized_query = " ".join(query.split())
        self.assertIn("status = 'completed'", normalized_query)
        self.assertIn("is_authoritative = TRUE", normalized_query)
        self.assertIn("split = %s", normalized_query)
        self.assertIn("(model = %s OR model IS NULL)", normalized_query)
        self.assertIn("SELECT DISTINCT ON", normalized_query)
        self.assertIn("completed_at DESC", normalized_query)
        self.assertEqual(params, ("dev", "gpt-4o-mini"))
        self.assertEqual(rows[0]["run_id"], "latest-dev-run")

    def test_parameter_summary_is_run_and_model_aware_and_returns_json_numbers(self):
        cursor = _FakeCursor(
            rows=[
                {
                    "run_id": "retrieval-run",
                    "model": None,
                    "accuracy": Decimal("0.625"),
                    "retrieval_selection_score": Decimal("0.8125"),
                    "avg_precision_at_k": Decimal("0.75"),
                    "selection_details": {
                        "nested_decimal": Decimal("0.5"),
                    },
                }
            ]
        )
        connection = _FakeConnection(cursor)

        with patch.object(results_logger, "get_connection", return_value=connection):
            rows = results_logger.get_parameter_summary(
                "top_chunks_evidence_sweep",
                split="train",
                model="gpt-4o-mini",
                run_ids=["retrieval-run"],
                ensure_table=False,
            )

        query, params = cursor.executions[0]
        normalized_query = " ".join(query.split())
        self.assertIn("result.run_id", normalized_query)
        self.assertIn("run.model", normalized_query)
        self.assertIn("comparison_method", normalized_query)
        self.assertIn("run.is_authoritative = TRUE", normalized_query)
        self.assertIn("result.run_id = ANY(%s)", normalized_query)
        self.assertIn("0.50 * COALESCE", normalized_query)
        self.assertIn("0.30 * COALESCE", normalized_query)
        self.assertIn("0.10 * COALESCE", normalized_query)
        self.assertIn("retrieval_selection_score DESC NULLS LAST", normalized_query)
        self.assertIn("::double precision AS accuracy", normalized_query)
        self.assertEqual(
            params,
            ("top_chunks_evidence_sweep", ["retrieval-run"]),
        )
        self.assertIsInstance(rows[0]["accuracy"], float)
        self.assertIsInstance(rows[0]["retrieval_selection_score"], float)
        self.assertIsInstance(rows[0]["avg_precision_at_k"], float)
        self.assertIsInstance(
            rows[0]["selection_details"]["nested_decimal"],
            float,
        )

    def test_latest_math_run_allows_completed_partial_run(self):
        cursor = _FakeCursor(
            row={
                "run_id": "partial-math-run",
                "experiment_name": "full_rag_math_tool_agent",
                "status": "completed",
                "is_full_run": False,
                "is_authoritative": False,
            }
        )
        connection = _FakeConnection(cursor)

        with patch.object(
            results_logger,
            "get_connection",
            return_value=connection,
        ):
            row = results_logger.get_latest_completed_math_agent_run(
                split="train_dev",
                model="gpt-4o-mini",
                ensure_table=False,
            )

        query, params = cursor.executions[0]
        normalized_query = " ".join(query.split())
        self.assertIn("status = 'completed'", normalized_query)
        self.assertIn("experiment_name = %s", normalized_query)
        self.assertNotIn("is_authoritative = TRUE", normalized_query)
        self.assertEqual(
            params,
            (
                "full_rag_math_tool_agent",
                "train_dev",
                "gpt-4o-mini",
            ),
        )
        self.assertEqual(row["run_id"], "partial-math-run")

    def test_partial_parameter_summary_still_requires_completed_exact_run(self):
        cursor = _FakeCursor(rows=[])
        connection = _FakeConnection(cursor)

        with patch.object(
            results_logger,
            "get_connection",
            return_value=connection,
        ):
            results_logger.get_parameter_summary(
                "full_rag_math_tool_agent",
                split="train_dev",
                model="gpt-4o-mini",
                run_ids=["partial-math-run"],
                ensure_table=False,
                include_partial_runs=True,
            )

        query, params = cursor.executions[0]
        normalized_query = " ".join(query.split())
        self.assertIn("run.status = 'completed'", normalized_query)
        self.assertNotIn("run.is_authoritative = TRUE", normalized_query)
        self.assertIn("result.run_id = ANY(%s)", normalized_query)
        self.assertEqual(
            params,
            ("full_rag_math_tool_agent", ["partial-math-run"]),
        )

    def test_final_results_keeps_partial_math_run_separate(self):
        partial_run = {
            "run_id": "partial-math-run",
            "experiment_name": "full_rag_math_tool_agent",
            "split": "train_dev",
            "model": "gpt-4o-mini",
            "status": "completed",
            "is_full_run": False,
            "is_authoritative": False,
        }

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(results_logger, "create_results_table")
            )
            stack.enter_context(
                patch.object(
                    results_logger,
                    "get_latest_completed_runs",
                    return_value=[],
                )
            )
            stack.enter_context(
                patch.object(
                    results_logger,
                    "get_latest_completed_math_agent_run",
                    return_value=partial_run,
                )
            )
            for function_name in (
                "get_experiment_counts",
                "get_retrieval_shortlists",
                "get_best_rag_parameter_rows",
                "get_statistical_analyses",
            ):
                stack.enter_context(
                    patch.object(
                        results_logger,
                        function_name,
                        return_value=[],
                    )
                )
            summaries = stack.enter_context(
                patch.object(
                    results_logger,
                    "get_parameter_summary",
                    return_value=[{"accuracy": 0.5}],
                )
            )

            result = results_logger.get_final_result_tables(
                split="train_dev",
                model="gpt-4o-mini",
            )

        exploratory = result["exploratory_math_agent"]
        self.assertFalse(exploratory["authoritative"])
        self.assertEqual(exploratory["run"]["run_id"], "partial-math-run")
        exploratory_call = next(
            call
            for call in summaries.call_args_list
            if call.kwargs.get("include_partial_runs") is True
        )
        self.assertEqual(
            exploratory_call.kwargs["run_ids"],
            ["partial-math-run"],
        )
        canonical_calls = [
            call
            for call in summaries.call_args_list
            if not call.kwargs.get("include_partial_runs", False)
        ]
        self.assertTrue(
            all(call.kwargs["run_ids"] == [] for call in canonical_calls)
        )

    def test_schema_contains_run_tracking_artifact_links_and_reporting_indexes(self):
        cursor = _FakeCursor()
        connection = _FakeConnection(cursor)

        with (
            patch.object(results_logger, "_SCHEMA_READY", False),
            patch.object(results_logger, "get_connection", return_value=connection),
        ):
            results_logger.create_results_table()

        schema_sql = " ".join(
            query for query, _params in cursor.executions
        )
        normalized_schema_sql = " ".join(schema_sql.split())
        self.assertIn("CREATE TABLE IF NOT EXISTS rag_experiment_runs", normalized_schema_sql)
        self.assertIn("run_id TEXT", normalized_schema_sql)
        self.assertIn("status TEXT NOT NULL", normalized_schema_sql)
        self.assertIn("is_full_run BOOLEAN NOT NULL", normalized_schema_sql)
        self.assertIn("is_authoritative BOOLEAN NOT NULL", normalized_schema_sql)
        self.assertIn("expected_rows BIGINT", normalized_schema_sql)
        self.assertIn("last_checkpoint_at TIMESTAMPTZ", normalized_schema_sql)
        self.assertIn("result_key TEXT", normalized_schema_sql)
        self.assertIn("ALTER TABLE rag_best_parameters ADD COLUMN IF NOT EXISTS run_id TEXT", normalized_schema_sql)
        self.assertIn("ALTER TABLE rag_retrieval_shortlists ADD COLUMN IF NOT EXISTS run_id TEXT", normalized_schema_sql)
        self.assertIn("ALTER TABLE rag_statistical_analyses ADD COLUMN IF NOT EXISTS run_id TEXT", normalized_schema_sql)
        self.assertIn("rag_results_run_experiment_split_idx", normalized_schema_sql)
        self.assertIn("rag_results_run_result_key_idx", normalized_schema_sql)
        self.assertIn("ON rag_results (run_id, result_key)", normalized_schema_sql)
        self.assertIn("rag_experiment_runs_latest_authoritative_idx_v2", normalized_schema_sql)
        self.assertIn("WHERE status = 'completed' AND is_authoritative = TRUE", normalized_schema_sql)

    def test_resume_validates_the_exact_saved_manifest_and_running_status(self):
        saved_run = {
            "run_id": "run-123",
            "experiment_name": "full_rag_dev_shortlist_sweep",
            "dataset": "docfinqa",
            "split": "dev",
            "model": "gpt-4o-mini",
            "status": "running",
            "parameters": {
                "sample_size": 500,
                "sample_seed": 42,
                "question_ids": ["2", "7"],
            },
            "expected_rows": 1500,
        }
        requested = {
            "run_id": "run-123",
            "experiment_name": "full_rag_dev_shortlist_sweep",
            "dataset": "docfinqa",
            "split": "dev",
            "model": "gpt-4o-mini",
            "parameters": saved_run["parameters"],
            "expected_rows": 1500,
        }

        with patch.object(
            results_logger,
            "get_experiment_run",
            return_value=saved_run,
        ):
            resumed = results_logger.resume_experiment_run(**requested)

        self.assertEqual(resumed, saved_run)

        mismatched = dict(requested)
        mismatched["parameters"] = {
            **saved_run["parameters"],
            "sample_seed": 43,
        }
        with patch.object(
            results_logger,
            "get_experiment_run",
            return_value=saved_run,
        ):
            with self.assertRaisesRegex(ValueError, "does not match"):
                results_logger.resume_experiment_run(**mismatched)

        completed_run = {**saved_run, "status": "completed"}
        with patch.object(
            results_logger,
            "get_experiment_run",
            return_value=completed_run,
        ):
            with self.assertRaisesRegex(ValueError, "already completed"):
                results_logger.resume_experiment_run(**requested)

    def test_checkpoint_progress_updates_only_a_running_run(self):
        cursor = _FakeCursor(rowcount=1)
        connection = _FakeConnection(cursor)

        with (
            patch.object(results_logger, "create_results_table"),
            patch.object(
                results_logger,
                "get_connection",
                return_value=connection,
            ),
        ):
            results_logger.update_experiment_run_progress(
                run_id="run-123",
                questions_processed=125,
                rows_saved=375,
            )

        query, params = cursor.executions[0]
        normalized_query = " ".join(query.split())
        self.assertIn("last_checkpoint_at = CURRENT_TIMESTAMP", normalized_query)
        self.assertIn("status = 'running'", normalized_query)
        self.assertEqual(
            params,
            {
                "run_id": "run-123",
                "questions_processed": 125,
                "rows_saved": 375,
            },
        )

        cursor = _FakeCursor(rowcount=0)
        connection = _FakeConnection(cursor)
        with (
            patch.object(results_logger, "create_results_table"),
            patch.object(
                results_logger,
                "get_connection",
                return_value=connection,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "running state"):
                results_logger.update_experiment_run_progress(
                    run_id="completed-run",
                    questions_processed=125,
                    rows_saved=375,
                )

    def test_authoritative_completion_verifies_durable_and_unique_row_counts(self):
        cursor = _FakeCursor(row=(1500, 1500, 1500), rowcount=1)
        connection = _FakeConnection(cursor)

        with (
            patch.object(results_logger, "create_results_table"),
            patch.object(
                results_logger,
                "get_connection",
                return_value=connection,
            ),
        ):
            results_logger.complete_experiment_run(
                run_id="run-123",
                questions_processed=500,
                rows_saved=1500,
                is_full_run=False,
                is_authoritative=True,
            )

        self.assertEqual(len(cursor.executions), 2)
        count_query, count_params = cursor.executions[0]
        completion_query, completion_params = cursor.executions[1]
        normalized_count_query = " ".join(count_query.split())
        normalized_completion_query = " ".join(completion_query.split())
        self.assertIn("COUNT(DISTINCT result.result_key)", normalized_count_query)
        self.assertIn("result.run_id = run.run_id", normalized_count_query)
        self.assertEqual(count_params, ("run-123",))
        self.assertIn("is_authoritative = %(is_authoritative)s", normalized_completion_query)
        self.assertTrue(completion_params["is_authoritative"])
        self.assertFalse(completion_params["is_full_run"])

        invalid_counts = (
            ((1500, 1499, 1499), "durable rows"),
            ((1500, 1500, 1499), "cannot be authoritative"),
        )
        for counts, message in invalid_counts:
            with self.subTest(counts=counts):
                invalid_cursor = _FakeCursor(row=counts, rowcount=1)
                invalid_connection = _FakeConnection(invalid_cursor)
                with (
                    patch.object(results_logger, "create_results_table"),
                    patch.object(
                        results_logger,
                        "get_connection",
                        return_value=invalid_connection,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        results_logger.complete_experiment_run(
                            run_id="run-123",
                            questions_processed=500,
                            rows_saved=1500,
                            is_full_run=False,
                            is_authoritative=True,
                        )
                self.assertEqual(len(invalid_cursor.executions), 1)

    def test_run_scoped_results_require_stable_result_keys(self):
        result_without_key = {
            "run_id": "run-123",
            "experiment_name": "baseline",
            "question_id": "7",
        }

        with patch.object(results_logger, "create_results_table"):
            with self.assertRaisesRegex(ValueError, "requires result_key"):
                results_logger.log_question_result(result_without_key)

            with self.assertRaisesRegex(ValueError, "requires result_key"):
                results_logger.log_question_results(
                    [result_without_key],
                    ensure_table=False,
                )

        cursor = _FakeCursor(rows=[])
        connection = _FakeConnection(cursor)
        with (
            patch.object(results_logger, "create_results_table"),
            patch.object(
                results_logger,
                "get_connection",
                return_value=connection,
            ),
        ):
            rows = results_logger.get_run_results_for_resume("run-123")

        query, params = cursor.executions[0]
        normalized_query = " ".join(query.split())
        self.assertIn("result_key", normalized_query)
        self.assertIn("WHERE run_id = %s", normalized_query)
        self.assertEqual(params, ("run-123",))
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
