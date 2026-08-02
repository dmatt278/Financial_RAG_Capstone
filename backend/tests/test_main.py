import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = BACKEND_ROOT / "app" / "main.py"


class _FakeFastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    post = get


class _FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_main_module():
    fastapi = _module(
        "fastapi",
        FastAPI=_FakeFastAPI,
        HTTPException=_FakeHTTPException,
    )
    data_loader = _module(
        "app.data.data_loader",
        DEFAULT_SAMPLE_SEED=42,
        TRAIN_SPLIT="train",
        TUNING_SPLIT="train_dev",
        prepare_finqa_files=Mock(return_value={"loaded": True}),
    )
    pipeline = _module(
        "app.rag.pipeline",
        DEV_SAMPLE_SIZE=500,
        MATH_SAMPLE_SIZE=100,
        TEST_SAMPLE_SIZE=500,
        TRAIN_SAMPLE_SIZE=300,
        full_rag_shortlist_sweep=Mock(return_value={"stage": 2}),
        full_rag_with_math_agent=Mock(return_value={"math": True}),
        get_baseline_results=Mock(return_value={"stage": 3}),
        top_chunks=Mock(return_value={"stage": 1}),
    )
    vector_store = _module(
        "app.rag.vector_store",
        insert_docfinqa_chunk_sweep=Mock(return_value={"loaded": True}),
    )
    results_logger = _module(
        "app.results_logger",
        get_experiment_run=Mock(return_value={"run_id": "run-1"}),
        get_final_result_tables=Mock(return_value={"results": True}),
    )
    replacements = {
        "fastapi": fastapi,
        "app.data.data_loader": data_loader,
        "app.rag.pipeline": pipeline,
        "app.rag.vector_store": vector_store,
        "app.results_logger": results_logger,
    }
    spec = importlib.util.spec_from_file_location("main_under_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class MainEndpointTests(unittest.TestCase):
    def setUp(self):
        self.main = _load_main_module()

    def test_stage_defaults_and_resume_ids_are_forwarded(self):
        self.main.chunk_rag(resume_run_id="stage-1")
        self.main.top_chunks.assert_called_once_with(
            split="train",
            start_index=0,
            limit=None,
            chunk_size=None,
            log_results=True,
            return_results=False,
            sample_size=300,
            sample_seed=42,
            resume_run_id="stage-1",
        )

        self.main.full_rag_pipeline(resume_run_id="stage-2")
        self.main.full_rag_shortlist_sweep.assert_called_once_with(
            start_index=0,
            limit=None,
            log_results=True,
            return_results=False,
            sample_size=500,
            sample_seed=42,
            resume_run_id="stage-2",
        )

        self.main.get_baselines(resume_run_id="stage-3")
        self.main.get_baseline_results.assert_called_once_with(
            sample_size=500,
            sample_seed=42,
            resume_run_id="stage-3",
        )

    def test_math_defaults_to_one_config_and_100_questions(self):
        self.main.full_rag_math_agent_pipeline(resume_run_id="math-run")

        self.main.full_rag_with_math_agent.assert_called_once_with(
            split="train_dev",
            start_index=0,
            limit=None,
            top_k_values=[5],
            strategies=["section"],
            retrieval_methods=["hybrid"],
            chunk_size=512,
            log_results=True,
            return_results=False,
            sample_size=100,
            sample_seed=42,
            resume_run_id="math-run",
        )

    def test_long_run_lock_rejects_overlapping_work(self):
        self.main._LONG_EXPERIMENT_LOCK.acquire()
        try:
            with self.assertRaises(_FakeHTTPException) as raised:
                self.main.chunk_rag()
        finally:
            self.main._LONG_EXPERIMENT_LOCK.release()

        self.assertEqual(raised.exception.status_code, 409)

    def test_experiment_status_returns_checkpoint_or_404(self):
        self.assertEqual(
            self.main.experiment_status("run-1"),
            {"run_id": "run-1"},
        )
        self.main.get_experiment_run.return_value = None
        with self.assertRaises(_FakeHTTPException) as raised:
            self.main.experiment_status("missing")
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
