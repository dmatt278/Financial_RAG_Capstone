import os
import sys
import unittest
import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _stub_module(name, **attributes):
    if importlib.util.find_spec(name) is not None:
        return
    module = ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    sys.modules[name] = module


_stub_module("ijson", items=lambda *_args, **_kwargs: iter(()))
_stub_module("huggingface_hub", hf_hub_download=lambda **_kwargs: "")
_stub_module("rank_bm25", BM25Okapi=object)
_stub_module("sentence_transformers", CrossEncoder=object)

if importlib.util.find_spec("llama_index") is None:
    llama_index_stub = ModuleType("llama_index")
    llama_index_core_stub = ModuleType("llama_index.core")
    node_parser_stub = ModuleType("llama_index.core.node_parser")
    node_parser_stub.TokenTextSplitter = object
    node_parser_stub.SentenceSplitter = object
    llama_index_core_stub.node_parser = node_parser_stub
    llama_index_stub.core = llama_index_core_stub
    sys.modules["llama_index"] = llama_index_stub
    sys.modules["llama_index.core"] = llama_index_core_stub
    sys.modules["llama_index.core.node_parser"] = node_parser_stub

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


from app.rag.pipeline import (  # noqa: E402
    DEV_SHORTLIST_EXPERIMENT,
    RETRIEVAL_SHORTLIST_SIZE,
    STAGE_2_ANALYSIS_NAME,
    STAGE_3_ANALYSIS_NAME,
    _get_docfinqa_evidence_alignment,
    full_rag_parameter_sweep,
    full_rag_shortlist_sweep,
    full_rag_with_math_agent,
    get_baseline_results,
    top_chunks,
)


def _example():
    return {
        "question_id": "1",
        "question": "What is the value?",
        "gold_answer": "10",
        "document_id": "document-1",
        "document_text": "full document text",
        "program": "answer = 10",
    }


def _chunks(count=10):
    return [
        {
            "id": f"chunk-{index}",
            "chunk_id": f"chunk-{index}",
            "text": f"chunk text {index}",
            "score": 1.0,
            "metadata": {},
        }
        for index in range(count)
    ]


def _retrieval_metrics():
    return {
        "all_evidence_hit_at_k": 1.0,
        "evidence_recall_at_k": 1.0,
        "reciprocal_rank": 1.0,
        "precision_at_k": 1.0,
    }


def _sweep_output(retrieval_configs, max_top_k=10):
    chunks = _chunks(max(40, max_top_k))
    return {
        "all_chunks": chunks,
        "rankings": {
            config: [dict(chunk) for chunk in chunks[:max_top_k]]
            for config in retrieval_configs
        },
        "timings": {
            "corpus_load_seconds": 0.0,
            "keyword_ranking_seconds": 0.0,
            "semantic_ranking_seconds": 0.0,
            "hybrid_fusion_seconds": 0.0,
            "reranking_seconds": 0.0,
        },
    }


class StagedPipelineTests(unittest.TestCase):
    @patch("app.rag.pipeline.align_docfinqa_evidence", return_value=[])
    @patch("app.rag.pipeline.get_all_chunks", return_value=_chunks())
    @patch(
        "app.rag.pipeline.get_finqa_gold_evidence",
        return_value=["evidence"],
    )
    def test_evidence_alignment_forwards_docfinqa_provenance(
        self,
        get_gold_evidence,
        _get_chunks,
        _align_evidence,
    ):
        example = _example()

        _get_docfinqa_evidence_alignment(
            example,
            split="dev",
            where={"document_id": {"$eq": "document-1"}},
        )

        get_gold_evidence.assert_called_once_with(
            split="dev",
            question=example["question"],
            gold_answer=example["gold_answer"],
            gold_program=example["program"],
            document_text=example["document_text"],
        )

    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.align_docfinqa_evidence", return_value=[])
    @patch("app.rag.pipeline.get_finqa_gold_evidence", return_value=["evidence"])
    @patch("app.rag.pipeline.get_parameter_sweep_rankings")
    @patch("app.rag.pipeline.iter_sampled_docfinqa_examples")
    def test_retrieval_sweep_includes_off_and_all_reranker_pools(
        self,
        iter_examples,
        get_sweep_rankings,
        _gold_evidence,
        _align_evidence,
        evaluate_retrieval,
    ):
        iter_examples.return_value = iter([_example()])
        get_sweep_rankings.side_effect = lambda **kwargs: _sweep_output(
            kwargs["retrieval_configs"],
            kwargs["max_top_k"],
        )
        evaluate_retrieval.return_value = _retrieval_metrics()

        summary = top_chunks(
            split="train",
            top_k_values=[3],
            strategies=["fixed"],
            retrieval_methods=["semantic"],
            chunk_size=512,
            log_results=False,
            return_results=True,
        )

        self.assertEqual(summary["parameter_combinations_per_question"], 4)
        self.assertEqual(
            summary["reranker_configurations"],
            [
                {
                    "reranker_enabled": False,
                    "reranker_pool_size": None,
                },
                {
                    "reranker_enabled": True,
                    "reranker_pool_size": 10,
                },
                {
                    "reranker_enabled": True,
                    "reranker_pool_size": 20,
                },
                {
                    "reranker_enabled": True,
                    "reranker_pool_size": 40,
                },
            ],
        )
        get_sweep_rankings.assert_called_once()
        self.assertEqual(
            set(get_sweep_rankings.call_args.kwargs["retrieval_configs"]),
            {
                ("semantic", False, None),
                ("semantic", True, 10),
                ("semantic", True, 20),
                ("semantic", True, 40),
            },
        )
        self.assertEqual(
            {
                (
                    result["reranker_used"],
                    result["reranker_pool_size"],
                )
                for result in summary["results"]
            },
            {
                (False, None),
                (True, 10),
                (True, 20),
                (True, 40),
            },
        )
        self.assertFalse(summary["shortlist_saved"])
        self.assertEqual(
            summary["shortlist_save_reason"],
            "result_logging_disabled",
        )
        example = _example()
        _gold_evidence.assert_called_once_with(
            split="train",
            question=example["question"],
            gold_answer=example["gold_answer"],
            gold_program=example["program"],
            document_text=example["document_text"],
        )

    @patch("app.rag.pipeline.complete_experiment_run")
    @patch(
        "app.rag.pipeline.start_experiment_run",
        return_value="train-run-id",
    )
    @patch("app.rag.pipeline.log_question_results", return_value=[1])
    @patch("app.rag.pipeline.update_experiment_run_progress")
    @patch("app.rag.pipeline.create_results_table")
    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.align_docfinqa_evidence", return_value=[])
    @patch("app.rag.pipeline.get_finqa_gold_evidence", return_value=["evidence"])
    @patch("app.rag.pipeline.get_parameter_sweep_rankings")
    @patch("app.rag.pipeline.iter_sampled_docfinqa_examples")
    def test_partial_retrieval_run_is_tracked_but_not_marked_full(
        self,
        iter_examples,
        get_sweep_rankings,
        _gold_evidence,
        _align_evidence,
        evaluate_retrieval,
        _create_table,
        update_progress,
        log_results,
        start_run,
        complete_run,
    ):
        iter_examples.return_value = iter([_example()])
        get_sweep_rankings.side_effect = lambda **kwargs: _sweep_output(
            kwargs["retrieval_configs"],
            kwargs["max_top_k"],
        )
        evaluate_retrieval.return_value = _retrieval_metrics()

        summary = top_chunks(
            split="train",
            top_k_values=[3],
            strategies=["fixed"],
            retrieval_methods=["semantic"],
            reranker_enabled_values=[False],
            chunk_size=512,
            log_results=True,
            return_results=True,
        )

        self.assertEqual(summary["run_id"], "train-run-id")
        self.assertFalse(summary["is_full_run"])
        self.assertFalse(summary["is_authoritative"])
        self.assertEqual(summary["rows_saved"], 1)
        self.assertEqual(summary["results"][0]["run_id"], "train-run-id")
        iter_examples.assert_called_once_with(
            split="train",
            sample_size=50,
            seed=42,
        )
        start_run.assert_called_once()
        start_arguments = start_run.call_args.kwargs
        self.assertEqual(start_arguments["dataset"], "docfinqa")
        self.assertEqual(
            start_arguments["experiment_name"],
            "top_chunks_evidence_sweep",
        )
        self.assertEqual(start_arguments["split"], "train")
        self.assertIsNone(start_arguments["model"])
        self.assertEqual(start_arguments["expected_rows"], 1)
        self.assertEqual(
            start_arguments["parameters"]["question_ids"],
            ["1"],
        )
        self.assertEqual(
            start_arguments["parameters"]["sample_size"],
            50,
        )
        self.assertEqual(
            start_arguments["parameters"]["reranker_model"],
            "jinaai/jina-reranker-v1-tiny-en",
        )
        self.assertEqual(
            start_arguments["parameters"]["reranker_model_revision"],
            "aca45de6945b5dc6399abcd2a9c55ded5dc9111f",
        )
        self.assertEqual(
            start_arguments["parameters"]["semantic_search_backend"],
            "chroma_hnsw_depth_200",
        )
        update_progress.assert_called_once_with(
            run_id="train-run-id",
            questions_processed=1,
            rows_saved=1,
        )
        complete_run.assert_called_once_with(
            run_id="train-run-id",
            questions_processed=1,
            rows_saved=1,
            is_full_run=False,
            is_authoritative=False,
        )

    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.align_docfinqa_evidence", return_value=[])
    @patch("app.rag.pipeline.get_finqa_gold_evidence", return_value=["evidence"])
    @patch("app.rag.pipeline.get_parameter_sweep_rankings")
    @patch("app.rag.pipeline._materialize_protocol_examples")
    def test_default_retrieval_sweep_builds_324_unique_results_from_9_corpora(
        self,
        materialize_examples,
        get_sweep_rankings,
        _gold_evidence,
        _align_evidence,
        evaluate_retrieval,
    ):
        materialize_examples.return_value = ([_example()], ["1"])
        get_sweep_rankings.side_effect = lambda **kwargs: _sweep_output(
            kwargs["retrieval_configs"],
            kwargs["max_top_k"],
        )
        evaluate_retrieval.return_value = _retrieval_metrics()

        summary = top_chunks(
            split="train",
            log_results=False,
            return_results=True,
        )

        self.assertEqual(summary["parameter_combinations_per_question"], 324)
        self.assertEqual(summary["rows_processed"], 324)
        self.assertEqual(summary["results_returned"], 324)
        self.assertEqual(len(summary["results"]), 324)
        self.assertEqual(
            len({result["result_key"] for result in summary["results"]}),
            324,
        )
        self.assertEqual(
            len(
                {
                    (
                        result["chunk_size"],
                        result["chunk_strategy"],
                        result["retrieval_method"],
                        result["top_k"],
                        result["reranker_used"],
                        result["reranker_pool_size"],
                    )
                    for result in summary["results"]
                }
            ),
            324,
        )
        self.assertEqual(get_sweep_rankings.call_count, 9)
        self.assertEqual(
            sum(not result["reranker_used"] for result in summary["results"]),
            81,
        )
        for pool_size in (10, 20, 40):
            self.assertEqual(
                sum(
                    result["reranker_used"]
                    and result["reranker_pool_size"] == pool_size
                    for result in summary["results"]
                ),
                81,
            )
        self.assertTrue(
            all(
                len(call.kwargs["retrieval_configs"]) == 12
                for call in get_sweep_rankings.call_args_list
            )
        )

    @patch("app.rag.pipeline.complete_experiment_run")
    @patch("app.rag.pipeline.start_experiment_run")
    @patch("app.rag.pipeline.resume_experiment_run")
    @patch("app.rag.pipeline.get_run_results_for_resume")
    @patch("app.rag.pipeline.log_question_results")
    @patch("app.rag.pipeline.update_experiment_run_progress")
    @patch("app.rag.pipeline.create_results_table")
    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.align_docfinqa_evidence")
    @patch("app.rag.pipeline.get_finqa_gold_evidence")
    @patch("app.rag.pipeline.get_parameter_sweep_rankings")
    @patch("app.rag.pipeline._materialize_protocol_examples")
    def test_resume_skips_all_retrieval_work_for_checkpointed_question(
        self,
        materialize_examples,
        get_sweep_rankings,
        get_gold_evidence,
        align_evidence,
        evaluate_retrieval,
        _create_table,
        update_progress,
        log_results,
        get_saved_results,
        resume_run,
        start_run,
        complete_run,
    ):
        materialize_examples.return_value = ([_example()], ["1"])
        get_saved_results.return_value = [
            {
                "result_key": (
                    "1::semantic__fixed__chunk_512__top_3__"
                    "reranker_off::retrieval"
                ),
                "question_id": "1",
                "is_correct": None,
                "retrieval_method": "semantic",
                "chunk_strategy": "fixed",
                "chunk_size": 512,
                "top_k": 3,
                "reranker_used": False,
                "reranker_pool_size": None,
                "retrieval_metrics": _retrieval_metrics(),
                "generation_metrics": {},
            }
        ]

        summary = top_chunks(
            split="train",
            top_k_values=[3],
            strategies=["fixed"],
            retrieval_methods=["semantic"],
            reranker_enabled_values=[False],
            chunk_size=512,
            sample_size=1,
            resume_run_id="retrieval-resume-run",
            return_results=True,
        )

        self.assertEqual(summary["run_id"], "retrieval-resume-run")
        self.assertEqual(summary["rows_already_saved"], 1)
        self.assertEqual(summary["rows_added_this_attempt"], 0)
        self.assertEqual(summary["rows_processed"], 0)
        self.assertEqual(summary["rows_saved"], 1)
        self.assertEqual(summary["results_returned"], 0)
        start_run.assert_not_called()
        resume_run.assert_called_once()
        get_saved_results.assert_called_once_with("retrieval-resume-run")
        get_sweep_rankings.assert_not_called()
        get_gold_evidence.assert_not_called()
        align_evidence.assert_not_called()
        evaluate_retrieval.assert_not_called()
        log_results.assert_not_called()
        update_progress.assert_called_once_with(
            run_id="retrieval-resume-run",
            questions_processed=1,
            rows_saved=1,
        )
        complete_run.assert_called_once_with(
            run_id="retrieval-resume-run",
            questions_processed=1,
            rows_saved=1,
            is_full_run=False,
            is_authoritative=False,
        )

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
    )
    @patch("app.rag.pipeline.evaluate_docfinqa_answer_metrics")
    @patch("app.rag.pipeline.generate_answer")
    @patch("app.rag.pipeline.limit_chunks_to_prompt_budget")
    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.get_top_k_chunks")
    @patch("app.rag.pipeline._get_docfinqa_evidence_alignment")
    @patch("app.rag.pipeline.iter_sampled_docfinqa_examples")
    def test_generation_sweep_runs_only_complete_candidate_configs(
        self,
        iter_examples,
        evidence_alignment,
        get_top_k,
        evaluate_retrieval,
        limit_prompt,
        generate_answer,
        evaluate_answer,
    ):
        iter_examples.return_value = iter([_example()])
        evidence_alignment.return_value = []
        get_top_k.return_value = _chunks()
        evaluate_retrieval.return_value = _retrieval_metrics()
        limit_prompt.side_effect = lambda **kwargs: kwargs["retrieved_chunks"]
        generate_answer.return_value = "10"
        evaluate_answer.return_value = {
            "docfinqa_answer_correct": True,
            "within_one_percent": True,
        }
        candidate_configs = [
            {
                "chunk_size": 512,
                "strategy": "section",
                "retrieval_method": "hybrid",
                "top_k": 3,
                "reranker_enabled": False,
                "reranker_pool_size": None,
            },
            {
                "chunk_size": 1024,
                "strategy": "fixed",
                "retrieval_method": "semantic",
                "top_k": 5,
                "reranker_enabled": True,
                "reranker_pool_size": 20,
            },
        ]

        summary = full_rag_parameter_sweep(
            split="dev",
            candidate_configs=candidate_configs,
            log_results=False,
        )

        self.assertEqual(summary["parameter_combinations_per_question"], 2)
        self.assertEqual(summary["rows_processed"], 2)
        self.assertFalse(summary["statistical_analysis_saved"])
        self.assertEqual(
            summary["statistical_analysis_save_reason"],
            "result_logging_disabled",
        )
        self.assertEqual(get_top_k.call_count, 2)
        called_configs = {
            (
                call.kwargs["top_k"],
                call.kwargs["retrieval_method"],
                call.kwargs["reranker_enabled"],
                call.kwargs["reranker_pool_size"],
            )
            for call in get_top_k.call_args_list
        }
        self.assertEqual(
            called_configs,
            {
                (3, "hybrid", False, 20),
                (5, "semantic", True, 20),
            },
        )

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
    )
    @patch("app.rag.pipeline.complete_experiment_run")
    @patch("app.rag.pipeline.start_experiment_run")
    @patch("app.rag.pipeline.resume_experiment_run")
    @patch("app.rag.pipeline.get_run_results_for_resume")
    @patch("app.rag.pipeline.log_question_results")
    @patch("app.rag.pipeline.update_experiment_run_progress")
    @patch("app.rag.pipeline.create_results_table")
    @patch("app.rag.pipeline.generate_answer")
    @patch("app.rag.pipeline.get_top_k_chunks")
    @patch("app.rag.pipeline._get_docfinqa_evidence_alignment", return_value=[])
    @patch("app.rag.pipeline._materialize_protocol_examples")
    def test_resume_skips_an_already_checkpointed_generation_result(
        self,
        materialize_examples,
        _evidence_alignment,
        get_top_k,
        generate_answer,
        _create_table,
        update_progress,
        log_results,
        get_saved_results,
        resume_run,
        start_run,
        complete_run,
    ):
        example = _example()
        materialize_examples.return_value = ([example], ["1"])
        config = {
            "chunk_size": 512,
            "strategy": "fixed",
            "retrieval_method": "semantic",
            "top_k": 3,
            "reranker_enabled": False,
            "reranker_pool_size": None,
        }
        get_saved_results.return_value = [
            {
                "result_key": (
                    "1::semantic__fixed__chunk_512__top_3__"
                    "reranker_off::generated_answer"
                ),
                "question_id": "1",
                "is_correct": True,
                "retrieval_method": "semantic",
                "chunk_strategy": "fixed",
                "chunk_size": 512,
                "top_k": 3,
                "reranker_used": False,
                "reranker_pool_size": None,
                "retrieval_metrics": _retrieval_metrics(),
                "generation_metrics": {"within_one_percent": True},
            }
        ]

        summary = full_rag_parameter_sweep(
            split="dev",
            candidate_configs=[config],
            sample_size=1,
            resume_run_id="resume-run-id",
        )

        self.assertEqual(summary["run_id"], "resume-run-id")
        self.assertEqual(summary["rows_already_saved"], 1)
        self.assertEqual(summary["rows_added_this_attempt"], 0)
        self.assertEqual(summary["rows_processed"], 0)
        self.assertEqual(summary["rows_saved"], 1)
        start_run.assert_not_called()
        resume_run.assert_called_once()
        self.assertEqual(
            resume_run.call_args.kwargs["run_id"],
            "resume-run-id",
        )
        self.assertEqual(
            resume_run.call_args.kwargs["expected_rows"],
            1,
        )
        get_saved_results.assert_called_once_with("resume-run-id")
        get_top_k.assert_not_called()
        generate_answer.assert_not_called()
        log_results.assert_not_called()
        update_progress.assert_not_called()
        complete_run.assert_called_once_with(
            run_id="resume-run-id",
            questions_processed=1,
            rows_saved=1,
            is_full_run=False,
            is_authoritative=False,
        )

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
    )
    @patch("app.rag.pipeline.complete_experiment_run")
    @patch(
        "app.rag.pipeline.start_experiment_run",
        return_value="dev-run-id",
    )
    @patch("app.rag.pipeline.save_best_rag_parameters")
    @patch("app.rag.pipeline.save_statistical_analysis")
    @patch("app.rag.pipeline.log_question_results")
    @patch("app.rag.pipeline.update_experiment_run_progress")
    @patch("app.rag.pipeline.create_results_table")
    @patch("app.rag.pipeline.evaluate_docfinqa_answer_metrics")
    @patch("app.rag.pipeline.generate_answer")
    @patch("app.rag.pipeline.limit_chunks_to_prompt_budget")
    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.get_top_k_chunks")
    @patch("app.rag.pipeline._get_docfinqa_evidence_alignment")
    @patch("app.rag.pipeline.iter_sampled_docfinqa_examples")
    @patch("app.rag.pipeline.DEV_SAMPLE_SIZE", 2)
    def test_complete_dev_shortlist_saves_statistics_without_changing_winner(
        self,
        iter_examples,
        evidence_alignment,
        get_top_k,
        evaluate_retrieval,
        limit_prompt,
        generate_answer,
        evaluate_answer,
        _create_table,
        update_progress,
        log_results,
        save_statistics,
        save_best,
        start_run,
        complete_run,
    ):
        iter_examples.return_value = iter(
            [
                _example(),
                {
                    **_example(),
                    "question_id": "2",
                    "question": "What is the second value?",
                },
            ]
        )
        evidence_alignment.return_value = []

        def retrieve(**kwargs):
            chunk_size = kwargs["where"]["$and"][2]["chunk_size"]["$eq"]
            chunks = _chunks()
            chunks[0]["text"] = str(chunk_size)
            return chunks

        get_top_k.side_effect = retrieve
        evaluate_retrieval.return_value = _retrieval_metrics()
        limit_prompt.side_effect = lambda **kwargs: kwargs["retrieved_chunks"]
        generate_answer.side_effect = lambda **kwargs: (
            "10"
            if kwargs["retrieved_chunks"][0]["text"] == "256"
            else "wrong"
        )
        evaluate_answer.side_effect = lambda generated_answer, **_kwargs: {
            "docfinqa_answer_correct": generated_answer == "10",
            "within_one_percent": generated_answer == "10",
        }
        log_results.side_effect = lambda results, ensure_table=False: list(
            range(1, len(results) + 1)
        )
        candidate_configs = [
            {
                "chunk_size": 256 + index,
                "strategy": "section",
                "retrieval_method": "hybrid",
                "top_k": 5,
                "reranker_enabled": False,
                "reranker_pool_size": None,
            }
            for index in range(RETRIEVAL_SHORTLIST_SIZE)
        ]

        summary = full_rag_parameter_sweep(
            split="dev",
            candidate_configs=candidate_configs,
            experiment_name=DEV_SHORTLIST_EXPERIMENT,
            winner_source_split="dev",
            return_results=True,
            sample_size=2,
        )

        self.assertEqual(summary["best_parameters"]["chunk_size"], 256)
        self.assertEqual(
            summary["statistical_analysis"]["pairwise"]["family_size"],
            3,
        )
        self.assertFalse(
            summary["statistical_analysis"][
                "statistics_used_to_select_winner"
            ]
        )
        self.assertTrue(summary["statistical_analysis_saved"])
        self.assertEqual(summary["run_id"], "dev-run-id")
        self.assertFalse(summary["is_full_run"])
        self.assertTrue(summary["is_authoritative"])
        iter_examples.assert_called_once_with(
            split="dev",
            sample_size=2,
            seed=42,
        )
        start_run.assert_called_once()
        start_arguments = start_run.call_args.kwargs
        self.assertEqual(start_arguments["dataset"], "docfinqa")
        self.assertEqual(
            start_arguments["experiment_name"],
            DEV_SHORTLIST_EXPERIMENT,
        )
        self.assertEqual(start_arguments["split"], "dev")
        self.assertEqual(start_arguments["model"], "gpt-4o-mini")
        self.assertEqual(start_arguments["expected_rows"], 6)
        self.assertEqual(
            start_arguments["parameters"]["sample_size"],
            2,
        )
        self.assertEqual(
            start_arguments["parameters"]["sample_seed"],
            42,
        )
        self.assertEqual(
            start_arguments["parameters"]["question_ids"],
            ["1", "2"],
        )
        self.assertEqual(
            len(start_arguments["parameters"]["candidate_configs"]),
            3,
        )
        self.assertEqual(len(summary["results"]), 6)
        self.assertTrue(
            all(
                result["run_id"] == "dev-run-id"
                for result in summary["results"]
            )
        )
        save_statistics.assert_called_once()
        self.assertEqual(
            save_statistics.call_args.kwargs["run_id"],
            "dev-run-id",
        )
        self.assertEqual(
            save_statistics.call_args.kwargs["analysis_name"],
            STAGE_2_ANALYSIS_NAME,
        )
        save_best.assert_called_once()
        self.assertEqual(save_best.call_args.kwargs["run_id"], "dev-run-id")
        self.assertEqual(save_best.call_args.kwargs["config"]["chunk_size"], 256)
        update_progress.assert_called_once_with(
            run_id="dev-run-id",
            questions_processed=2,
            rows_saved=6,
        )
        complete_run.assert_called_once_with(
            run_id="dev-run-id",
            questions_processed=2,
            rows_saved=6,
            is_full_run=False,
            is_authoritative=True,
        )

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
    )
    @patch("app.rag.pipeline.complete_experiment_run")
    @patch(
        "app.rag.pipeline.start_experiment_run",
        return_value="math-run-id",
    )
    @patch("app.rag.pipeline.save_statistical_analysis")
    @patch(
        "app.rag.pipeline.log_question_results",
        return_value=[1, 2],
    )
    @patch("app.rag.pipeline.update_experiment_run_progress")
    @patch("app.rag.pipeline.create_results_table")
    @patch("app.rag.pipeline.evaluate_docfinqa_answer_metrics")
    @patch("app.rag.pipeline.math_agent")
    @patch("app.rag.pipeline.generate_answer")
    @patch("app.rag.pipeline.limit_chunks_to_prompt_budget")
    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline.get_top_k_chunks")
    @patch("app.rag.pipeline._get_docfinqa_evidence_alignment")
    @patch("app.rag.pipeline.iter_sampled_docfinqa_examples")
    @patch("app.rag.pipeline.MATH_SAMPLE_SIZE", 1)
    def test_complete_math_agent_run_uses_one_run_id(
        self,
        iter_examples,
        evidence_alignment,
        get_top_k,
        evaluate_retrieval,
        limit_chunks,
        generate_answer,
        run_math_agent,
        evaluate_answer,
        _create_table,
        update_progress,
        log_results,
        save_statistics,
        start_run,
        complete_run,
    ):
        iter_examples.return_value = iter([_example()])
        evidence_alignment.return_value = []
        get_top_k.return_value = _chunks(3)
        limit_chunks.side_effect = lambda **kwargs: list(
            kwargs["retrieved_chunks"]
        )
        evaluate_retrieval.return_value = _retrieval_metrics()

        def generate_direct(
            question,
            retrieved_chunks,
            generation_context_metrics=None,
        ):
            if generation_context_metrics is not None:
                generation_context_metrics.update(
                    {
                        "source_context_chunk_count": len(retrieved_chunks),
                        "generation_context_chunk_count": len(
                            retrieved_chunks
                        ),
                        "generation_context_chunk_ids": [
                            chunk["id"] for chunk in retrieved_chunks
                        ],
                        "prompt_was_truncated": False,
                    }
                )
            return "10"

        generate_answer.side_effect = generate_direct
        run_math_agent.return_value = {
            "answer": "10",
            "model": "gpt-4o-mini",
            "prompt_version": "test",
            "status": "ok",
            "program": [],
            "raw_answer": 10,
            "raw_model_output": "{}",
            "program_parse_succeeded": True,
            "execution_succeeded": True,
            "error": None,
            "execution_steps": [],
            "literal_operand_count": 1,
            "grounded_operand_count": 1,
            "operand_grounding_rate": 1.0,
            "ungrounded_operands": [],
            "generation_context_chunk_ids": [
                "chunk-0",
                "chunk-1",
                "chunk-2",
            ],
            "generation_context_chunk_count": 3,
            "prompt_was_truncated": False,
        }
        evaluate_answer.side_effect = lambda **_kwargs: {
            "docfinqa_answer_correct": True,
        }

        summary = full_rag_with_math_agent(
            split="train_dev",
            log_results=True,
            return_results=True,
            sample_size=1,
        )

        self.assertEqual(summary["run_id"], "math-run-id")
        self.assertFalse(summary["is_full_run"])
        self.assertTrue(summary["is_authoritative"])
        self.assertEqual(summary["rows_processed"], 2)
        self.assertEqual(summary["rows_saved"], 2)
        self.assertEqual(summary["results_returned"], 2)
        iter_examples.assert_called_once_with(
            split="train_dev",
            sample_size=1,
            seed=42,
        )
        self.assertEqual(
            {
                result["generation_metrics"]["comparison_method"]
                for result in summary["results"]
            },
            {"direct_llm", "math_agent"},
        )
        self.assertEqual(
            generate_answer.call_args.kwargs["retrieved_chunks"],
            run_math_agent.call_args.kwargs["chunks"],
        )
        self.assertEqual(log_results.call_count, 1)
        self.assertEqual(
            start_run.call_args.kwargs["experiment_name"],
            "full_rag_math_tool_agent",
        )
        self.assertEqual(start_run.call_args.kwargs["split"], "train_dev")
        self.assertEqual(start_run.call_args.kwargs["expected_rows"], 2)
        self.assertEqual(
            start_run.call_args.kwargs["parameters"]["sample_size"],
            1,
        )
        self.assertEqual(
            start_run.call_args.kwargs["parameters"]["question_ids"],
            ["1"],
        )
        self.assertEqual(
            start_run.call_args.kwargs["parameters"]["top_k_values"],
            [5],
        )
        self.assertEqual(
            start_run.call_args.kwargs["parameters"]["strategies"],
            ["section"],
        )
        self.assertEqual(
            start_run.call_args.kwargs["parameters"]["retrieval_methods"],
            ["hybrid"],
        )
        self.assertEqual(
            start_run.call_args.kwargs["parameters"]["comparison_methods"],
            ["direct_llm", "math_agent"],
        )
        save_statistics.assert_called_once()
        self.assertIn(
            "math_agent_vs_direct_llm__",
            save_statistics.call_args.kwargs["analysis_name"],
        )
        update_progress.assert_called_once_with(
            run_id="math-run-id",
            questions_processed=1,
            rows_saved=2,
        )
        complete_run.assert_called_once_with(
            run_id="math-run-id",
            questions_processed=1,
            rows_saved=2,
            is_full_run=False,
            is_authoritative=True,
        )

        iter_examples.return_value = iter([_example()])
        partial_summary = full_rag_with_math_agent(
            split="train_dev",
            start_index=0,
            limit=1,
            log_results=False,
            return_results=False,
            sample_size=1,
        )
        self.assertFalse(partial_summary["is_full_run"])
        self.assertFalse(partial_summary["is_authoritative"])
        self.assertEqual(partial_summary["statistical_analyses_saved"], 0)
        self.assertEqual(partial_summary["results_returned"], 0)

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
    )
    @patch("app.rag.pipeline.complete_experiment_run")
    @patch(
        "app.rag.pipeline.start_experiment_run",
        return_value="test-run-id",
    )
    @patch("app.rag.pipeline.save_statistical_analysis")
    @patch("app.rag.pipeline.log_question_results")
    @patch("app.rag.pipeline.update_experiment_run_progress")
    @patch("app.rag.generator.get_openai_client")
    @patch("app.rag.pipeline.create_results_table")
    @patch("app.rag.pipeline.evaluate_docfinqa_answer_metrics")
    @patch("app.rag.pipeline.evaluate_retrieval")
    @patch("app.rag.pipeline._get_docfinqa_evidence_alignment")
    @patch("app.rag.pipeline.get_top_k_chunks")
    @patch("app.rag.pipeline.generate_answer")
    @patch("app.rag.pipeline.generate_no_context_answer")
    @patch("app.rag.pipeline.iter_sampled_docfinqa_examples")
    @patch("app.rag.pipeline.get_best_rag_parameters")
    @patch("app.rag.pipeline.TEST_SAMPLE_SIZE", 4)
    def test_stage_three_compares_frozen_winner_with_three_baselines(
        self,
        get_best,
        iter_examples,
        generate_no_context,
        generate_answer,
        get_top_k,
        evidence_alignment,
        evaluate_retrieval,
        evaluate_answer,
        _create_table,
        get_client,
        update_progress,
        log_results,
        save_statistics,
        start_run,
        complete_run,
    ):
        get_best.return_value = {
            "source_experiment": DEV_SHORTLIST_EXPERIMENT,
            "retrieval_method": "hybrid",
            "reranker_used": True,
            "chunk_strategy": "section",
            "chunk_size": 512,
            "top_k": 5,
            "reranker_pool_size": 20,
            "source_split": "dev",
            "selection_metrics": {"accuracy": 0.75},
        }
        examples = [
            {
                **_example(),
                "question_id": str(index),
                "question": f"q{index}",
                "gold_answer": f"q{index}",
                "document_text": "full_document",
            }
            for index in range(4)
        ]
        iter_examples.return_value = iter(examples)
        saved_rows = []

        def save_batch(results, ensure_table=False):
            saved_rows.extend(dict(result) for result in results)
            return list(range(1, len(results) + 1))

        log_results.side_effect = save_batch
        generate_no_context.side_effect = lambda question: (
            f"no_context:{question}"
        )

        def retrieve(**kwargs):
            chunks = _chunks()
            chunks[0]["text"] = (
                "naive_rag"
                if kwargs["retrieval_method"] == "semantic"
                else "optimized_rag"
            )
            return chunks

        get_top_k.side_effect = retrieve

        def generate(
            question,
            retrieved_chunks,
            generation_context_metrics=None,
        ):
            if generation_context_metrics is not None:
                generation_context_metrics.update(
                    {
                        "source_context_chunk_count": len(
                            retrieved_chunks
                        ),
                        "generation_context_chunk_count": len(
                            retrieved_chunks
                        ),
                        "generation_context_chunk_ids": [
                            chunk.get("chunk_id")
                            for chunk in retrieved_chunks
                        ],
                        "prompt_was_truncated": False,
                    }
                )
            return f"{retrieved_chunks[0]['text']}:{question}"

        generate_answer.side_effect = generate
        evidence_alignment.return_value = []
        evaluate_retrieval.return_value = _retrieval_metrics()
        correct_questions = {
            "no_context": {"q0"},
            "full_document": {"q0", "q1"},
            "naive_rag": {"q0", "q1", "q2"},
            "optimized_rag": {"q0", "q1", "q2", "q3"},
        }

        def evaluate(generated_answer, gold_answer):
            method, _, _question = generated_answer.partition(":")
            is_correct = gold_answer in correct_questions[method]
            return {
                "docfinqa_answer_correct": is_correct,
                "within_one_percent": is_correct,
            }

        evaluate_answer.side_effect = evaluate

        summary = get_baseline_results(sample_size=4)

        analysis = summary["statistical_analysis"]
        self.assertEqual(analysis["systems_compared"], 4)
        self.assertEqual(analysis["primary_system"], "optimized_rag")
        self.assertEqual(analysis["pairwise"]["family_size"], 3)
        self.assertTrue(analysis["optimized_parameters_frozen_before_test"])
        self.assertTrue(summary["statistical_analysis_saved"])
        self.assertEqual(summary["run_id"], "test-run-id")
        self.assertFalse(summary["is_full_run"])
        self.assertTrue(summary["is_authoritative"])
        self.assertEqual(summary["rows_saved"], 16)
        self.assertEqual(summary["rows_added_this_attempt"], 16)
        iter_examples.assert_called_once_with(
            split="test",
            sample_size=4,
            seed=42,
        )
        get_client.assert_called_once_with()
        start_run.assert_called_once()
        start_arguments = start_run.call_args.kwargs
        self.assertEqual(start_arguments["dataset"], "docfinqa")
        self.assertEqual(start_arguments["experiment_name"], "baseline")
        self.assertEqual(start_arguments["split"], "test")
        self.assertEqual(start_arguments["model"], "gpt-4o-mini")
        self.assertEqual(start_arguments["expected_rows"], 16)
        self.assertEqual(start_arguments["parameters"]["sample_size"], 4)
        self.assertEqual(start_arguments["parameters"]["sample_seed"], 42)
        self.assertEqual(
            start_arguments["parameters"]["question_ids"],
            ["0", "1", "2", "3"],
        )
        self.assertEqual(
            start_arguments["parameters"]["baseline_methods"],
            ["no_context", "full_document", "naive_rag", "optimized_rag"],
        )
        self.assertEqual(log_results.call_count, 1)
        self.assertEqual(len(saved_rows), 16)
        full_document_rows = [
            row
            for row in saved_rows
            if row["generation_metrics"].get("baseline") == "full_document"
        ]
        self.assertEqual(len(full_document_rows), 4)
        self.assertTrue(
            all(
                row["generation_metrics"]["prompt_was_truncated"]
                is False
                for row in full_document_rows
            )
        )
        self.assertTrue(
            all(
                row["run_id"] == "test-run-id"
                and row["result_key"]
                for row in saved_rows
            )
        )
        update_progress.assert_called_once_with(
            run_id="test-run-id",
            questions_processed=4,
            rows_saved=16,
        )
        save_statistics.assert_called_once()
        self.assertEqual(
            save_statistics.call_args.kwargs["run_id"],
            "test-run-id",
        )
        self.assertEqual(
            save_statistics.call_args.kwargs["analysis_name"],
            STAGE_3_ANALYSIS_NAME,
        )
        self.assertEqual(save_statistics.call_args.kwargs["source_split"], "test")
        complete_run.assert_called_once_with(
            run_id="test-run-id",
            questions_processed=4,
            rows_saved=16,
            is_full_run=False,
            is_authoritative=True,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("app.rag.pipeline.get_best_rag_parameters")
    def test_stage_three_rejects_non_shortlist_dev_parameters(self, get_best):
        get_best.return_value = {
            "source_experiment": "legacy_dev_experiment",
        }

        with self.assertRaisesRegex(RuntimeError, "dev shortlist sweep"):
            get_baseline_results()

    @patch("app.rag.pipeline.full_rag_parameter_sweep")
    @patch("app.rag.pipeline.get_retrieval_shortlist")
    def test_dev_sweep_loads_exact_saved_three(
        self,
        get_shortlist,
        run_sweep,
    ):
        candidates = [
            {
                "chunk_size": 256 + index,
                "strategy": "section",
                "retrieval_method": "hybrid",
                "top_k": 5,
                "reranker_enabled": False,
                "reranker_pool_size": None,
            }
            for index in range(RETRIEVAL_SHORTLIST_SIZE)
        ]
        get_shortlist.return_value = {
            "source_experiment": "top_chunks_evidence_sweep",
            "source_split": "train",
            "questions_run": 50,
            "updated_at": "2026-07-30",
            "ranking_rule": {
                "reranker_model": "jinaai/jina-reranker-v1-tiny-en",
                "reranker_model_revision": (
                    "aca45de6945b5dc6399abcd2a9c55ded5dc9111f"
                ),
                "semantic_search_backend": "chroma_hnsw_depth_200",
            },
            "candidates": candidates,
        }
        run_sweep.return_value = {"rows_processed": 0}

        summary = full_rag_shortlist_sweep()

        self.assertEqual(
            len(run_sweep.call_args.kwargs["candidate_configs"]),
            RETRIEVAL_SHORTLIST_SIZE,
        )
        self.assertEqual(run_sweep.call_args.kwargs["split"], "dev")
        self.assertEqual(
            run_sweep.call_args.kwargs["experiment_name"],
            DEV_SHORTLIST_EXPERIMENT,
        )
        self.assertEqual(
            run_sweep.call_args.kwargs["winner_source_split"],
            "dev",
        )
        self.assertEqual(
            summary["retrieval_shortlist_source"]["questions_run"],
            50,
        )

    @patch("app.rag.pipeline.get_retrieval_shortlist", return_value=None)
    def test_dev_sweep_requires_completed_train_shortlist(self, _shortlist):
        with self.assertRaisesRegex(
            RuntimeError,
            "Run POST /run-chunk-rag first",
        ):
            full_rag_shortlist_sweep()


if __name__ == "__main__":
    unittest.main()
