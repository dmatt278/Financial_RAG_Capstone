import importlib.util
import sys
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _stub_module(name, **attributes):
    if importlib.util.find_spec(name) is not None:
        return
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    sys.modules[name] = module


_stub_module("ijson", items=lambda *_args, **_kwargs: iter(()))
_stub_module("huggingface_hub", hf_hub_download=lambda **_kwargs: "")
_stub_module("rank_bm25", BM25Okapi=object)
_stub_module("sentence_transformers", CrossEncoder=object)

if importlib.util.find_spec("llama_index") is None:
    llama_index_stub = ModuleType("llama_index")
    llama_index_stub.__spec__ = ModuleSpec("llama_index", loader=None)
    llama_index_core_stub = ModuleType("llama_index.core")
    llama_index_core_stub.__spec__ = ModuleSpec(
        "llama_index.core",
        loader=None,
    )
    node_parser_stub = ModuleType("llama_index.core.node_parser")
    node_parser_stub.__spec__ = ModuleSpec(
        "llama_index.core.node_parser",
        loader=None,
    )
    node_parser_stub.TokenTextSplitter = object
    node_parser_stub.SentenceSplitter = object
    llama_index_core_stub.node_parser = node_parser_stub
    llama_index_stub.core = llama_index_core_stub
    sys.modules["llama_index"] = llama_index_stub
    sys.modules["llama_index.core"] = llama_index_core_stub
    sys.modules["llama_index.core.node_parser"] = node_parser_stub


from app.rag import retriever  # noqa: E402


def _chunk(chunk_id, *, embedding=None):
    chunk = {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "distance": None,
        "score": None,
        "text": f"text for {chunk_id}",
        "metadata": {"chunk_id": chunk_id},
    }
    if embedding is not None:
        chunk["embedding"] = embedding
    return chunk


def _ids(chunks):
    return [chunk["id"] for chunk in chunks]


class ParameterSweepRankingTests(unittest.TestCase):
    def setUp(self):
        retriever._cached_query_embedding.cache_clear()
        retriever.get_reranker.cache_clear()

    def tearDown(self):
        retriever._cached_query_embedding.cache_clear()
        retriever.get_reranker.cache_clear()

    @patch(
        "app.rag.retriever._cached_query_embedding",
        return_value=(0.25, 0.75),
    )
    def test_semantic_search_uses_shared_depth_and_returns_requested_prefix(
        self,
        cached_query_embedding,
    ):
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["a", "b", "c"]],
            "documents": [["A", "B", "C"]],
            "metadatas": [[
                {"chunk_id": "a"},
                {"chunk_id": "b"},
                {"chunk_id": "c"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        where = {"strategy": {"$eq": "section"}}

        result = retriever.semantic_search(
            collection,
            "question",
            2,
            where,
        )

        collection.query.assert_called_once_with(
            query_embeddings=[[0.25, 0.75]],
            n_results=200,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        cached_query_embedding.assert_called_once_with("question")
        self.assertEqual(_ids(result), ["a", "b"])

    @patch("app.rag.retriever.score_reranker_candidates")
    @patch("app.rag.retriever.semantic_search")
    @patch("app.rag.retriever.rank_keyword_chunks")
    def test_one_corpus_load_reuses_keyword_and_semantic_rankings(
        self,
        rank_keyword,
        rank_semantic,
        score_reranker,
    ):
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["a", "b", "c", "d"],
            "documents": ["A", "B", "C", "D"],
            "metadatas": [
                {"chunk_id": "a"},
                {"chunk_id": "b"},
                {"chunk_id": "c"},
                {"chunk_id": "d"},
            ],
            "embeddings": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.0, 1.0],
            ],
        }
        keyword_ranking = [_chunk(value) for value in ("a", "b", "c", "d")]
        semantic_ranking = [_chunk(value) for value in ("d", "c", "b", "a")]
        rank_keyword.return_value = keyword_ranking
        rank_semantic.return_value = semantic_ranking
        score_reranker.return_value = {
            "a": 0.1,
            "b": 0.2,
            "c": 0.3,
            "d": 0.4,
        }
        where = {"document_id": {"$eq": "document-1"}}
        configs = [
            ("keyword", False, None),
            ("keyword", True, 3),
            ("semantic", False, None),
            ("semantic", True, 3),
            ("hybrid", False, None),
        ]

        result = retriever.get_parameter_sweep_rankings(
            question="question",
            max_top_k=2,
            where=where,
            retrieval_configs=configs,
            collection=collection,
        )

        collection.get.assert_called_once_with(
            where=where,
            include=["documents", "metadatas"],
        )
        loaded_chunks = result["all_chunks"]
        rank_keyword.assert_called_once_with(loaded_chunks, "question")
        rank_semantic.assert_called_once_with(
            collection,
            "question",
            60,
            where,
        )
        score_reranker.assert_called_once()
        self.assertEqual(set(result["rankings"]), set(configs))

    @patch("app.rag.retriever.semantic_search")
    @patch("app.rag.retriever.rank_keyword_chunks")
    @patch("app.rag.retriever.get_all_chunks")
    @patch("app.rag.retriever.get_reranker_batch_size", return_value=8)
    @patch("app.rag.retriever.get_reranker")
    def test_pool_sizes_are_separate_but_reranker_pairs_are_deduplicated(
        self,
        get_reranker,
        _get_batch_size,
        get_all_chunks,
        rank_keyword,
        rank_semantic,
    ):
        chunks = [_chunk(value) for value in ("a", "b", "c", "d", "e")]
        get_all_chunks.return_value = chunks
        rank_keyword.return_value = [chunks[index] for index in (0, 1, 2, 3, 4)]
        rank_semantic.return_value = [chunks[index] for index in (1, 2, 4, 3, 0)]
        model = get_reranker.return_value
        model.predict.return_value = [0.1, 0.2, 0.9, 0.8, 0.7]
        keyword_pool_2 = ("keyword", True, 2)
        keyword_pool_4 = ("keyword", True, 4)
        semantic_pool_3 = ("semantic", True, 3)

        result = retriever.get_parameter_sweep_rankings(
            question="question",
            max_top_k=2,
            where={},
            retrieval_configs=[
                keyword_pool_2,
                keyword_pool_4,
                semantic_pool_3,
            ],
            collection=MagicMock(),
        )

        model.predict.assert_called_once_with(
            [
                ("question", "text for a"),
                ("question", "text for b"),
                ("question", "text for c"),
                ("question", "text for d"),
                ("question", "text for e"),
            ],
            batch_size=8,
            show_progress_bar=False,
        )
        self.assertEqual(_ids(result["rankings"][keyword_pool_2]), ["b", "a"])
        self.assertEqual(_ids(result["rankings"][keyword_pool_4]), ["c", "d"])
        self.assertEqual(_ids(result["rankings"][semantic_pool_3]), ["c", "e"])

    @patch("app.rag.retriever.semantic_search")
    @patch("app.rag.retriever.rank_keyword_chunks")
    @patch("app.rag.retriever.get_all_chunks")
    def test_hybrid_result_is_the_expected_rrf_prefix(
        self,
        get_all_chunks,
        rank_keyword,
        rank_semantic,
    ):
        chunks = [_chunk(value) for value in ("a", "b", "c")]
        get_all_chunks.return_value = chunks
        rank_keyword.return_value = [chunks[index] for index in (0, 1, 2)]
        rank_semantic.return_value = [chunks[index] for index in (1, 2, 0)]
        config = ("hybrid", False, None)

        result = retriever.get_parameter_sweep_rankings(
            question="question",
            max_top_k=2,
            where={},
            retrieval_configs=[config],
            collection=MagicMock(),
        )

        self.assertEqual(_ids(result["rankings"][config]), ["b", "a"])
        self.assertEqual(len(result["rankings"][config]), 2)

    @patch("app.rag.retriever._cached_query_embedding")
    @patch("app.rag.retriever.semantic_search")
    @patch("app.rag.retriever.rank_keyword_chunks")
    def test_keyword_only_does_not_request_or_require_embeddings(
        self,
        rank_keyword,
        rank_semantic,
        cached_query_embedding,
    ):
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["a", "b"],
            "documents": ["A", "B"],
            "metadatas": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        }
        rank_keyword.side_effect = lambda chunks, _question: list(chunks)
        config = ("keyword", False, None)
        where = {"strategy": {"$eq": "section"}}

        result = retriever.get_parameter_sweep_rankings(
            question="question",
            max_top_k=2,
            where=where,
            retrieval_configs=[config],
            collection=collection,
        )

        collection.get.assert_called_once_with(
            where=where,
            include=["documents", "metadatas"],
        )
        self.assertTrue(
            all("embedding" not in chunk for chunk in result["all_chunks"])
        )
        rank_semantic.assert_not_called()
        cached_query_embedding.assert_not_called()
        self.assertEqual(_ids(result["rankings"][config]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
