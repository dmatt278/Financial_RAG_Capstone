import importlib.util
import os
import sys
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


from app.rag import embedder, retriever  # noqa: E402


class InferenceDeviceTests(unittest.TestCase):
    def setUp(self):
        embedder._get_sentence_transformer.cache_clear()

    def tearDown(self):
        embedder._get_sentence_transformer.cache_clear()

    def test_auto_selects_cuda_when_available(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
        )

        with patch.dict(os.environ, {"RAG_DEVICE": "auto"}):
            with patch("app.rag.embedder._get_torch", return_value=torch):
                self.assertEqual(embedder.get_inference_device(), "cuda")

    def test_auto_falls_back_to_cpu(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        with patch.dict(os.environ, {"RAG_DEVICE": "auto"}):
            with patch("app.rag.embedder._get_torch", return_value=torch):
                self.assertEqual(embedder.get_inference_device(), "cpu")

    def test_explicit_cuda_fails_when_unavailable(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        with patch.dict(os.environ, {"RAG_DEVICE": "cuda"}):
            with patch("app.rag.embedder._get_torch", return_value=torch):
                with self.assertRaisesRegex(RuntimeError, "cannot access"):
                    embedder.get_inference_device()

    @patch("app.rag.embedder.get_inference_device", return_value="cuda")
    def test_embedding_model_is_created_on_selected_device(self, _get_device):
        sentence_transformers = sys.modules["sentence_transformers"]
        constructor = MagicMock()

        with patch.object(
            sentence_transformers,
            "SentenceTransformer",
            constructor,
            create=True,
        ):
            model = embedder._get_sentence_transformer()

        self.assertIs(model, constructor.return_value)
        constructor.assert_called_once_with(
            embedder.DEFAULT_EMBEDDING_MODEL,
            trust_remote_code=True,
            device="cuda",
        )


class GpuInferencePathTests(unittest.TestCase):
    def setUp(self):
        retriever.get_reranker.cache_clear()
        retriever._cached_query_embedding.cache_clear()

    def tearDown(self):
        retriever.get_reranker.cache_clear()
        retriever._cached_query_embedding.cache_clear()

    @patch("app.rag.retriever.get_inference_device", return_value="cuda")
    @patch("app.rag.retriever.CrossEncoder")
    def test_reranker_is_created_on_selected_device(
        self,
        cross_encoder,
        _get_device,
    ):
        model = retriever.get_reranker()

        self.assertIs(model, cross_encoder.return_value)
        cross_encoder.assert_called_once_with(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
        )

    @patch("app.rag.retriever.embed_queries", return_value=[[0.1, 0.2]])
    def test_repeated_question_embedding_is_reused(self, embed_queries):
        collection = MagicMock()
        collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

        retriever.semantic_search(collection, "question", 5, None)
        retriever.semantic_search(collection, "question", 10, None)
        retriever.semantic_search(collection, "different question", 5, None)

        self.assertEqual(embed_queries.call_count, 2)
        self.assertEqual(
            collection.query.call_args_list[0].kwargs["query_embeddings"],
            [[0.1, 0.2]],
        )

    @patch("app.rag.retriever.get_reranker")
    def test_reranker_uses_configured_batch_size(self, get_reranker):
        get_reranker.return_value.predict.return_value = [0.2, 0.8]
        chunks = [
            {"id": "one", "text": "first"},
            {"id": "two", "text": "second"},
        ]

        with patch.dict(os.environ, {"RERANKER_BATCH_SIZE": "4"}):
            ranked = retriever.cross_encoder_reranker(
                chunks,
                "question",
                top_k=1,
            )

        get_reranker.return_value.predict.assert_called_once_with(
            [("question", "first"), ("question", "second")],
            batch_size=4,
            show_progress_bar=False,
        )
        self.assertEqual(ranked[0]["id"], "two")

    def test_invalid_reranker_batch_size_fails(self):
        with patch.dict(os.environ, {"RERANKER_BATCH_SIZE": "0"}):
            with self.assertRaisesRegex(RuntimeError, "positive integer"):
                retriever.get_reranker_batch_size()


if __name__ == "__main__":
    unittest.main()
