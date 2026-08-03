import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock, call, patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE_PATH = BACKEND_ROOT / "app" / "rag" / "vector_store.py"


def _module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_vector_store_module():
    replacements = {
        "app.data.data_loader": _module(
            "app.data.data_loader",
            iter_unique_documents=Mock(return_value=iter(())),
        ),
        "app.rag.chunker": _module(
            "app.rag.chunker",
            chunk_document=Mock(return_value=[]),
        ),
        "app.rag.embedder": _module(
            "app.rag.embedder",
            embed_documents=Mock(return_value=[]),
        ),
    }
    spec = importlib.util.spec_from_file_location(
        "vector_store_under_test",
        VECTOR_STORE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class VectorStoreCacheTests(unittest.TestCase):
    def setUp(self):
        self.vector_store = _load_vector_store_module()

    def test_persistent_client_is_reused_for_the_same_path(self):
        client = MagicMock()
        chromadb = _module(
            "chromadb",
            PersistentClient=Mock(return_value=client),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            persist_dir = Path(temp_dir) / "nested" / "chroma"
            with patch.dict(
                os.environ,
                {"CHROMA_PERSIST_DIR": str(persist_dir)},
            ), patch.dict(sys.modules, {"chromadb": chromadb}):
                first = self.vector_store.get_chroma_client()
                second = self.vector_store.get_chroma_client()

            self.assertTrue(persist_dir.is_dir())

        self.assertIs(first, client)
        self.assertIs(second, client)
        chromadb.PersistentClient.assert_called_once_with(
            path=str(Path(persist_dir).resolve()),
        )

    def test_different_persistence_paths_use_different_clients(self):
        first_client = MagicMock()
        second_client = MagicMock()
        chromadb = _module(
            "chromadb",
            PersistentClient=Mock(
                side_effect=[first_client, second_client],
            ),
        )

        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            with patch.dict(sys.modules, {"chromadb": chromadb}):
                with patch.dict(
                    os.environ,
                    {"CHROMA_PERSIST_DIR": first_dir},
                ):
                    first = self.vector_store.get_chroma_client()
                with patch.dict(
                    os.environ,
                    {"CHROMA_PERSIST_DIR": second_dir},
                ):
                    second = self.vector_store.get_chroma_client()

        self.assertIs(first, first_client)
        self.assertIs(second, second_client)
        self.assertEqual(chromadb.PersistentClient.call_count, 2)

    def test_collection_is_reused_for_the_same_path_and_name(self):
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection

        with tempfile.TemporaryDirectory() as persist_dir:
            with patch.dict(
                os.environ,
                {"CHROMA_PERSIST_DIR": persist_dir},
            ), patch.object(
                self.vector_store,
                "get_chroma_client",
                return_value=client,
            ):
                first = self.vector_store.get_collection("chunks")
                second = self.vector_store.get_collection("chunks")

        self.assertIs(first, collection)
        self.assertIs(second, collection)
        client.get_or_create_collection.assert_called_once_with(
            name="chunks",
            metadata={
                "description": "DocFinQA chunks for Week 2 semantic retrieval",
            },
        )

    def test_reset_replaces_only_the_targeted_cached_collection(self):
        client = MagicMock()
        old_target = MagicMock()
        untouched = MagicMock()
        new_target = MagicMock()
        client.get_or_create_collection.side_effect = [
            old_target,
            untouched,
            new_target,
        ]

        with tempfile.TemporaryDirectory() as persist_dir:
            with patch.dict(
                os.environ,
                {"CHROMA_PERSIST_DIR": persist_dir},
            ), patch.object(
                self.vector_store,
                "get_chroma_client",
                return_value=client,
            ):
                self.assertIs(
                    self.vector_store.get_collection("target"),
                    old_target,
                )
                self.assertIs(
                    self.vector_store.get_collection("untouched"),
                    untouched,
                )
                self.assertIs(
                    self.vector_store.reset_collection("target"),
                    new_target,
                )
                self.assertIs(
                    self.vector_store.get_collection("target"),
                    new_target,
                )
                self.assertIs(
                    self.vector_store.get_collection("untouched"),
                    untouched,
                )

        client.delete_collection.assert_called_once_with("target")
        self.assertEqual(
            client.get_or_create_collection.call_args_list,
            [
                call(
                    name="target",
                    metadata={
                        "description": "DocFinQA chunks for Week 2 semantic retrieval",
                    },
                ),
                call(
                    name="untouched",
                    metadata={
                        "description": "DocFinQA chunks for Week 2 semantic retrieval",
                    },
                ),
                call(name="target"),
            ],
        )

    def test_reset_replaces_cached_collection_when_delete_reports_missing(self):
        client = MagicMock()
        old_collection = MagicMock()
        new_collection = MagicMock()
        client.get_or_create_collection.side_effect = [
            old_collection,
            new_collection,
        ]
        client.delete_collection.side_effect = RuntimeError("missing")

        with tempfile.TemporaryDirectory() as persist_dir:
            with patch.dict(
                os.environ,
                {"CHROMA_PERSIST_DIR": persist_dir},
            ), patch.object(
                self.vector_store,
                "get_chroma_client",
                return_value=client,
            ):
                self.vector_store.get_collection("chunks")
                reset_collection = self.vector_store.reset_collection("chunks")
                cached_collection = self.vector_store.get_collection("chunks")

        self.assertIs(reset_collection, new_collection)
        self.assertIs(cached_collection, new_collection)
        self.assertEqual(client.get_or_create_collection.call_count, 2)


if __name__ == "__main__":
    unittest.main()
