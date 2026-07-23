import itertools
import os
from pathlib import Path
from typing import Any

from app.data.data_loader import iter_unique_documents
from app.rag.chunker import chunk_document
from app.rag.embedder import embed_documents


DEFAULT_COLLECTION_NAME = "docfinqa_chunks"
DEFAULT_CHROMA_PATH = "data/chroma"
DEFAULT_UPSERT_BATCH_SIZE = 1000
DEFAULT_CHUNK_STRATEGIES = ["fixed", "sentence", "section"]
DEFAULT_CHUNK_SIZES = [256, 512, 1024]


def get_chroma_path() -> str:
    """
    Returns the directory where Chroma should persist its local vector database.
    """

    return os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PATH)


def get_chroma_client():
    """
    Creates a persistent Chroma client and ensures the storage directory exists.
    """

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Run `pip install -r backend/requirements.txt` "
            "or rebuild the backend container."
        ) from exc

    persist_path = Path(get_chroma_path())
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def get_collection(collection_name: str = DEFAULT_COLLECTION_NAME):
    """
    Gets or creates the Chroma collection used to store DocFinQA chunks.
    """

    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "DocFinQA chunks for Week 2 semantic retrieval"},
    )


def reset_collection(collection_name: str = DEFAULT_COLLECTION_NAME):
    """
    Deletes and recreates a Chroma collection so indexing can start fresh.
    """

    client = get_chroma_client()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return client.get_or_create_collection(name=collection_name)


def count_docfinqa_samples(
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """
    Counts unique documents represented in Chroma metadata.
    """

    return len(get_docfinqa_document_ids(collection_name=collection_name))


def get_docfinqa_document_ids(
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[str]:
    """
    Gets unique document_ids represented in Chroma metadata.
    """

    collection = get_collection(collection_name)
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    document_ids = sorted({
        metadata["document_id"]
        for metadata in metadatas
        if metadata and "document_id" in metadata
    })

    return document_ids


def _chunk_metadata(
    document_id: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Builds the metadata stored with each chunk in Chroma.
    """

    metadata = {
        "document_id": document_id,
        "chunk_id": chunk["chunk_id"],
        "tokens": chunk.get("tokens", 0),
        "strategy": chunk.get("strategy", ""),
        "chunk_size": chunk.get("chunk_size", 0),
        "part": chunk.get("part", ""),
        "item": chunk.get("item", ""),
        "header": chunk.get("header", ""),
        "is_table": bool(chunk.get("is_table", False)),
        "start": chunk.get("start", -1),
        "end": chunk.get("end", -1),
    }

    return metadata


def _upsert_chunks_in_batches(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
) -> int:
    """
    Embeds and upserts chunks in smaller batches to stay under Chroma's limit.
    """

    inserted_chunks = 0

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_documents = documents[start:end]

        collection.upsert(
            ids=ids[start:end],
            documents=batch_documents,
            metadatas=metadatas[start:end],
            embeddings=embed_documents(batch_documents),
        )
        inserted_chunks += len(batch_documents)

    return inserted_chunks


def insert_docfinqa_chunk_sweep(
    start_index: int = 0,
    limit: int | None = None,
    strategies: list[str] | None = None,
    chunk_sizes: list[int] | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Stores every chunk strategy and chunk size combination for each unique
    DocFinQA document (deduplicated across splits, so a document shared by
    multiple questions is only chunked and embedded once).
    """

    strategies = strategies or DEFAULT_CHUNK_STRATEGIES
    chunk_sizes = chunk_sizes or DEFAULT_CHUNK_SIZES
    collection = reset_collection(collection_name) if reset else get_collection(collection_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    inserted_documents = 0
    inserted_chunks = 0
    config_results = []

    end_index = None if limit is None else start_index + limit
    document_stream = itertools.islice(iter_unique_documents(), start_index, end_index)

    for document in document_stream:
        inserted_documents += 1

        for strategy in strategies:
            for chunk_size in chunk_sizes:
                chunks = chunk_document(
                    document["document_text"],
                    strategy=strategy,
                    size=chunk_size,
                )

                config_results.append(
                    {
                        "document_id": document["document_id"],
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunks": len(chunks),
                    }
                )

                for chunk in chunks:
                    chunk_id = chunk["chunk_id"]
                    ids.append(
                        f"{document['document_id']}-{strategy}-{chunk_size}-{chunk_id}"
                    )
                    documents.append(chunk["text"])
                    metadatas.append(_chunk_metadata(document["document_id"], chunk))

                    if len(documents) >= DEFAULT_UPSERT_BATCH_SIZE:
                        inserted_chunks += _upsert_chunks_in_batches(
                            collection=collection,
                            ids=ids,
                            documents=documents,
                            metadatas=metadatas,
                        )
                        ids.clear()
                        documents.clear()
                        metadatas.clear()

    if documents:
        inserted_chunks += _upsert_chunks_in_batches(
            collection=collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    return {
        "collection": collection_name,
        "persist_path": get_chroma_path(),
        "start_index": start_index,
        "limit": limit,
        "strategies": strategies,
        "chunk_sizes": chunk_sizes,
        "inserted_documents": inserted_documents,
        "inserted_configs_per_document": len(strategies) * len(chunk_sizes),
        "inserted_chunks": inserted_chunks,
        "collection_count": collection.count(),
        "config_results": config_results,
    }
