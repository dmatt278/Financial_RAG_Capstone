import os
from pathlib import Path
from typing import Any

from app.data.data_loader import iter_docfinqa_examples
from app.rag.chunker import chunk_document
from app.rag.embedder import embed_texts


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
    Counts unique DocFinQA samples represented in Chroma metadata.
    """

    return len(get_docfinqa_sample_indexes(collection_name=collection_name))


def get_docfinqa_sample_indexes(
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[int]:
    """
    Gets unique DocFinQA source indexes represented in Chroma metadata.
    """

    collection = get_collection(collection_name)
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    source_indexes = sorted({
        metadata["source_index"]
        for metadata in metadatas
        if metadata and "source_index" in metadata
    })

    return source_indexes


def _chunk_metadata(
    example: dict[str, Any],
    split: str,
    source_index: int,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Builds the metadata stored with each chunk in Chroma.
    """

    metadata = {
        "split": split,
        "source_index": source_index,
        "question_id": example["question_id"],
        "question": example["question"],
        "gold_answer": str(example["gold_answer"]),
        "chunk_id": chunk["chunk_id"],
        "tokens": chunk.get("tokens", 0),
        "strategy": chunk.get("strategy", ""),
        "chunk_size": chunk.get("chunk_size", 0),
        "chunk_overlap": chunk.get("chunk_overlap", 0),
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
            embeddings=embed_texts(batch_documents),
        )
        inserted_chunks += len(batch_documents)

    return inserted_chunks


def insert_docfinqa_examples(
    split: str = "train",
    start_index: int = 0,
    limit: int = 10,
    strategy: str = "section",
    chunk_size: int = 512,
    overlap: int = 50,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Streams DocFinQA examples, chunks their text, embeds chunks, and stores them in Chroma.
    """

    collection = reset_collection(collection_name) if reset else get_collection(collection_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    inserted_examples = 0
    inserted_chunks = 0

    for source_index, example in enumerate(
        iter_docfinqa_examples(split=split, start_index=start_index, limit=limit),
        start=start_index,
    ):
        chunks = chunk_document(
            example["document_text"],
            strategy=strategy,
            size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            ids.append(
                f"{split}-{example['question_id']}-{strategy}-{chunk_size}-{overlap}-{chunk_id}"
            )
            documents.append(chunk["text"])
            metadatas.append(_chunk_metadata(example, split, source_index, chunk))

        inserted_examples += 1

    if documents:
        inserted_chunks = _upsert_chunks_in_batches(
            collection=collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    return {
        "collection": collection_name,
        "persist_path": get_chroma_path(),
        "split": split,
        "start_index": start_index,
        "limit": limit,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "inserted_examples": inserted_examples,
        "inserted_chunks": inserted_chunks,
        "collection_count": collection.count(),
    }


def insert_docfinqa_chunk_sweep(
    split: str = "train",
    start_index: int = 0,
    limit: int = 10,
    strategies: list[str] | None = None,
    chunk_sizes: list[int] | None = None,
    overlap: int = 50,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Stores every chunk strategy and chunk size combination for each DocFinQA example.
    """

    strategies = strategies or DEFAULT_CHUNK_STRATEGIES
    chunk_sizes = chunk_sizes or DEFAULT_CHUNK_SIZES
    collection = reset_collection(collection_name) if reset else get_collection(collection_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    inserted_examples = 0
    config_results = []

    for source_index, example in enumerate(
        iter_docfinqa_examples(split=split, start_index=start_index, limit=limit),
        start=start_index,
    ):
        inserted_examples += 1

        for strategy in strategies:
            for chunk_size in chunk_sizes:
                chunks = chunk_document(
                    example["document_text"],
                    strategy=strategy,
                    size=chunk_size,
                    overlap=overlap,
                )

                config_results.append(
                    {
                        "source_index": source_index,
                        "question_id": example["question_id"],
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunks": len(chunks),
                    }
                )

                for chunk in chunks:
                    chunk_id = chunk["chunk_id"]
                    ids.append(
                        f"{split}-{example['question_id']}-{strategy}-{chunk_size}-{overlap}-{chunk_id}"
                    )
                    documents.append(chunk["text"])
                    metadatas.append(_chunk_metadata(example, split, source_index, chunk))

    inserted_chunks = 0

    if documents:
        inserted_chunks = _upsert_chunks_in_batches(
            collection=collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    return {
        "collection": collection_name,
        "persist_path": get_chroma_path(),
        "split": split,
        "start_index": start_index,
        "limit": limit,
        "strategies": strategies,
        "chunk_sizes": chunk_sizes,
        "overlap": overlap,
        "inserted_examples": inserted_examples,
        "inserted_configs_per_example": len(strategies) * len(chunk_sizes),
        "inserted_chunks": inserted_chunks,
        "collection_count": collection.count(),
        "config_results": config_results,
    }
