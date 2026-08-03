import os
import re
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, List
from typing import Literal
from app.data.data_loader import load_docfinqa_example
from app.rag.chunker import chunk_document
from app.rag.embedder import embed_queries, get_inference_device
from app.rag.vector_store import DEFAULT_COLLECTION_NAME, get_collection
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_BATCH_SIZE = 16
DEFAULT_RERANKER_MODEL = "jinaai/jina-reranker-v1-tiny-en"
DEFAULT_RERANKER_REVISION = "aca45de6945b5dc6399abcd2a9c55ded5dc9111f"
SEMANTIC_SEARCH_QUERY_DEPTH = 200
SEMANTIC_SEARCH_BACKEND = "chroma_hnsw_depth_200"


def get_reranker_batch_size() -> int:
    """Returns a positive inference batch size suitable for the reranker."""

    raw_value = os.getenv(
        "RERANKER_BATCH_SIZE",
        str(DEFAULT_RERANKER_BATCH_SIZE),
    )
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "RERANKER_BATCH_SIZE must be a positive integer."
        ) from exc

    if batch_size <= 0:
        raise RuntimeError(
            "RERANKER_BATCH_SIZE must be a positive integer."
        )
    return batch_size


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        DEFAULT_RERANKER_MODEL,
        device=get_inference_device(),
        revision=DEFAULT_RERANKER_REVISION,
        trust_remote_code=True,
    )


@lru_cache(maxsize=128)
def _cached_query_embedding(question: str) -> tuple[float, ...]:
    """Embeds a repeated sweep question once without changing its ranking."""

    return tuple(embed_queries([question])[0])


def _format_chroma_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Converts Chroma query output into the chunk format used by the RAG pipeline.
    """

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    retrieved = []

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        retrieved.append(
            {
                "id": ids[index] if index < len(ids) else "",
                "chunk_id": metadata.get("chunk_id"),
                "distance": distance,
                "score": None if distance is None else 1 / (1 + distance),
                "text": document,
                "metadata": metadata,
            }
        )

    return retrieved


def get_all_chunks(
    where: dict[str, Any],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    collection=None,
) -> list[dict[str, Any]]:
    """Gets every stored chunk matching one document chunk configuration."""

    if collection is None:
        collection = get_collection(collection_name)
    result = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    chunks = []
    for chunk_id, document, metadata in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        chunk = {
            "id": chunk_id,
            "chunk_id": metadata.get("chunk_id"),
            "distance": None,
            "score": None,
            "text": document,
            "metadata": metadata,
        }
        chunks.append(chunk)
    return chunks


def rank_keyword_chunks(
    chunks: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """Builds one reusable BM25 ranking for an in-memory candidate corpus."""

    if not chunks:
        return []

    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenize(question))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    return [
        {
            **chunks[index],
            "distance": None,
            "score": float(scores[index]),
        }
        for index in ranked_indices
    ]


def fuse_ranked_chunks(
    keyword_ranking: list[dict[str, Any]],
    semantic_ranking: list[dict[str, Any]],
    retrieval_count: int,
) -> list[dict[str, Any]]:
    """Reproduces hybrid RRF while reusing precomputed base rankings."""

    fusion_depth = max(retrieval_count * 5, 60)
    mixed = {}
    for ranking in (
        keyword_ranking[:fusion_depth],
        semantic_ranking[:fusion_depth],
    ):
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk["id"]
            mixed.setdefault(chunk_id, {**chunk, "score": 0.0})
            mixed[chunk_id]["score"] += 1 / (60 + rank)

    return sorted(
        mixed.values(),
        key=lambda chunk: chunk["score"],
        reverse=True,
    )[:fusion_depth]


def score_reranker_candidates(
    question: str,
    candidate_sets: list[list[dict[str, Any]]],
) -> dict[str, float]:
    """Scores each unique question/chunk pair once across all candidate pools."""

    unique_chunks = {}
    for candidates in candidate_sets:
        for chunk in candidates:
            unique_chunks.setdefault(chunk["id"], chunk)
    if not unique_chunks:
        return {}

    chunks = list(unique_chunks.values())
    scores = get_reranker().predict(
        [(question, chunk["text"]) for chunk in chunks],
        batch_size=get_reranker_batch_size(),
        show_progress_bar=False,
    )
    flattened_scores = list(scores)
    if len(flattened_scores) != len(chunks):
        raise RuntimeError(
            "The reranker returned a different number of scores than candidates."
        )
    return {
        chunk["id"]: _reranker_score_to_float(score)
        for chunk, score in zip(chunks, flattened_scores)
    }


def _reranker_score_to_float(score) -> float:
    """Normalizes scalar tensors and one-value arrays returned by rerankers."""

    return float(score.item() if hasattr(score, "item") else score)


def apply_reranker_scores(
    chunks: list[dict[str, Any]],
    scores_by_id: dict[str, float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Ranks one candidate pool using scores computed in a shared batch."""

    scored_chunks = [
        {
            **chunk,
            "rerank_score": scores_by_id[chunk["id"]],
        }
        for chunk in chunks
    ]
    return sorted(
        scored_chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )[:top_k]


def get_parameter_sweep_rankings(
    *,
    question: str,
    max_top_k: int,
    where: dict[str, Any],
    retrieval_configs: list[tuple[str, bool, int | None]],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    collection=None,
) -> dict[str, Any]:
    """Builds all requested Stage 1 rankings from one filtered corpus load."""

    timings = {
        "corpus_load_seconds": 0.0,
        "keyword_ranking_seconds": 0.0,
        "semantic_ranking_seconds": 0.0,
        "hybrid_fusion_seconds": 0.0,
        "reranking_seconds": 0.0,
    }
    requested_methods = {config[0] for config in retrieval_configs}
    needs_keyword = bool(requested_methods & {"keyword", "hybrid"})
    needs_semantic = bool(requested_methods & {"semantic", "hybrid"})
    if collection is None:
        collection = get_collection(collection_name)

    started = perf_counter()
    all_chunks = get_all_chunks(
        where=where,
        collection_name=collection_name,
        collection=collection,
    )
    timings["corpus_load_seconds"] += perf_counter() - started

    keyword_ranking = []
    if needs_keyword:
        started = perf_counter()
        keyword_ranking = rank_keyword_chunks(all_chunks, question)
        timings["keyword_ranking_seconds"] += perf_counter() - started

    semantic_ranking = []
    if needs_semantic and all_chunks:
        semantic_depth = max(
            (
                max(
                    max_top_k,
                    int(reranker_pool_size),
                )
                if reranker_enabled
                else max_top_k
            )
            * (5 if retrieval_method == "hybrid" else 1)
            for retrieval_method, reranker_enabled, reranker_pool_size in (
                config
                for config in retrieval_configs
                if config[0] in {"semantic", "hybrid"}
            )
        )
        if "hybrid" in requested_methods:
            semantic_depth = max(semantic_depth, 60)
        started = perf_counter()
        semantic_ranking = semantic_search(
            collection,
            question,
            semantic_depth,
            where,
        )
        timings["semantic_ranking_seconds"] += perf_counter() - started

    base_candidates = {}
    hybrid_cache = {}
    for retrieval_method, reranker_enabled, reranker_pool_size in retrieval_configs:
        retrieval_count = (
            max(max_top_k, int(reranker_pool_size))
            if reranker_enabled
            else max_top_k
        )
        if retrieval_method == "keyword":
            candidates = keyword_ranking[:retrieval_count]
        elif retrieval_method == "semantic":
            candidates = semantic_ranking[:retrieval_count]
        elif retrieval_method == "hybrid":
            if retrieval_count not in hybrid_cache:
                started = perf_counter()
                hybrid_cache[retrieval_count] = fuse_ranked_chunks(
                    keyword_ranking,
                    semantic_ranking,
                    retrieval_count,
                )
                timings["hybrid_fusion_seconds"] += perf_counter() - started
            candidates = hybrid_cache[retrieval_count][:retrieval_count]
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
        base_candidates[
            (retrieval_method, reranker_enabled, reranker_pool_size)
        ] = candidates

    reranker_keys = [
        config for config in retrieval_configs if config[1]
    ]
    scores_by_id = {}
    if reranker_keys:
        started = perf_counter()
        scores_by_id = score_reranker_candidates(
            question,
            [base_candidates[key] for key in reranker_keys],
        )
        timings["reranking_seconds"] += perf_counter() - started

    rankings = {}
    for config, candidates in base_candidates.items():
        rankings[config] = (
            apply_reranker_scores(candidates, scores_by_id, max_top_k)
            if config[1]
            else [dict(chunk) for chunk in candidates[:max_top_k]]
        )

    return {
        "all_chunks": all_chunks,
        "rankings": rankings,
        "timings": timings,
    }


def get_top_k_chunks(
    question: str,
    top_k: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    where: dict[str, Any] | None = None,
    retrieval_method: Literal["keyword", "semantic", "hybrid"] = "keyword",
    reranker_enabled: bool = False,
    reranker_pool_size: int = 20,
) -> List[Dict]:
    """
    Embeds a question and retrieves the top-k most similar chunks from Chroma.
    """

    collection = get_collection(collection_name)
    retrieval_count = (
        max(top_k, reranker_pool_size) if reranker_enabled else top_k
    )

    if retrieval_method == "keyword":
        chunks = keyword_search(collection, question, retrieval_count, where)
    elif retrieval_method == "semantic":
        chunks = semantic_search(collection, question, retrieval_count, where)
    else:
        chunks = hybrid_search(collection, question, retrieval_count, where)

    chunks = chunks[:retrieval_count]
    if reranker_enabled:
        return cross_encoder_reranker(chunks, question, top_k)
    return chunks[:top_k]
 
 
def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumeric tokens (with $, %, -, . retained so
    financial tokens like '10-k', '$1.2b', '3.5%' stay intact), optionally
    drop stopwords."""
    text = text.lower()
    _TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.\%\$]*")
    tokens = _TOKEN_RE.findall(text)
    return tokens


def keyword_search(collection, question, top_k, where):
    #get all the chunks from the document for said method
    if where:
        chunks = collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
    else:
        chunks = collection.get(include=["documents", "metadatas"])

    document = chunks["documents"]
    metadata = chunks["metadatas"]
    ids = chunks["ids"]

    if not document:
        return []

    #tokenize each chunk
    tokenize_chunks = [tokenize(x) for x in document]
    #build the index
    bm25 = BM25Okapi(tokenize_chunks)
    #tokenize query
    query_token = tokenize(question)
    #score and rank
    scores = bm25.get_scores(query_token)
    ranked_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "id": ids[i],
            "chunk_id": metadata[i].get("chunk_id"),
            "distance": None,
            "text": document[i],
            "metadata": metadata[i],
            "score": float(scores[i])
        }
        for i in ranked_index
    ]


def semantic_search(collection, question, top_k, where):
    query_depth = max(top_k, SEMANTIC_SEARCH_QUERY_DEPTH)
    result = collection.query(
        query_embeddings=[list(_cached_query_embedding(question))],
        n_results=query_depth,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _format_chroma_results(result)[:top_k]


def hybrid_search(collection, question, top_k, where):
    top_k = max(top_k * 5, 60)
    keyword = keyword_search(collection, question, top_k, where)
    semantic = semantic_search(collection, question, top_k, where)

    mixed = {}

    for rank, chunk in enumerate(keyword, start=1):
        chunk_id = chunk["id"]
        mixed.setdefault(chunk_id, {**chunk, "score": 0})
        mixed[chunk_id]["score"] += 1/(60 + rank)

    for rank, chunk in enumerate(semantic, start=1):
        chunk_id = chunk["id"]
        mixed.setdefault(chunk_id, {**chunk, "score": 0})
        mixed[chunk_id]["score"] += 1/(60 + rank)

    ranked = sorted(
        mixed.values(),
        key=lambda chunk: chunk["score"],
        reverse=True
    )

    return ranked[:top_k]


def cross_encoder_reranker(chunks, question, top_k):
    if not chunks:
        return []

    model = get_reranker()
    combined = [(question, chunk["text"]) for chunk in chunks]
    scores = model.predict(
        combined,
        batch_size=get_reranker_batch_size(),
        show_progress_bar=False,
    )

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = _reranker_score_to_float(score)

    #need to sort by score
    reranked = sorted(
        chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]
