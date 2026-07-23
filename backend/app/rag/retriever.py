import re
from typing import Any, Dict, List
from typing import Literal
from app.data.data_loader import load_docfinqa_example
from app.rag.chunker import chunk_document
from app.rag.embedder import embed_queries
from app.rag.vector_store import DEFAULT_COLLECTION_NAME, get_collection
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from functools import lru_cache
from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


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
    result = collection.query(
        query_embeddings=embed_queries([question]),
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _format_chroma_results(result)


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
    scores = model.predict(combined)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    #need to sort by score
    reranked = sorted(
        chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]
