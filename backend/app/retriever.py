import re
from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits a long document into overlapping text chunks.

    This is a simple character-based chunker for Week 1.
    Later, this will be replaced with token-aware and section-aware chunking.
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def tokenize(text: str) -> List[str]:
    """
    Lowercases text and extracts simple word tokens.
    """

    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def score_chunk(question: str, chunk: str) -> int:
    """
    Scores a chunk by counting how many question words appear in the chunk.
    This is a very simple retrieval baseline.
    """

    question_words = set(tokenize(question))
    chunk_words = set(tokenize(chunk))

    common_words = question_words.intersection(chunk_words)

    return len(common_words)

def retrieve_top_k_chunks(
    question: str,
    document_text: str,
    top_k: int = 3,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict]:
    """
    Retrieves the top-k chunks from the document using simple keyword overlap.
    """

    chunks = chunk_text(
        text=document_text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    scored_chunks = []

    for index, chunk in enumerate(chunks):
        score = score_chunk(question, chunk)

        scored_chunks.append({
            "chunk_id": index,
            "score": score,
            "text": chunk
        })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return scored_chunks[:top_k]