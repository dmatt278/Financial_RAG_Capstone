from functools import lru_cache


DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_EMBEDDING_BATCH_SIZE = 8


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    """
    Loads the fixed sentence-transformers model once for uniform experiments.
    """

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            DEFAULT_EMBEDDING_MODEL,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load jinaai/jina-embeddings-v2-small-en. "
            "Install backend requirements and make sure the model is available "
            "before indexing or querying Chroma."
        ) from exc


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embeds document chunks for storage in Chroma.
    """

    model = _get_sentence_transformer()
    return model.encode(
        texts,
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
    ).tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """
    Embeds retrieval questions for comparison with stored document chunks.
    """

    model = _get_sentence_transformer()
    return model.encode(
        texts,
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
    ).tolist()
