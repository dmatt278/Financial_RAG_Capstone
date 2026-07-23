from functools import lru_cache


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_BATCH_SIZE = 8


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    """
    Loads the fixed sentence-transformers model once for uniform experiments.
    """

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen/Qwen3-Embedding-0.6B. "
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
    Embeds retrieval questions using Qwen's query instruction.
    """

    model = _get_sentence_transformer()
    return model.encode(
        texts,
        prompt_name="query",
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
    ).tolist()
