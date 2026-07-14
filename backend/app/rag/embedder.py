from functools import lru_cache


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
            "Failed to load sentence-transformers/all-MiniLM-L6-v2. "
            "Install backend requirements and make sure the model is available "
            "before indexing or querying Chroma."
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds text for Chroma using all-MiniLM-L6-v2.
    """

    model = _get_sentence_transformer()
    return model.encode(texts, normalize_embeddings=True).tolist()
