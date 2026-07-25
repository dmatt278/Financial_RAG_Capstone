from functools import lru_cache
import os


DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_CPU_EMBEDDING_BATCH_SIZE = 8
DEFAULT_GPU_EMBEDDING_BATCH_SIZE = 64


def _get_device() -> str:
    """
    Uses a Runpod GPU when CUDA is available and otherwise falls back to CPU.
    """

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_embedding_batch_size() -> int:
    """
    Uses a larger batch on GPU and allows an environment override.
    """

    configured_size = os.getenv("EMBEDDING_BATCH_SIZE")
    if configured_size:
        return int(configured_size)

    if _get_device() == "cuda":
        return DEFAULT_GPU_EMBEDDING_BATCH_SIZE

    return DEFAULT_CPU_EMBEDDING_BATCH_SIZE


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
            device=_get_device(),
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
        batch_size=_get_embedding_batch_size(),
        normalize_embeddings=True,
    ).tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """
    Embeds retrieval questions for comparison with stored document chunks.
    """

    model = _get_sentence_transformer()
    return model.encode(
        texts,
        batch_size=_get_embedding_batch_size(),
        normalize_embeddings=True,
    ).tolist()
