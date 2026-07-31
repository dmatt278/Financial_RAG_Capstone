import os
import re
from functools import lru_cache


DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_INFERENCE_DEVICE = "auto"


def _get_torch():
    import torch

    return torch


def get_inference_device() -> str:
    """Selects CUDA when available while keeping an explicit CPU fallback."""

    configured_device = os.getenv(
        "RAG_DEVICE",
        DEFAULT_INFERENCE_DEVICE,
    ).strip().lower()
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", configured_device):
        raise RuntimeError(
            "RAG_DEVICE must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'."
        )

    torch = _get_torch()
    cuda_available = bool(torch.cuda.is_available())

    if configured_device == "auto":
        return "cuda" if cuda_available else "cpu"

    if configured_device.startswith("cuda") and not cuda_available:
        raise RuntimeError(
            "RAG_DEVICE requests CUDA, but PyTorch cannot access a CUDA GPU. "
            "Check the RunPod PyTorch environment before starting the sweep."
        )

    return configured_device


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    """
    Loads the fixed sentence-transformers model once for uniform experiments.
    """

    device = get_inference_device()
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            DEFAULT_EMBEDDING_MODEL,
            trust_remote_code=True,
            device=device,
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
