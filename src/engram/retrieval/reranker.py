"""Jina Reranker v3 for improved retrieval precision.

Uses jinaai/jina-reranker-v3 model which supports Russian
and provides strong cross-lingual reranking capabilities.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from engram.config import settings

if TYPE_CHECKING:
    from transformers import AutoModel

logger = logging.getLogger(__name__)


@dataclass
class RerankedItem:
    """Item after reranking with cross-encoder score."""

    id: str
    content: str
    rerank_score: float
    original_score: float
    original_rank: int


@lru_cache(maxsize=1)
def get_reranker() -> "AutoModel":
    """
    Get or create cached Jina reranker (singleton with lazy loading).

    The model is loaded on first call and cached for subsequent calls.

    Returns:
        Jina Reranker v3 model instance
    """
    if not settings.reranker_enabled:
        raise RuntimeError("Reranker is disabled in settings")

    logger.info(f"Loading reranker model: {settings.reranker_model}")

    import torch
    from transformers import AutoModel

    # Determine device
    device = settings.reranker_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Load Jina Reranker v3
    model = AutoModel.from_pretrained(
        settings.reranker_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # Move to device
    model = model.to(device)
    model.eval()

    logger.info(f"Reranker model loaded successfully on {device}")
    return model


def rerank(
    query: str,
    candidates: list[tuple[str, str, float]],
    top_k: int | None = None,
) -> list[RerankedItem]:
    """
    Rerank candidates using Jina cross-encoder.

    Args:
        query: Query text
        candidates: List of (id, content, original_score) tuples
        top_k: Number of top results to return (default: settings.retrieval_top_k)

    Returns:
        List of RerankedItem sorted by rerank_score descending
    """
    import time

    if not settings.reranker_enabled:
        # Return candidates as-is if reranker disabled
        return [
            RerankedItem(
                id=id_,
                content=content,
                rerank_score=score,
                original_score=score,
                original_rank=i,
            )
            for i, (id_, content, score) in enumerate(candidates)
        ][:top_k or settings.retrieval_top_k]

    if not candidates:
        return []

    top_k = top_k if top_k is not None else settings.retrieval_top_k
    model = get_reranker()

    # Extract documents from candidates
    documents = [content for _, content, _ in candidates]

    # Log stats for debugging
    avg_len = sum(len(d) for d in documents) / len(documents) if documents else 0
    logger.debug(f"Reranking {len(documents)} docs, avg length: {avg_len:.0f} chars")

    # Use Jina's rerank method (returns list of dicts with 'index' and 'relevance_score')
    start = time.perf_counter()
    results = model.rerank(query, documents, top_n=min(top_k, len(candidates)))
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug(f"Reranker inference took {elapsed:.1f}ms")

    # Build output mapping results back to original candidates
    reranked_items: list[RerankedItem] = []
    for result in results:
        idx = result["index"]
        score = result["relevance_score"]
        id_, content, original_score = candidates[idx]

        reranked_items.append(
            RerankedItem(
                id=id_,
                content=content,
                rerank_score=float(score),
                original_score=original_score,
                original_rank=idx,
            )
        )

    return reranked_items


def rerank_with_fallback(
    query: str,
    candidates: list[tuple[str, str, float]],
    top_k: int | None = None,
) -> list[RerankedItem]:
    """
    Rerank candidates with fallback to original scores on error.

    Args:
        query: Query text
        candidates: List of (id, content, original_score) tuples
        top_k: Number of top results to return

    Returns:
        List of RerankedItem sorted by best available score
    """
    try:
        return rerank(query, candidates, top_k)
    except Exception as e:
        logger.warning(f"Reranker failed, falling back to original scores: {e}")
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        return [
            RerankedItem(
                id=id_,
                content=content,
                rerank_score=score,
                original_score=score,
                original_rank=i,
            )
            for i, (id_, content, score) in enumerate(candidates)
        ][:top_k]


def is_reranker_available() -> bool:
    """
    Check if reranker is enabled and can be loaded.

    Returns:
        True if reranker is available
    """
    if not settings.reranker_enabled:
        return False

    try:
        get_reranker()
        return True
    except Exception as e:
        logger.warning(f"Reranker not available: {e}")
        return False


def clear_reranker_cache() -> None:
    """Clear the cached reranker model (useful for testing)."""
    get_reranker.cache_clear()


def preload_reranker() -> None:
    """
    Preload reranker model at startup to avoid delay on first query.

    Call this during application startup (in lifespan handler).
    Does nothing if reranker is disabled.
    """
    if not settings.reranker_enabled:
        logger.info("Reranker disabled, skipping preload")
        return

    try:
        logger.info("Preloading reranker model...")
        get_reranker()
        logger.info("Reranker model ready")
    except Exception as e:
        logger.warning(f"Failed to preload reranker: {e}")
