"""BGE cross-encoder reranker for improved retrieval precision.

Uses FlagEmbedding's BGE-reranker-v2-m3 model which supports Russian
and provides strong cross-lingual reranking capabilities.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from engram.config import settings

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker

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
def get_reranker() -> "FlagReranker":
    """
    Get or create cached BGE reranker (singleton with lazy loading).

    The model is loaded on first call and cached for subsequent calls.

    Returns:
        FlagReranker instance
    """
    if not settings.reranker_enabled:
        raise RuntimeError("Reranker is disabled in settings")

    logger.info(f"Loading reranker model: {settings.reranker_model}")

    from FlagEmbedding import FlagReranker

    # Load model with settings
    # use_fp16=True for faster inference on supported GPUs
    # devices=["cuda:0"] limits to single GPU (reranker doesn't need multi-GPU)
    reranker = FlagReranker(
        settings.reranker_model,
        use_fp16=True,
        devices=["cuda:0"],  # Single GPU is enough for reranking
    )

    logger.info("Reranker model loaded successfully")
    return reranker


def rerank(
    query: str,
    candidates: list[tuple[str, str, float]],
    top_k: int | None = None,
) -> list[RerankedItem]:
    """
    Rerank candidates using cross-encoder.

    Args:
        query: Query text
        candidates: List of (id, content, original_score) tuples
        top_k: Number of top results to return (default: settings.retrieval_top_k)

    Returns:
        List of RerankedItem sorted by rerank_score descending
    """
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
    reranker = get_reranker()

    # Prepare query-passage pairs
    pairs = [(query, content) for _, content, _ in candidates]

    # Get reranker scores
    scores = reranker.compute_score(pairs, normalize=True)

    # Handle single item case (reranker returns float instead of list)
    if isinstance(scores, float):
        scores = [scores]

    # Build results with both scores
    results = [
        RerankedItem(
            id=id_,
            content=content,
            rerank_score=float(score),
            original_score=original_score,
            original_rank=i,
        )
        for i, ((id_, content, original_score), score) in enumerate(zip(candidates, scores, strict=True))
    ]

    # Sort by rerank score descending
    results.sort(key=lambda x: x.rerank_score, reverse=True)
    return results[:top_k]


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
