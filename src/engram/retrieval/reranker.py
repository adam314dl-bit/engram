"""Reranker module supporting BGE and Jina cross-encoder models.

v5: Default is BGE-reranker-v2-m3 (better Russian performance).
Supports fallback to Jina Reranker v3 for compatibility.
"""

# IMPORTANT: Set offline mode BEFORE any HuggingFace imports
# This prevents HTTP requests during model.rerank() calls
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from engram.config import settings

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker
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


def _is_bge_model(model_name: str) -> bool:
    """Check if model is a BGE reranker."""
    return "bge-reranker" in model_name.lower() or "baai" in model_name.lower()


def _is_jina_model(model_name: str) -> bool:
    """Check if model is a Jina reranker."""
    return "jina" in model_name.lower()


@lru_cache(maxsize=1)
def get_bge_reranker() -> "FlagReranker":
    """
    Get or create cached BGE reranker (singleton with lazy loading).

    Returns:
        FlagReranker instance
    """
    if not settings.reranker_enabled:
        raise RuntimeError("Reranker is disabled in settings")

    logger.info(f"Loading BGE reranker model: {settings.reranker_model}")

    import torch
    from FlagEmbedding import FlagReranker

    # Determine device
    device = settings.reranker_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Load BGE reranker (use devices=[device] to prevent multi-GPU pooling)
    model = FlagReranker(
        settings.reranker_model,
        use_fp16=settings.reranker_use_fp16 and device != "cpu",
        devices=[device],  # Single device list prevents multi-process pool
    )

    logger.info(f"BGE reranker loaded successfully on {device}")
    return model


@lru_cache(maxsize=1)
def get_jina_reranker() -> "AutoModel":
    """
    Get or create cached Jina reranker (singleton with lazy loading).

    Returns:
        Jina Reranker v3 model instance
    """
    if not settings.reranker_enabled:
        raise RuntimeError("Reranker is disabled in settings")

    logger.info(f"Loading Jina reranker model: {settings.reranker_model}")

    import torch
    from transformers import AutoModel

    # Determine device
    device = settings.reranker_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Load Jina Reranker v3 (try local cache first)
    try:
        model = AutoModel.from_pretrained(
            settings.reranker_model,
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    except OSError:
        # Model not cached - temporarily allow network access to download
        logger.info("Model not in cache, downloading from HuggingFace...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try:
            model = AutoModel.from_pretrained(
                settings.reranker_model,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        finally:
            # Re-enable offline mode after download
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Move to device
    model = model.to(device)
    model.eval()

    logger.info(f"Jina reranker loaded successfully on {device}")
    return model


def _rerank_bge(
    query: str,
    candidates: list[tuple[str, str, float]],
    top_k: int,
) -> list[RerankedItem]:
    """Rerank using BGE reranker."""
    import time

    model = get_bge_reranker()

    # Build query-document pairs
    pairs = [[query, content] for _, content, _ in candidates]

    # Log stats
    avg_len = sum(len(p[1]) for p in pairs) / len(pairs) if pairs else 0
    logger.debug(f"BGE reranking {len(pairs)} docs, avg length: {avg_len:.0f} chars")

    # Compute scores
    start = time.perf_counter()
    scores = model.compute_score(pairs, normalize=True)
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug(f"BGE reranker inference took {elapsed:.1f}ms")

    # Handle single result case
    if not isinstance(scores, list):
        scores = [scores]

    # Build results with scores
    scored_items = []
    for i, (id_, content, original_score) in enumerate(candidates):
        scored_items.append((
            id_,
            content,
            float(scores[i]),
            original_score,
            i,  # original rank
        ))

    # Sort by rerank score descending
    scored_items.sort(key=lambda x: x[2], reverse=True)

    # Return top_k
    return [
        RerankedItem(
            id=item[0],
            content=item[1],
            rerank_score=item[2],
            original_score=item[3],
            original_rank=item[4],
        )
        for item in scored_items[:top_k]
    ]


def _rerank_jina(
    query: str,
    candidates: list[tuple[str, str, float]],
    top_k: int,
) -> list[RerankedItem]:
    """Rerank using Jina reranker."""
    import time

    model = get_jina_reranker()

    # Extract documents
    documents = [content for _, content, _ in candidates]

    # Log stats
    avg_len = sum(len(d) for d in documents) / len(documents) if documents else 0
    logger.debug(f"Jina reranking {len(documents)} docs, avg length: {avg_len:.0f} chars")

    # Use Jina's rerank method
    start = time.perf_counter()
    results = model.rerank(query, documents, top_n=min(top_k, len(candidates)))
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug(f"Jina reranker inference took {elapsed:.1f}ms")

    # Build output
    reranked_items = []
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


def rerank(
    query: str,
    candidates: list[tuple[str, str, float]],
    top_k: int | None = None,
) -> list[RerankedItem]:
    """
    Rerank candidates using cross-encoder.

    Automatically selects BGE or Jina based on model name in settings.

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

    # Select reranker based on model name
    if _is_bge_model(settings.reranker_model):
        return _rerank_bge(query, candidates, top_k)
    elif _is_jina_model(settings.reranker_model):
        return _rerank_jina(query, candidates, top_k)
    else:
        # Default to BGE for unknown models
        logger.warning(f"Unknown reranker model type: {settings.reranker_model}, trying BGE")
        return _rerank_bge(query, candidates, top_k)


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
        if _is_bge_model(settings.reranker_model):
            get_bge_reranker()
        else:
            get_jina_reranker()
        return True
    except Exception as e:
        logger.warning(f"Reranker not available: {e}")
        return False


def clear_reranker_cache() -> None:
    """Clear the cached reranker models (useful for testing)."""
    get_bge_reranker.cache_clear()
    get_jina_reranker.cache_clear()


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
        if _is_bge_model(settings.reranker_model):
            get_bge_reranker()
        else:
            get_jina_reranker()
        logger.info("Reranker model ready")
    except Exception as e:
        logger.warning(f"Failed to preload reranker: {e}")


def get_reranker():
    """
    Get the appropriate reranker based on settings.

    Backwards-compatible function that returns BGE or Jina reranker.

    Returns:
        Reranker model instance
    """
    if _is_bge_model(settings.reranker_model):
        return get_bge_reranker()
    else:
        return get_jina_reranker()
