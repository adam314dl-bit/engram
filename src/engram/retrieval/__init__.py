"""Retrieval layer for Engram."""

from engram.retrieval.embeddings import (
    EmbeddingService,
    cosine_similarity,
    cosine_similarity_batch,
    embed,
    embed_batch,
    get_embedding_service,
)
from engram.retrieval.hybrid_search import (
    HybridSearch,
    ScoredEpisode,
    ScoredMemory,
    hybrid_search,
    reciprocal_rank_fusion,
)
from engram.retrieval.pipeline import (
    RetrievalPipeline,
    RetrievalResult,
    retrieve,
)
from engram.retrieval.spreading_activation import (
    ActivatedConcept,
    ActivationResult,
    SpreadingActivation,
    spread_activation,
)

__all__ = [
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    "embed",
    "embed_batch",
    "cosine_similarity",
    "cosine_similarity_batch",
    # Spreading activation
    "SpreadingActivation",
    "ActivatedConcept",
    "ActivationResult",
    "spread_activation",
    # Hybrid search
    "HybridSearch",
    "ScoredMemory",
    "ScoredEpisode",
    "hybrid_search",
    "reciprocal_rank_fusion",
    # Pipeline
    "RetrievalPipeline",
    "RetrievalResult",
    "retrieve",
]
