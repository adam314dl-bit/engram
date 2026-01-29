"""Retrieval layer for Engram."""

# v5: BGE-M3 embeddings (primary)
from engram.embeddings.bge_service import (
    BGEEmbeddingService,
    embed_batch_bge,
    embed_bge,
    get_bge_embedding_service,
)

# Legacy embeddings (kept for backwards compatibility and utility functions)
from engram.retrieval.embeddings import (
    EmbeddingService,
    cosine_similarity,
    cosine_similarity_batch,
    embed,
    embed_batch,
    get_embedding_service,
)
from engram.retrieval.fusion import (
    FusedResult,
    RankedItem,
    normalize_scores,
    reciprocal_rank_fusion,
    rrf_scores_only,
    weighted_rrf,
)
from engram.retrieval.hybrid_search import (
    HybridSearch,
    ScoredEpisode,
    ScoredMemory,
    hybrid_search,
)
from engram.retrieval.observability import (
    ChunkTrace,
    RetrievalTrace,
    StepTrace,
    create_trace,
)
from engram.retrieval.path_retrieval import (
    PathBasedRetriever,
    PathRetrievalResult,
)
from engram.retrieval.pipeline import (
    RetrievalPipeline,
    RetrievalResult,
    retrieve,
)
from engram.retrieval.traced_retriever import TracedRetriever
from engram.retrieval.reranker import (
    RerankedItem,
    clear_reranker_cache,
    get_reranker,
    is_reranker_available,
    rerank,
    rerank_with_fallback,
)
from engram.retrieval.spreading_activation import (
    ActivatedConcept,
    ActivationResult,
    SpreadingActivation,
    spread_activation,
)

__all__ = [
    # BGE-M3 Embeddings (v5 - primary)
    "BGEEmbeddingService",
    "get_bge_embedding_service",
    "embed_bge",
    "embed_batch_bge",
    # Legacy Embeddings (backwards compatibility)
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
    # Fusion
    "RankedItem",
    "FusedResult",
    "reciprocal_rank_fusion",
    "rrf_scores_only",
    "normalize_scores",
    "weighted_rrf",
    # Reranker
    "RerankedItem",
    "get_reranker",
    "rerank",
    "rerank_with_fallback",
    "is_reranker_available",
    "clear_reranker_cache",
    # Hybrid search
    "HybridSearch",
    "ScoredMemory",
    "ScoredEpisode",
    "hybrid_search",
    # Path-based retrieval (v4.5)
    "PathBasedRetriever",
    "PathRetrievalResult",
    # Observability (v4.5)
    "ChunkTrace",
    "StepTrace",
    "RetrievalTrace",
    "create_trace",
    "TracedRetriever",
    # Pipeline
    "RetrievalPipeline",
    "RetrievalResult",
    "retrieve",
]
