"""Configuration management using Pydantic Settings."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment presets."""

    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore deprecated settings in .env files
    )

    # LLM Configuration (remote OpenAI-compatible endpoint)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = "ollama"
    llm_max_concurrent: int = 16
    llm_timeout: float = 120.0
    ingestion_max_concurrent: int = 8  # Max parallel document ingestion

    # Language
    primary_language: str = Field(
        default="ru",
        description="Primary content language: 'ru' or 'en'"
    )

    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "engram_password"
    neo4j_database: str = "neo4j"

    # Embeddings Configuration (local HuggingFace model)
    # Dev: all-MiniLM-L6-v2 (384 dims)
    # Prod: ai-sage/Giga-Embeddings-instruct (2048 dims)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_query_prefix: str = Field(
        default="",
        description="Query prefix for asymmetric embedding models (e.g., 'Instruct: ' for GigaEmbeddings)"
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for embedding generation (increase for H100/H200)"
    )
    embedding_multi_gpu: bool = Field(
        default=False,
        description="Use multiple GPUs for embedding (requires CUDA)"
    )
    embedding_gpu_count: int = Field(
        default=1,
        description="Number of GPUs to use for embedding (0=all available)"
    )

    # Spreading Activation Parameters
    activation_decay: float = 0.85
    activation_threshold: float = 0.15  # Lowered from 0.3 to improve retrieval
    activation_max_hops: int = 3
    activation_rescale: float = 0.4
    activation_top_k_per_hop: int = 20

    # Consolidation Parameters
    consolidation_min_repetitions: int = 3
    consolidation_min_success_rate: float = 0.85
    consolidation_min_importance: float = 7.0
    reflection_importance_threshold: float = 150.0

    # Retrieval Parameters
    retrieval_top_k: int = 200
    retrieval_bm25_k: int = 200
    retrieval_vector_k: int = 200

    # Retrieval Mode
    retrieval_mode: str = Field(
        default="bm25_graph",
        description="Retrieval mode: 'bm25_graph' (default, no embeddings) or 'hybrid' (with vector search)"
    )

    # RRF Fusion Parameters
    rrf_k: int = Field(
        default=60,
        description="RRF constant k, higher values reduce top rank dominance"
    )
    rrf_bm25_weight: float = Field(
        default=0.40,
        description="BM25 weight in weighted RRF fusion"
    )
    rrf_vector_weight: float = Field(
        default=0.0,
        description="Vector search weight (disabled in bm25_graph mode)"
    )
    rrf_graph_weight: float = Field(
        default=0.40,
        description="Graph traversal weight in weighted RRF fusion"
    )

    # Reranker Parameters
    reranker_enabled: bool = Field(
        default=True,
        description="Enable Jina cross-encoder reranking"
    )
    reranker_model: str = Field(
        default="jinaai/jina-reranker-v3",
        description="Cross-encoder model for reranking (Jina v3 replaces BGE)"
    )
    reranker_candidates: int = Field(
        default=64,
        description="Number of candidates to pass to reranker (batch size for single pass)"
    )
    reranker_device: str = Field(
        default="cuda:0",
        description="Device for reranker model (cuda:0, cpu, etc.)"
    )

    # BM25 Parameters
    bm25_lemmatize: bool = Field(
        default=True,
        description="Use PyMorphy3 lemmatization for Russian BM25"
    )
    bm25_remove_stopwords: bool = Field(
        default=True,
        description="Remove Russian stopwords from BM25 queries"
    )

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # SM-2 Spaced Repetition Parameters
    sm2_initial_ef: float = Field(
        default=2.5,
        description="Initial easiness factor for new memories"
    )
    sm2_ef_min: float = Field(
        default=1.3,
        description="Minimum easiness factor"
    )
    sm2_ef_max: float = Field(
        default=3.0,
        description="Maximum easiness factor"
    )

    # ACT-R Base-Level Learning Parameters
    actr_decay_d: float = Field(
        default=0.5,
        description="ACT-R decay parameter d (higher = faster forgetting)"
    )
    actr_threshold_tau: float = Field(
        default=-2.0,
        description="ACT-R retrieval threshold tau (log scale)"
    )

    # Forgetting/Archival Parameters
    forgetting_deprioritize_threshold: float = Field(
        default=-1.0,
        description="Base-level activation below which memories are deprioritized"
    )
    forgetting_archive_threshold: float = Field(
        default=-2.5,
        description="Base-level activation below which memories are archived"
    )

    # Contradiction Detection Parameters
    contradiction_auto_resolve_gap: float = Field(
        default=0.3,
        description="Confidence gap required for auto-resolution (higher confidence wins)"
    )

    # v3.3 Quality Filtering Parameters
    chunk_quality_threshold: float = Field(
        default=0.4,
        description="Minimum quality score for chunks (0-1)"
    )
    min_chunk_words: int = Field(
        default=20,
        description="Minimum word count for quality chunks"
    )

    # v3.3 MMR (Maximal Marginal Relevance) Parameters
    mmr_enabled: bool = Field(
        default=True,
        description="Enable MMR reranking for result diversity"
    )
    mmr_lambda: float = Field(
        default=0.5,
        description="MMR balance: 1.0=pure relevance, 0.0=pure diversity"
    )
    mmr_fetch_k: int = Field(
        default=200,
        description="Number of candidates to consider for MMR"
    )

    # v3.3 Dynamic top_k Parameters
    dynamic_topk_enabled: bool = Field(
        default=False,
        description="Enable dynamic top_k based on query complexity"
    )
    topk_simple: int = Field(
        default=4,
        description="top_k for simple factoid queries"
    )
    topk_moderate: int = Field(
        default=6,
        description="top_k for moderate complexity queries"
    )
    topk_complex: int = Field(
        default=8,
        description="top_k for complex analytical queries"
    )

    # v4.4: Graph Quality Optimization Parameters
    semantic_edge_boost: float = Field(
        default=1.5,
        description="Boost factor for semantic + universal edges in spreading activation"
    )
    dedup_auto_merge_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for auto-merging duplicate concepts"
    )
    dedup_review_threshold: float = Field(
        default=0.80,
        description="Similarity threshold for creating POSSIBLE_DUPLICATE edges"
    )
    dedup_possible_threshold: float = Field(
        default=0.60,
        description="Minimum similarity to track as potential duplicate"
    )

def get_dev_settings() -> Settings:
    """Get development environment settings."""
    return Settings(
        retrieval_mode="bm25_graph",  # No vector search in dev
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_query_prefix="",
        reranker_enabled=False,
        bm25_lemmatize=False,
        mmr_enabled=False,  # Disable MMR in dev for speed
        dynamic_topk_enabled=False,  # Use fixed top_k in dev
    )


def get_prod_settings() -> Settings:
    """Get production environment settings.

    Optimized for high-core servers with multiple GPUs (e.g., H100/H200).
    Default: bm25_graph mode (no vector search).
    Set retrieval_mode='hybrid' to enable vector search.
    """
    return Settings(
        retrieval_mode="bm25_graph",  # Default: BM25 + Graph only (no embeddings)
        embedding_model="ai-sage/Giga-Embeddings-instruct",
        embedding_dimensions=2048,
        embedding_query_prefix="Instruct: Найди релевантные факты для ответа на вопрос\nQuery: ",
        embedding_batch_size=128,  # Larger batches for H100/H200
        embedding_multi_gpu=True,  # Use all available GPUs
        embedding_gpu_count=0,  # 0 = use all available
        ingestion_max_concurrent=32,  # Higher parallelism for 128+ cores
        llm_max_concurrent=32,  # Higher LLM concurrency
        reranker_enabled=True,
        bm25_lemmatize=True,
        bm25_remove_stopwords=True,
    )


def get_test_settings() -> Settings:
    """Get test environment settings."""
    return Settings(
        retrieval_mode="bm25_graph",  # No vector search in tests
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_query_prefix="",
        reranker_enabled=False,
        bm25_lemmatize=False,
        neo4j_database="neo4j_test",
        mmr_enabled=False,  # Disable MMR in tests
        dynamic_topk_enabled=False,  # Use fixed top_k in tests
    )


# Global settings instance
settings = Settings()
