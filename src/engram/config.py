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
    )

    # LLM Configuration (remote OpenAI-compatible endpoint)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = "ollama"
    llm_max_concurrent: int = 16
    llm_timeout: float = 120.0
    ingestion_max_concurrent: int = 8  # Max parallel document ingestion

    # Kimi K2 Thinking specific settings
    llm_temperature: float = Field(
        default=1.0,
        description="Kimi K2 Thinking recommends 1.0"
    )
    llm_min_p: float = Field(
        default=0.01,
        description="Kimi recommended to suppress unlikely tokens"
    )

    # Anti-leakage settings
    strip_thinking: bool = Field(
        default=True,
        description="Always strip thinking tags from output"
    )
    aggressive_strip: bool = Field(
        default=False,
        description="Also strip reasoning phrases (for extraction tasks)"
    )
    use_output_markers: bool = Field(
        default=False,
        description="Use ===РЕЗУЛЬТАТ=== markers for reliable extraction"
    )

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
    activation_threshold: float = 0.3
    activation_max_hops: int = 3
    activation_rescale: float = 0.4
    activation_top_k_per_hop: int = 20

    # Consolidation Parameters
    consolidation_min_repetitions: int = 3
    consolidation_min_success_rate: float = 0.85
    consolidation_min_importance: float = 7.0
    reflection_importance_threshold: float = 150.0

    # Retrieval Parameters
    retrieval_top_k: int = 5
    retrieval_bm25_k: int = 100
    retrieval_vector_k: int = 100

    # RRF Fusion Parameters
    rrf_k: int = Field(
        default=60,
        description="RRF constant k, higher values reduce top rank dominance"
    )

    # Reranker Parameters
    reranker_enabled: bool = Field(
        default=True,
        description="Enable BGE cross-encoder reranking"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking"
    )
    reranker_candidates: int = Field(
        default=30,
        description="Number of candidates to pass to reranker"
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


def get_dev_settings() -> Settings:
    """Get development environment settings."""
    return Settings(
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_query_prefix="",
        reranker_enabled=False,
        bm25_lemmatize=False,
    )


def get_prod_settings() -> Settings:
    """Get production environment settings.

    Optimized for high-core servers with multiple GPUs (e.g., H100/H200).
    """
    return Settings(
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
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_query_prefix="",
        reranker_enabled=False,
        bm25_lemmatize=False,
        neo4j_database="neo4j_test",
    )


# Global settings instance
settings = Settings()
