"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_max_concurrent: int = 16
    llm_timeout: float = 120.0

    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "engram_password"
    neo4j_database: str = "neo4j"

    # Embeddings Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

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
    retrieval_top_k: int = 10
    retrieval_bm25_k: int = 20
    retrieval_vector_k: int = 20

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False


# Global settings instance
settings = Settings()
