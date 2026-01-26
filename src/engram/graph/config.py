"""Configuration for graph quality optimization."""

from dataclasses import dataclass, field


@dataclass
class DeduplicationConfig:
    """Configuration for concept deduplication."""

    # LaBSE model for cross-lingual embeddings
    labse_model: str = "sentence-transformers/LaBSE"

    # Similarity thresholds
    auto_merge_threshold: float = 0.95  # High confidence: auto-merge
    review_threshold: float = 0.80  # Medium: create POSSIBLE_DUPLICATE edge
    possible_threshold: float = 0.60  # Low: track but don't act

    # Component weights for combined similarity
    labse_weight: float = 0.50  # Cross-lingual semantic similarity
    phonetic_weight: float = 0.25  # Transliteration-based phonetic match
    string_weight: float = 0.25  # Jaro-Winkler string similarity

    # Batch processing
    batch_size: int = 256  # LaBSE embedding batch size
    max_candidates_per_concept: int = 10  # Max duplicates to consider per concept


@dataclass
class EnrichmentConfig:
    """Configuration for semantic enrichment."""

    # LLM settings (uses existing LLM config from settings)
    max_definitions_per_batch: int = 10
    max_relations_per_concept: int = 5

    # Edge boosting
    semantic_edge_boost: float = 1.5  # Boost for is_semantic + is_universal edges

    # Caching
    cache_ttl_hours: int = 24 * 7  # 1 week cache for definitions


@dataclass
class GraphQualityConfig:
    """Combined configuration for graph quality optimization."""

    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)

    # Processing options
    dry_run: bool = False  # Preview changes without applying
    backup_before_merge: bool = True  # Create backup before destructive operations
    verbose: bool = False  # Extra logging
