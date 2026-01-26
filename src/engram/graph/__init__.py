"""Graph quality optimization module for Engram v4.4.

Provides:
- Concept deduplication (cross-lingual duplicate detection/merging)
- Semantic enrichment (world knowledge injection, edge classification)
- Concept resolution (alias→canonical mappings)
- Graph quality metrics
"""

from engram.graph.config import DeduplicationConfig, EnrichmentConfig, GraphQualityConfig
from engram.graph.models import (
    CanonicalConcept,
    ConceptStatus,
    DuplicateCandidate,
    EdgeSourceType,
    EnrichedEdge,
    MatchConfidence,
)

__all__ = [
    # Config
    "DeduplicationConfig",
    "EnrichmentConfig",
    "GraphQualityConfig",
    # Models
    "ConceptStatus",
    "MatchConfidence",
    "DuplicateCandidate",
    "CanonicalConcept",
    "EdgeSourceType",
    "EnrichedEdge",
]
