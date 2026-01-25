"""Query understanding and enrichment module for v4.3.

Provides:
- KB-aware query enrichment
- Multi-query generation (BM25 expansion, semantic rewrite)
- Query understanding (type, complexity classification)
"""

from engram.query.enrichment import (
    BM25Expander,
    EnrichedQuery,
    HyDEGenerator,
    QueryComplexity,
    QueryEnrichmentPipeline,
    QueryType,
    QueryUnderstanding,
    QueryUnderstandingModule,
    SemanticRewriter,
)

__all__ = [
    "QueryType",
    "QueryComplexity",
    "QueryUnderstanding",
    "EnrichedQuery",
    "QueryUnderstandingModule",
    "BM25Expander",
    "SemanticRewriter",
    "HyDEGenerator",
    "QueryEnrichmentPipeline",
]
