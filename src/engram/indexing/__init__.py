"""Indexing module for post-ingestion processing.

Provides:
- KB summary generation for query enrichment
"""

from engram.indexing.kb_summary import (
    KBSummary,
    KBSummaryGenerator,
    KBSummaryStore,
)

__all__ = [
    "KBSummary",
    "KBSummaryGenerator",
    "KBSummaryStore",
]
