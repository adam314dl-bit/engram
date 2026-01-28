"""Reciprocal Rank Fusion (RRF) for combining multiple retrieval results.

RRF is a simple but effective method to combine ranked lists from different
retrieval methods (vector search, BM25, graph traversal, etc.).

v5: Supports 4-way fusion (vector, BM25, graph, path) with source tracking.
"""

from dataclasses import dataclass, field
from typing import TypeVar

from engram.config import settings

T = TypeVar("T")


@dataclass
class RankedItem:
    """An item with its ID and score from a single ranker."""

    id: str
    score: float
    source: str  # Which ranker produced this result


@dataclass
class FusedResult:
    """Result of fusion with combined score and source tracking."""

    id: str
    fused_score: float
    sources: list[str] = field(default_factory=list)  # Which rankers contributed
    original_scores: dict[str, float] = field(default_factory=dict)  # Scores by source
    original_ranks: dict[str, int] = field(default_factory=dict)  # Ranks by source (v5)

    # v5: Convenience properties for individual source scores
    @property
    def vector_score(self) -> float | None:
        """Get vector search score if present."""
        return self.original_scores.get("vector")

    @property
    def bm25_score(self) -> float | None:
        """Get BM25 score if present."""
        return self.original_scores.get("bm25")

    @property
    def graph_score(self) -> float | None:
        """Get graph (spreading) score if present."""
        return self.original_scores.get("graph")

    @property
    def path_score(self) -> float | None:
        """Get path retrieval score if present."""
        return self.original_scores.get("path")

    @property
    def vector_rank(self) -> int | None:
        """Get vector search rank if present."""
        return self.original_ranks.get("vector")

    @property
    def bm25_rank(self) -> int | None:
        """Get BM25 rank if present."""
        return self.original_ranks.get("bm25")


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int | None = None,
    source_names: list[str] | None = None,
) -> list[FusedResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each list where item appears.

    The k constant controls how much weight is given to top-ranked items:
    - Lower k: Top ranks dominate more
    - Higher k: More uniform weighting across ranks

    Args:
        ranked_lists: List of [(item_id, score)] lists, each sorted by score desc
        k: RRF constant (default: from settings.rrf_k = 60)
        source_names: Names for each ranked list (for tracking)

    Returns:
        List of FusedResult sorted by fused_score descending
    """
    k = k if k is not None else settings.rrf_k

    if source_names is None:
        source_names = [f"source_{i}" for i in range(len(ranked_lists))]

    # Track scores, ranks, and sources for each item
    fused_scores: dict[str, float] = {}
    item_sources: dict[str, list[str]] = {}
    original_scores: dict[str, dict[str, float]] = {}
    original_ranks: dict[str, dict[str, int]] = {}

    for source_name, ranked_list in zip(source_names, ranked_lists, strict=True):
        for rank, (item_id, score) in enumerate(ranked_list):
            # RRF formula: 1 / (k + rank + 1) where rank is 0-indexed
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[item_id] = fused_scores.get(item_id, 0) + rrf_score

            # Track sources
            if item_id not in item_sources:
                item_sources[item_id] = []
            item_sources[item_id].append(source_name)

            # Track original scores
            if item_id not in original_scores:
                original_scores[item_id] = {}
            original_scores[item_id][source_name] = score

            # Track original ranks (1-indexed for human readability)
            if item_id not in original_ranks:
                original_ranks[item_id] = {}
            original_ranks[item_id][source_name] = rank + 1

    # Build results
    results = [
        FusedResult(
            id=item_id,
            fused_score=score,
            sources=item_sources[item_id],
            original_scores=original_scores[item_id],
            original_ranks=original_ranks[item_id],
        )
        for item_id, score in fused_scores.items()
    ]

    # Sort by fused score descending
    results.sort(key=lambda x: x.fused_score, reverse=True)
    return results


def rrf_scores_only(
    ranked_lists: list[list[tuple[str, float]]],
    k: int | None = None,
) -> dict[str, float]:
    """
    Simple RRF that returns just the fused scores.

    Compatibility function for existing code.

    Args:
        ranked_lists: List of [(item_id, score)] lists, each sorted by score desc
        k: RRF constant (default: from settings.rrf_k)

    Returns:
        Dictionary of item_id -> fused_score
    """
    k = k if k is not None else settings.rrf_k

    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked_list):
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[item_id] = fused_scores.get(item_id, 0) + rrf_score

    return fused_scores


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        scores: Dictionary of item_id -> score

    Returns:
        Normalized scores
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    range_score = max_score - min_score

    if range_score == 0:
        return {k: 1.0 for k in scores}

    return {k: (v - min_score) / range_score for k, v in scores.items()}


def weighted_rrf(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    k: int | None = None,
) -> dict[str, float]:
    """
    Weighted RRF where each source has a different weight.

    Args:
        ranked_lists: List of [(item_id, score)] lists
        weights: Weight for each ranked list (should sum to 1.0 for normalization)
        k: RRF constant

    Returns:
        Dictionary of item_id -> weighted fused score
    """
    k = k if k is not None else settings.rrf_k

    if len(ranked_lists) != len(weights):
        raise ValueError("Number of ranked lists must match number of weights")

    fused_scores: dict[str, float] = {}

    for weight, ranked_list in zip(weights, ranked_lists, strict=True):
        for rank, (item_id, _score) in enumerate(ranked_list):
            rrf_score = weight / (k + rank + 1)
            fused_scores[item_id] = fused_scores.get(item_id, 0) + rrf_score

    return fused_scores
