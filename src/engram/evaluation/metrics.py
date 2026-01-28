"""Retrieval evaluation metrics for v5.

Provides standard IR metrics:
- Recall@K: Fraction of relevant items in top K
- MRR (Mean Reciprocal Rank): Rank-weighted metric
- NDCG@K (Normalized Discounted Cumulative Gain): Graded relevance metric
"""

import math
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    recall_at_k: dict[int, float] = field(default_factory=dict)  # K -> recall
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)  # K -> NDCG
    precision_at_k: dict[int, float] = field(default_factory=dict)  # K -> precision
    num_queries: int = 0
    num_relevant: int = 0
    num_retrieved: int = 0


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k: Cutoff position

    Returns:
        Recall value in [0, 1]
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant_ids)
    return hits / len(relevant_ids)


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K = |relevant ∩ retrieved[:K]| / K

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k: Cutoff position

    Returns:
        Precision value in [0, 1]
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant_ids)
    return hits / k


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (for a single query).

    MRR = 1 / rank_of_first_relevant_item

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: Set of relevant item IDs

    Returns:
        Reciprocal rank value in [0, 1]
    """
    for rank, item_id in enumerate(retrieved_ids, 1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_dcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevance_scores: Mapping of item ID to relevance score
        k: Cutoff position

    Returns:
        DCG value
    """
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k], 1):
        rel = relevance_scores.get(item_id, 0.0)
        dcg += rel / math.log2(i + 1)
    return dcg


def calculate_ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """
    Calculate Normalized DCG at K.

    NDCG@K = DCG@K / IDCG@K (ideal DCG)

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevance_scores: Mapping of item ID to relevance score
        k: Cutoff position

    Returns:
        NDCG value in [0, 1]
    """
    dcg = calculate_dcg_at_k(retrieved_ids, relevance_scores, k)

    # Calculate ideal DCG (sorted by relevance)
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = calculate_dcg_at_k(ideal_order, relevance_scores, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_ndcg_binary(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate NDCG@K with binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k: Cutoff position

    Returns:
        NDCG value in [0, 1]
    """
    relevance_scores = {item_id: 1.0 for item_id in relevant_ids}
    return calculate_ndcg_at_k(retrieved_ids, relevance_scores, k)


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """
    Calculate all retrieval metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k_values: List of K values for @K metrics (default: [1, 3, 5, 10, 20])

    Returns:
        RetrievalMetrics with all calculated values
    """
    k_values = k_values or [1, 3, 5, 10, 20]

    metrics = RetrievalMetrics(
        num_queries=1,
        num_relevant=len(relevant_ids),
        num_retrieved=len(retrieved_ids),
    )

    # Calculate metrics at each K
    for k in k_values:
        metrics.recall_at_k[k] = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
        metrics.precision_at_k[k] = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        metrics.ndcg_at_k[k] = calculate_ndcg_binary(retrieved_ids, relevant_ids, k)

    # MRR
    metrics.mrr = calculate_mrr(retrieved_ids, relevant_ids)

    return metrics


def aggregate_metrics(
    all_metrics: list[RetrievalMetrics],
) -> RetrievalMetrics:
    """
    Aggregate metrics across multiple queries.

    Args:
        all_metrics: List of per-query metrics

    Returns:
        Aggregated RetrievalMetrics with averaged values
    """
    if not all_metrics:
        return RetrievalMetrics()

    # Collect all K values across all metrics
    all_k_values = set()
    for m in all_metrics:
        all_k_values.update(m.recall_at_k.keys())
        all_k_values.update(m.precision_at_k.keys())
        all_k_values.update(m.ndcg_at_k.keys())

    aggregated = RetrievalMetrics(
        num_queries=len(all_metrics),
        num_relevant=sum(m.num_relevant for m in all_metrics),
        num_retrieved=sum(m.num_retrieved for m in all_metrics),
    )

    # Average metrics at each K
    n = len(all_metrics)
    for k in all_k_values:
        aggregated.recall_at_k[k] = sum(m.recall_at_k.get(k, 0) for m in all_metrics) / n
        aggregated.precision_at_k[k] = sum(m.precision_at_k.get(k, 0) for m in all_metrics) / n
        aggregated.ndcg_at_k[k] = sum(m.ndcg_at_k.get(k, 0) for m in all_metrics) / n

    # Average MRR
    aggregated.mrr = sum(m.mrr for m in all_metrics) / n

    return aggregated


def format_metrics(metrics: RetrievalMetrics) -> str:
    """Format metrics for display."""
    lines = [
        f"Retrieval Metrics ({metrics.num_queries} queries)",
        f"  Total relevant: {metrics.num_relevant}",
        f"  Total retrieved: {metrics.num_retrieved}",
        f"  MRR: {metrics.mrr:.4f}",
        "",
        "  K    Recall   Precision   NDCG",
        "  ---  ------   ---------   ----",
    ]

    for k in sorted(metrics.recall_at_k.keys()):
        recall = metrics.recall_at_k.get(k, 0)
        precision = metrics.precision_at_k.get(k, 0)
        ndcg = metrics.ndcg_at_k.get(k, 0)
        lines.append(f"  {k:3d}  {recall:.4f}   {precision:.4f}      {ndcg:.4f}")

    return "\n".join(lines)
