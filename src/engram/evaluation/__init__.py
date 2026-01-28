"""Evaluation module for Engram quality metrics."""

from engram.evaluation.evaluator import (
    EngramClient,
    EngramEvaluator,
    EvaluationResult,
    EvaluationSummary,
    JudgeLLM,
)
from engram.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    evaluate_retrieval,
    format_metrics,
)
from engram.evaluation.runner import EvaluationRunner, GoldenQuery, QueryResult

__all__ = [
    # v4.2 Test set evaluation
    "EngramEvaluator",
    "EvaluationResult",
    "EvaluationSummary",
    "EngramClient",
    "JudgeLLM",
    # v5 Retrieval metrics
    "RetrievalMetrics",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "evaluate_retrieval",
    "aggregate_metrics",
    "format_metrics",
    "EvaluationRunner",
    "GoldenQuery",
    "QueryResult",
]
