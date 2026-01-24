"""Evaluation module for Engram quality metrics."""

from engram.evaluation.evaluator import (
    EngramClient,
    EngramEvaluator,
    EvaluationResult,
    EvaluationSummary,
    JudgeLLM,
)
from engram.evaluation.ragas_eval import (
    RAGASEvaluator,
    RAGASResult,
    evaluate_response,
)

__all__ = [
    # v4 RAGAS evaluation
    "RAGASEvaluator",
    "RAGASResult",
    "evaluate_response",
    # v4.2 Test set evaluation
    "EngramEvaluator",
    "EvaluationResult",
    "EvaluationSummary",
    "EngramClient",
    "JudgeLLM",
]
