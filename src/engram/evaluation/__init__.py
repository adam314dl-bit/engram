"""Evaluation module for Engram quality metrics."""

from engram.evaluation.evaluator import (
    EngramClient,
    EngramEvaluator,
    EvaluationResult,
    EvaluationSummary,
    JudgeLLM,
)

__all__ = [
    # v4.2 Test set evaluation
    "EngramEvaluator",
    "EvaluationResult",
    "EvaluationSummary",
    "EngramClient",
    "JudgeLLM",
]
