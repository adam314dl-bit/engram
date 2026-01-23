"""RAGAS-style evaluation metrics for RAG quality assessment.

Implements key metrics:
1. Faithfulness: Is the response supported by context?
2. Answer Relevancy: Is the response pertinent to the question?
3. Context Precision: Are relevant chunks ranked higher?
4. Context Recall: Are all needed facts retrieved? (requires ground truth)

Based on: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)

Scoring:
- Poor: < 0.5
- Acceptable: 0.5 - 0.7
- Good: 0.7 - 0.85
- Excellent: > 0.85
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality level based on score."""

    EXCELLENT = "excellent"  # > 0.85
    GOOD = "good"  # 0.7 - 0.85
    ACCEPTABLE = "acceptable"  # 0.5 - 0.7
    POOR = "poor"  # < 0.5


@dataclass
class MetricResult:
    """Result for a single metric."""

    name: str
    score: float
    reasoning: str
    details: dict = field(default_factory=dict)

    @property
    def level(self) -> QualityLevel:
        """Get quality level for score."""
        if self.score > 0.85:
            return QualityLevel.EXCELLENT
        if self.score > 0.7:
            return QualityLevel.GOOD
        if self.score > 0.5:
            return QualityLevel.ACCEPTABLE
        return QualityLevel.POOR


@dataclass
class RAGASResult:
    """Complete RAGAS evaluation result."""

    faithfulness: MetricResult
    answer_relevancy: MetricResult
    context_precision: MetricResult
    context_recall: MetricResult | None  # Requires ground truth

    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        scores = [
            self.faithfulness.score,
            self.answer_relevancy.score,
            self.context_precision.score,
        ]
        if self.context_recall:
            scores.append(self.context_recall.score)

        return sum(scores) / len(scores)

    @property
    def overall_level(self) -> QualityLevel:
        """Overall quality level."""
        score = self.overall_score
        if score > 0.85:
            return QualityLevel.EXCELLENT
        if score > 0.7:
            return QualityLevel.GOOD
        if score > 0.5:
            return QualityLevel.ACCEPTABLE
        return QualityLevel.POOR

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "faithfulness": self.faithfulness.score,
            "answer_relevancy": self.answer_relevancy.score,
            "context_precision": self.context_precision.score,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
        }
        if self.context_recall:
            result["context_recall"] = self.context_recall.score
        return result


FAITHFULNESS_PROMPT = """Оцени, насколько ответ подтверждается контекстом.

Вопрос: {query}

Контекст:
{context}

Ответ:
{response}

Проанализируй каждое утверждение в ответе:
1. Сколько утверждений полностью подтверждается контекстом?
2. Сколько утверждений противоречит контексту?
3. Сколько утверждений не имеет опоры в контексте?

Верни JSON:
{{
  "supported_claims": 0,
  "contradicted_claims": 0,
  "unsupported_claims": 0,
  "total_claims": 0,
  "reasoning": "анализ"
}}

JSON:"""


RELEVANCY_PROMPT = """Оцени релевантность ответа вопросу.

Вопрос: {query}

Ответ:
{response}

Оцени:
1. Отвечает ли ответ на заданный вопрос?
2. Насколько полно раскрыт вопрос?
3. Есть ли лишняя информация не по теме?

Верни JSON:
{{
  "addresses_question": true | false,
  "completeness": 0.0-1.0,
  "off_topic_ratio": 0.0-1.0,
  "reasoning": "анализ"
}}

JSON:"""


PRECISION_PROMPT = """Оцени качество ранжирования контекста.

Вопрос: {query}

Контекст (в порядке ранжирования):
{ranked_context}

Для каждого документа укажи, релевантен ли он вопросу (да/нет).

Верни JSON:
{{
  "relevance": [true, false, true, ...],
  "reasoning": "анализ"
}}

JSON:"""


RECALL_PROMPT = """Оцени полноту контекста относительно эталонного ответа.

Вопрос: {query}

Эталонный ответ (ground truth):
{ground_truth}

Контекст:
{context}

Какая часть информации из эталонного ответа присутствует в контексте?

Верни JSON:
{{
  "facts_in_ground_truth": 0,
  "facts_found_in_context": 0,
  "reasoning": "анализ"
}}

JSON:"""


class RAGASEvaluator:
    """
    RAGAS-style evaluator for RAG quality.

    Computes faithfulness, relevancy, precision, and optionally recall.
    Can run synchronously or asynchronously.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        async_evaluation: bool | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.async_evaluation = (
            async_evaluation
            if async_evaluation is not None
            else settings.ragas_async_evaluation
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        memories: list[ScoredMemory],
        ground_truth: str | None = None,
    ) -> RAGASResult:
        """
        Evaluate response quality.

        Args:
            query: User query
            response: Generated response
            memories: Retrieved memories (context)
            ground_truth: Optional ground truth answer for recall

        Returns:
            RAGASResult with all metrics
        """
        context = self._format_context(memories)

        if self.async_evaluation:
            # Run metrics in parallel
            tasks = [
                self._evaluate_faithfulness(query, response, context),
                self._evaluate_relevancy(query, response),
                self._evaluate_precision(query, memories),
            ]

            if ground_truth:
                tasks.append(self._evaluate_recall(query, context, ground_truth))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            faithfulness = results[0] if not isinstance(results[0], Exception) else self._error_metric("faithfulness", results[0])
            relevancy = results[1] if not isinstance(results[1], Exception) else self._error_metric("answer_relevancy", results[1])
            precision = results[2] if not isinstance(results[2], Exception) else self._error_metric("context_precision", results[2])
            recall = None
            if ground_truth and len(results) > 3:
                recall = results[3] if not isinstance(results[3], Exception) else self._error_metric("context_recall", results[3])

        else:
            # Run sequentially
            faithfulness = await self._evaluate_faithfulness(query, response, context)
            relevancy = await self._evaluate_relevancy(query, response)
            precision = await self._evaluate_precision(query, memories)
            recall = None
            if ground_truth:
                recall = await self._evaluate_recall(query, context, ground_truth)

        result = RAGASResult(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )

        logger.info(
            f"RAGAS evaluation: overall={result.overall_score:.2f} "
            f"({result.overall_level.value})"
        )

        return result

    async def _evaluate_faithfulness(
        self,
        query: str,
        response: str,
        context: str,
    ) -> MetricResult:
        """Evaluate faithfulness of response to context."""
        try:
            prompt = FAITHFULNESS_PROMPT.format(
                query=query,
                response=response,
                context=context,
            )

            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
                fallback=None,
            )

            if result:
                supported = result.get("supported_claims", 0)
                contradicted = result.get("contradicted_claims", 0)
                unsupported = result.get("unsupported_claims", 0)
                total = result.get("total_claims", 1)
                reasoning = result.get("reasoning", "")

                if total > 0:
                    # Faithfulness = supported / total
                    score = supported / total
                else:
                    score = 1.0

                return MetricResult(
                    name="faithfulness",
                    score=score,
                    reasoning=reasoning,
                    details={
                        "supported": supported,
                        "contradicted": contradicted,
                        "unsupported": unsupported,
                        "total": total,
                    },
                )

        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")

        return self._error_metric("faithfulness", e)

    async def _evaluate_relevancy(
        self,
        query: str,
        response: str,
    ) -> MetricResult:
        """Evaluate relevancy of response to question."""
        try:
            prompt = RELEVANCY_PROMPT.format(
                query=query,
                response=response,
            )

            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
                fallback=None,
            )

            if result:
                addresses = result.get("addresses_question", True)
                completeness = result.get("completeness", 0.5)
                off_topic = result.get("off_topic_ratio", 0.0)
                reasoning = result.get("reasoning", "")

                # Relevancy score
                score = completeness * (1 - off_topic)
                if not addresses:
                    score *= 0.5

                return MetricResult(
                    name="answer_relevancy",
                    score=score,
                    reasoning=reasoning,
                    details={
                        "addresses_question": addresses,
                        "completeness": completeness,
                        "off_topic_ratio": off_topic,
                    },
                )

        except Exception as e:
            logger.warning(f"Relevancy evaluation failed: {e}")

        return self._error_metric("answer_relevancy", e)

    async def _evaluate_precision(
        self,
        query: str,
        memories: list[ScoredMemory],
    ) -> MetricResult:
        """Evaluate precision of context ranking."""
        if not memories:
            return MetricResult(
                name="context_precision",
                score=0.0,
                reasoning="No context retrieved",
            )

        try:
            # Format ranked context
            ranked_context = "\n\n".join(
                f"[{i+1}] {m.memory.content[:300]}..."
                for i, m in enumerate(memories[:10])
            )

            prompt = PRECISION_PROMPT.format(
                query=query,
                ranked_context=ranked_context,
            )

            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
                fallback=None,
            )

            if result:
                relevance = result.get("relevance", [])
                reasoning = result.get("reasoning", "")

                if relevance:
                    # Precision @ k with rank weighting
                    # Higher weight for top ranks
                    total_weight = 0
                    weighted_relevant = 0

                    for i, is_relevant in enumerate(relevance):
                        weight = 1.0 / (i + 1)  # Rank weighting
                        total_weight += weight
                        if is_relevant:
                            weighted_relevant += weight

                    score = weighted_relevant / total_weight if total_weight > 0 else 0

                    return MetricResult(
                        name="context_precision",
                        score=score,
                        reasoning=reasoning,
                        details={
                            "relevance": relevance,
                            "relevant_count": sum(1 for r in relevance if r),
                            "total_count": len(relevance),
                        },
                    )

        except Exception as e:
            logger.warning(f"Precision evaluation failed: {e}")

        return self._error_metric("context_precision", e)

    async def _evaluate_recall(
        self,
        query: str,
        context: str,
        ground_truth: str,
    ) -> MetricResult:
        """Evaluate recall of context against ground truth."""
        try:
            prompt = RECALL_PROMPT.format(
                query=query,
                ground_truth=ground_truth,
                context=context,
            )

            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
                fallback=None,
            )

            if result:
                total_facts = result.get("facts_in_ground_truth", 1)
                found_facts = result.get("facts_found_in_context", 0)
                reasoning = result.get("reasoning", "")

                score = found_facts / total_facts if total_facts > 0 else 0

                return MetricResult(
                    name="context_recall",
                    score=score,
                    reasoning=reasoning,
                    details={
                        "total_facts": total_facts,
                        "found_facts": found_facts,
                    },
                )

        except Exception as e:
            logger.warning(f"Recall evaluation failed: {e}")

        return self._error_metric("context_recall", e)

    def _format_context(self, memories: list[ScoredMemory]) -> str:
        """Format memories as context string."""
        if not memories:
            return "Нет контекста."

        lines = []
        for i, m in enumerate(memories[:10], 1):
            lines.append(f"[{i}] {m.memory.content[:500]}")

        return "\n\n".join(lines)

    def _error_metric(self, name: str, error: Exception) -> MetricResult:
        """Create error metric result."""
        return MetricResult(
            name=name,
            score=0.5,  # Neutral on error
            reasoning=f"Evaluation error: {error}",
        )


async def evaluate_response(
    query: str,
    response: str,
    memories: list[ScoredMemory],
    ground_truth: str | None = None,
    llm_client: LLMClient | None = None,
) -> RAGASResult:
    """Convenience function for RAGAS evaluation."""
    evaluator = RAGASEvaluator(llm_client=llm_client)
    return await evaluator.evaluate(query, response, memories, ground_truth)
