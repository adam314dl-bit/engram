"""Confidence calibration for knowing when to abstain.

Combines signals from:
1. Retrieval confidence (document relevance, coverage)
2. Validation confidence (Self-RAG support level)
3. Generation confidence (LLM verbalized confidence)

Determines appropriate action based on combined confidence.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.reasoning.hallucination_detector import HallucinationResult
from engram.reasoning.self_rag import SelfRAGResult, SupportLevel
from engram.retrieval.crag import CRAGResult, RetrievalQuality
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""

    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


class ResponseAction(str, Enum):
    """Action to take based on confidence."""

    RESPOND_NORMALLY = "respond_normally"
    RESPOND_WITH_CAVEAT = "respond_with_caveat"
    REQUEST_CLARIFICATION = "request_clarification"
    ABSTAIN = "abstain"


@dataclass
class ConfidenceSignals:
    """Individual confidence signals."""

    retrieval_confidence: float  # Based on CRAG result
    validation_confidence: float  # Based on Self-RAG result
    generation_confidence: float  # Verbalized or hallucination-based

    @property
    def min_confidence(self) -> float:
        """Minimum of all signals."""
        return min(
            self.retrieval_confidence,
            self.validation_confidence,
            self.generation_confidence,
        )

    @property
    def max_confidence(self) -> float:
        """Maximum of all signals."""
        return max(
            self.retrieval_confidence,
            self.validation_confidence,
            self.generation_confidence,
        )


@dataclass
class ConfidenceResult:
    """Result of confidence calibration."""

    level: ConfidenceLevel
    action: ResponseAction
    combined_score: float
    signals: ConfidenceSignals
    reasoning: str
    caveat_text: str | None = None

    @property
    def should_abstain(self) -> bool:
        """Check if should abstain from answering."""
        return self.action == ResponseAction.ABSTAIN

    @property
    def needs_caveat(self) -> bool:
        """Check if response needs a caveat."""
        return self.action == ResponseAction.RESPOND_WITH_CAVEAT


# Caveats in Russian
CAVEAT_LOW_CONFIDENCE = """⚠️ Внимание: Ответ может быть неполным или неточным.
Рекомендую проверить информацию в первоисточниках."""

CAVEAT_PARTIAL_SUPPORT = """⚠️ Часть информации основана на неполных данных.
Для точного ответа может потребоваться дополнительная проверка."""

ABSTENTION_MESSAGE = """К сожалению, я не могу дать достоверный ответ на этот вопрос.

Возможные причины:
- Недостаточно информации в базе знаний
- Противоречивые данные
- Вопрос требует уточнения

Рекомендую:
- Уточнить вопрос
- Обратиться к первоисточникам
- Разбить вопрос на более простые части"""


VERBALIZED_CONFIDENCE_PROMPT = """Оцени уверенность в ответе по шкале 0-10.

Вопрос: {query}

Контекст (доступные факты):
{context}

Ответ:
{response}

Оцени:
1. Насколько полно контекст покрывает вопрос?
2. Есть ли неточности или пробелы в ответе?
3. Можно ли доверять этому ответу?

Верни результат в формате:
CONFIDENCE|оценка|обоснование

Где оценка: число 0-10

Пример:
CONFIDENCE|8|Ответ полностью основан на контексте

Ответ:"""


class ConfidenceCalibrator:
    """
    Calibrates confidence from multiple signals.

    Combines retrieval, validation, and generation confidence
    to determine appropriate response action.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        high_threshold: float | None = None,
        medium_threshold: float | None = None,
        low_threshold: float | None = None,
        abstain_on_very_low: bool | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.high_threshold = (
            high_threshold
            if high_threshold is not None
            else settings.confidence_high_threshold
        )
        self.medium_threshold = (
            medium_threshold
            if medium_threshold is not None
            else settings.confidence_medium_threshold
        )
        self.low_threshold = (
            low_threshold
            if low_threshold is not None
            else settings.confidence_low_threshold
        )
        self.abstain_on_very_low = (
            abstain_on_very_low
            if abstain_on_very_low is not None
            else settings.confidence_abstain_on_very_low
        )

        # Weights for combining signals
        self.retrieval_weight = settings.confidence_retrieval_weight
        self.validation_weight = settings.confidence_validation_weight
        self.generation_weight = settings.confidence_generation_weight

    async def calibrate(
        self,
        query: str,
        response: str,
        memories: list[ScoredMemory],
        crag_result: CRAGResult | None = None,
        self_rag_result: SelfRAGResult | None = None,
        hallucination_result: HallucinationResult | None = None,
    ) -> ConfidenceResult:
        """
        Calibrate confidence from all available signals.

        Args:
            query: User query
            response: Generated response
            memories: Retrieved memories
            crag_result: Optional CRAG evaluation result
            self_rag_result: Optional Self-RAG result
            hallucination_result: Optional hallucination detection result

        Returns:
            ConfidenceResult with level, action, and reasoning
        """
        # Calculate individual signals
        retrieval_conf = self._retrieval_confidence(crag_result, memories)
        validation_conf = self._validation_confidence(self_rag_result)
        generation_conf = await self._generation_confidence(
            query, response, memories, hallucination_result
        )

        signals = ConfidenceSignals(
            retrieval_confidence=retrieval_conf,
            validation_confidence=validation_conf,
            generation_confidence=generation_conf,
        )

        # Calculate combined score
        combined = (
            self.retrieval_weight * retrieval_conf +
            self.validation_weight * validation_conf +
            self.generation_weight * generation_conf
        )

        # Determine level
        if combined >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif combined >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        elif combined >= self.low_threshold:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        # Determine action
        action, caveat = self._determine_action(level, signals)

        # Build reasoning
        reasoning = self._build_reasoning(signals, combined, level)

        logger.info(
            f"Confidence calibration: {level.value} ({combined:.2f}), "
            f"action={action.value}"
        )

        return ConfidenceResult(
            level=level,
            action=action,
            combined_score=combined,
            signals=signals,
            reasoning=reasoning,
            caveat_text=caveat,
        )

    def _retrieval_confidence(
        self,
        crag_result: CRAGResult | None,
        memories: list[ScoredMemory],
    ) -> float:
        """Calculate confidence from retrieval quality."""
        if crag_result is None:
            # Fallback: use average memory score
            if not memories:
                return 0.3
            return sum(m.score for m in memories) / len(memories)

        # Map CRAG quality to confidence
        quality_map = {
            RetrievalQuality.CORRECT: 0.9,
            RetrievalQuality.AMBIGUOUS: 0.6,
            RetrievalQuality.INCORRECT: 0.2,
        }

        base_conf = quality_map.get(crag_result.quality, 0.5)

        # Adjust by relevant ratio
        if crag_result.relevant_ratio > 0:
            base_conf = (base_conf + crag_result.relevant_ratio) / 2

        return base_conf

    def _validation_confidence(
        self,
        self_rag_result: SelfRAGResult | None,
    ) -> float:
        """Calculate confidence from validation result."""
        if self_rag_result is None:
            return 0.7  # Default moderate confidence

        if self_rag_result.abstained:
            return 0.1

        # Map support level to confidence
        support_map = {
            SupportLevel.FULLY_SUPPORTED: 0.95,
            SupportLevel.PARTIALLY_SUPPORTED: 0.65,
            SupportLevel.NOT_SUPPORTED: 0.2,
        }

        validation = self_rag_result.final_validation
        base_conf = support_map.get(validation.support_level, 0.5)

        # Penalize multiple iterations
        iteration_penalty = 0.05 * (self_rag_result.iteration_count - 1)
        base_conf = max(0.1, base_conf - iteration_penalty)

        return base_conf

    async def _generation_confidence(
        self,
        query: str,
        response: str,
        memories: list[ScoredMemory],
        hallucination_result: HallucinationResult | None,
    ) -> float:
        """Calculate confidence from generation quality."""
        # Use hallucination result if available
        if hallucination_result is not None:
            return hallucination_result.faithfulness_score

        # Otherwise, use verbalized confidence from LLM
        try:
            context = "\n".join(
                m.memory.content[:300] for m in memories[:5]
            )

            prompt = VERBALIZED_CONFIDENCE_PROMPT.format(
                query=query,
                context=context,
                response=response,
            )

            llm_response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
            )

            confidence = self._parse_confidence_response(llm_response)
            return confidence / 10.0  # Normalize to 0-1

        except Exception as e:
            logger.warning(f"Verbalized confidence failed: {e}")
            return 0.5

    def _parse_confidence_response(self, text: str) -> float:
        """Parse pipe-delimited confidence response."""
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CONFIDENCE|"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    try:
                        confidence = float(parts[1].strip())
                        return max(0.0, min(10.0, confidence))
                    except ValueError:
                        pass
        return 5.0  # Default middle confidence

    def _determine_action(
        self,
        level: ConfidenceLevel,
        signals: ConfidenceSignals,
    ) -> tuple[ResponseAction, str | None]:
        """Determine action and caveat based on confidence."""
        if level == ConfidenceLevel.HIGH:
            return ResponseAction.RESPOND_NORMALLY, None

        if level == ConfidenceLevel.MEDIUM:
            # Check if any signal is particularly low
            if signals.min_confidence < 0.4:
                return ResponseAction.RESPOND_WITH_CAVEAT, CAVEAT_PARTIAL_SUPPORT
            return ResponseAction.RESPOND_NORMALLY, None

        if level == ConfidenceLevel.LOW:
            return ResponseAction.RESPOND_WITH_CAVEAT, CAVEAT_LOW_CONFIDENCE

        # VERY_LOW
        if self.abstain_on_very_low:
            return ResponseAction.ABSTAIN, ABSTENTION_MESSAGE

        return ResponseAction.RESPOND_WITH_CAVEAT, CAVEAT_LOW_CONFIDENCE

    def _build_reasoning(
        self,
        signals: ConfidenceSignals,
        combined: float,
        level: ConfidenceLevel,
    ) -> str:
        """Build reasoning explanation."""
        parts = [
            f"Retrieval: {signals.retrieval_confidence:.2f}",
            f"Validation: {signals.validation_confidence:.2f}",
            f"Generation: {signals.generation_confidence:.2f}",
            f"Combined: {combined:.2f} ({level.value})",
        ]
        return " | ".join(parts)

    def apply_action(
        self,
        response: str,
        confidence_result: ConfidenceResult,
    ) -> str:
        """Apply action to response (add caveat or abstain)."""
        if confidence_result.action == ResponseAction.ABSTAIN:
            return ABSTENTION_MESSAGE

        if confidence_result.needs_caveat and confidence_result.caveat_text:
            return f"{response}\n\n{confidence_result.caveat_text}"

        return response


async def calibrate_confidence(
    query: str,
    response: str,
    memories: list[ScoredMemory],
    crag_result: CRAGResult | None = None,
    self_rag_result: SelfRAGResult | None = None,
    hallucination_result: HallucinationResult | None = None,
    llm_client: LLMClient | None = None,
) -> ConfidenceResult:
    """Convenience function for confidence calibration."""
    calibrator = ConfidenceCalibrator(llm_client=llm_client)
    return await calibrator.calibrate(
        query=query,
        response=response,
        memories=memories,
        crag_result=crag_result,
        self_rag_result=self_rag_result,
        hallucination_result=hallucination_result,
    )
