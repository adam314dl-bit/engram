"""Self-RAG validation loop for response verification.

Validates that generated response is supported by the retrieved context.
If not supported, regenerates with guidance about unsupported claims.
After max iterations, abstains from answering.

Based on: "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


class SupportLevel(str, Enum):
    """Level of support for response by context."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


@dataclass
class UnsupportedClaim:
    """A claim in the response not supported by context."""

    claim: str
    reason: str


@dataclass
class ValidationResult:
    """Result of validating a single response."""

    support_level: SupportLevel
    unsupported_claims: list[UnsupportedClaim]
    supported_claims: list[str]
    validation_reasoning: str
    confidence: float

    @property
    def is_acceptable(self) -> bool:
        """Check if response is acceptable (at least partially supported)."""
        return self.support_level in (
            SupportLevel.FULLY_SUPPORTED,
            SupportLevel.PARTIALLY_SUPPORTED,
        )


@dataclass
class IterationResult:
    """Result of a single validation iteration."""

    iteration: int
    response: str
    validation: ValidationResult
    regeneration_guidance: str | None = None


@dataclass
class SelfRAGResult:
    """Complete result of Self-RAG validation loop."""

    final_response: str
    final_validation: ValidationResult
    iterations: list[IterationResult]
    abstained: bool = False
    abstention_reason: str | None = None

    @property
    def iteration_count(self) -> int:
        """Number of iterations performed."""
        return len(self.iterations)

    @property
    def was_regenerated(self) -> bool:
        """Check if response was regenerated."""
        return len(self.iterations) > 1


VALIDATION_PROMPT = """Проверь, подтверждается ли ответ контекстом.

Вопрос: {query}

Контекст (извлечённые факты):
{context}

Ответ для проверки:
{response}

Проанализируй каждое утверждение в ответе:
1. Подтверждается ли оно контекстом?
2. Есть ли противоречия с контекстом?
3. Есть ли утверждения без опоры на контекст?

Верни JSON:
{{
  "support_level": "fully_supported" | "partially_supported" | "not_supported",
  "supported_claims": ["список подтверждённых утверждений"],
  "unsupported_claims": [
    {{"claim": "неподтверждённое утверждение", "reason": "почему не подтверждено"}}
  ],
  "reasoning": "общий анализ"
}}

JSON:"""


REGENERATION_PROMPT = """Исправь ответ, убрав неподтверждённые утверждения.

Вопрос: {query}

Контекст (достоверные факты):
{context}

Предыдущий ответ имел проблемы:
{issues}

Инструкции:
1. Используй ТОЛЬКО информацию из контекста
2. Не выдумывай факты
3. Если информации недостаточно — скажи об этом
4. Убери или замени неподтверждённые утверждения

Исправленный ответ:"""


ABSTENTION_RESPONSE_RU = """К сожалению, я не могу дать достоверный ответ на этот вопрос.
Доступная информация недостаточна или противоречива.
Рекомендую обратиться к первоисточникам или уточнить вопрос."""


class SelfRAGValidator:
    """
    Self-RAG validator for response verification.

    Validates response against context and regenerates if not supported.
    Maximum iterations configurable (default 3).
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else settings.self_rag_max_iterations
        )

    async def validate_and_refine(
        self,
        query: str,
        initial_response: str,
        memories: list[ScoredMemory],
    ) -> SelfRAGResult:
        """
        Validate response and refine if needed.

        Args:
            query: User query
            initial_response: Initial generated response
            memories: Retrieved memories (context)

        Returns:
            SelfRAGResult with final response and validation history
        """
        context = self._format_context(memories)
        iterations: list[IterationResult] = []
        current_response = initial_response

        for i in range(self.max_iterations):
            # Validate current response
            validation = await self._validate_response(
                query=query,
                response=current_response,
                context=context,
            )

            iterations.append(IterationResult(
                iteration=i + 1,
                response=current_response,
                validation=validation,
            ))

            logger.info(
                f"Self-RAG iteration {i + 1}: {validation.support_level.value}, "
                f"{len(validation.unsupported_claims)} unsupported claims"
            )

            # If acceptable, return
            if validation.is_acceptable:
                return SelfRAGResult(
                    final_response=current_response,
                    final_validation=validation,
                    iterations=iterations,
                )

            # If not last iteration, try to regenerate
            if i < self.max_iterations - 1:
                guidance = self._build_regeneration_guidance(validation)
                iterations[-1].regeneration_guidance = guidance

                current_response = await self._regenerate_response(
                    query=query,
                    context=context,
                    issues=guidance,
                )

        # Max iterations reached, abstain
        logger.warning(
            f"Self-RAG: abstaining after {self.max_iterations} iterations"
        )
        return SelfRAGResult(
            final_response=ABSTENTION_RESPONSE_RU,
            final_validation=iterations[-1].validation,
            iterations=iterations,
            abstained=True,
            abstention_reason=f"Could not generate supported response after {self.max_iterations} iterations",
        )

    async def _validate_response(
        self,
        query: str,
        response: str,
        context: str,
    ) -> ValidationResult:
        """Validate a single response against context."""
        try:
            prompt = VALIDATION_PROMPT.format(
                query=query,
                context=context,
                response=response,
            )

            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024,
                fallback=None,
            )

            if result:
                support_level = SupportLevel(result.get("support_level", "not_supported"))

                unsupported_claims = [
                    UnsupportedClaim(claim=c.get("claim", ""), reason=c.get("reason", ""))
                    for c in result.get("unsupported_claims", [])
                ]

                supported_claims = result.get("supported_claims", [])
                reasoning = result.get("reasoning", "")

                # Calculate confidence based on support level
                confidence_map = {
                    SupportLevel.FULLY_SUPPORTED: 0.9,
                    SupportLevel.PARTIALLY_SUPPORTED: 0.6,
                    SupportLevel.NOT_SUPPORTED: 0.2,
                }
                confidence = confidence_map[support_level]

                return ValidationResult(
                    support_level=support_level,
                    unsupported_claims=unsupported_claims,
                    supported_claims=supported_claims,
                    validation_reasoning=reasoning,
                    confidence=confidence,
                )

        except Exception as e:
            logger.warning(f"Validation failed: {e}")

        # Default to partially supported on error
        return ValidationResult(
            support_level=SupportLevel.PARTIALLY_SUPPORTED,
            unsupported_claims=[],
            supported_claims=[],
            validation_reasoning=f"Validation error: {e}",
            confidence=0.5,
        )

    async def _regenerate_response(
        self,
        query: str,
        context: str,
        issues: str,
    ) -> str:
        """Regenerate response with guidance."""
        prompt = REGENERATION_PROMPT.format(
            query=query,
            context=context,
            issues=issues,
        )

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2048,
        )

        return response.strip()

    def _format_context(self, memories: list[ScoredMemory]) -> str:
        """Format memories as context string."""
        if not memories:
            return "Нет доступного контекста."

        lines = []
        for i, sm in enumerate(memories, 1):
            content = sm.memory.content[:500]  # Limit for token efficiency
            lines.append(f"{i}. {content}")

        return "\n".join(lines)

    def _build_regeneration_guidance(self, validation: ValidationResult) -> str:
        """Build guidance for regeneration based on validation."""
        issues = []

        for claim in validation.unsupported_claims:
            issues.append(f"- {claim.claim}: {claim.reason}")

        if not issues:
            issues.append("- Ответ не опирается на предоставленный контекст")

        return "\n".join(issues)


async def validate_response(
    query: str,
    response: str,
    memories: list[ScoredMemory],
    llm_client: LLMClient | None = None,
) -> SelfRAGResult:
    """Convenience function for Self-RAG validation."""
    validator = SelfRAGValidator(llm_client=llm_client)
    return await validator.validate_and_refine(query, response, memories)
