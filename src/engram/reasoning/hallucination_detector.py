"""NLI-based hallucination detection for post-generation verification.

Verifies each claim in the response against the retrieved context using:
1. mDeBERTa NLI model (fast, requires GPU memory)
2. LLM fallback (slower, no extra model)

Supports Russian via multilingual mDeBERTa-xnli model.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


class ClaimStatus(str, Enum):
    """Status of a claim verification."""

    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""

    claim: str
    status: ClaimStatus
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    best_evidence: str | None = None
    evidence_source_id: str | None = None

    @property
    def is_supported(self) -> bool:
        """Check if claim is supported."""
        return self.status == ClaimStatus.SUPPORTED

    @property
    def is_problematic(self) -> bool:
        """Check if claim is contradicted or not supported."""
        return self.status in (ClaimStatus.NOT_SUPPORTED, ClaimStatus.CONTRADICTED)


@dataclass
class HallucinationResult:
    """Complete hallucination detection result."""

    claims: list[ClaimVerification]
    faithfulness_score: float  # 0-1, higher is better
    supported_count: int
    unsupported_count: int
    contradicted_count: int
    uncertain_count: int

    @property
    def has_hallucinations(self) -> bool:
        """Check if any hallucinations detected."""
        return self.contradicted_count > 0 or self.unsupported_count > 0

    @property
    def is_faithful(self) -> bool:
        """Check if response is faithful to context."""
        return self.faithfulness_score >= 0.7


# Global NLI model cache
_nli_model: Any = None
_nli_tokenizer: Any = None


def _get_nli_model() -> tuple[Any, Any]:
    """Lazy load NLI model and tokenizer."""
    global _nli_model, _nli_tokenizer

    if _nli_model is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = settings.nli_model
            logger.info(f"Loading NLI model: {model_name}")

            _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                _nli_model = _nli_model.to("cuda")

            _nli_model.eval()
            logger.info("NLI model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            raise

    return _nli_model, _nli_tokenizer


def _extract_claims(response: str) -> list[str]:
    """Extract individual claims from response."""
    # Split by sentences
    sentences = re.split(r'[.!?]\s+', response)

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Filter out very short or non-informative sentences
        if len(sentence) > 20 and not sentence.startswith(("К сожалению", "Извините")):
            claims.append(sentence)

    return claims


CLAIM_VERIFICATION_PROMPT = """Проверь, подтверждается ли утверждение контекстом.

Контекст:
{context}

Утверждение: {claim}

Классифицируй:
- supported: утверждение подтверждается контекстом
- contradicted: утверждение противоречит контексту
- not_supported: утверждение не упоминается в контексте
- uncertain: невозможно определить

Верни результат в формате:
VERIFY|статус|уверенность|доказательство

Где:
- статус: supported, contradicted, not_supported, uncertain
- уверенность: число 0-100
- доказательство: цитата из контекста или "нет"

Пример:
VERIFY|supported|85|Docker использует контейнеры для изоляции

Ответ:"""


class HallucinationDetector:
    """
    NLI-based hallucination detector.

    Supports two backends:
    1. mDeBERTa NLI model (fast, ~50ms per claim)
    2. LLM fallback (slower, ~200ms per claim)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_model: bool | None = None,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.use_model = (
            use_model
            if use_model is not None
            else settings.nli_use_model
        )
        self.entailment_threshold = (
            entailment_threshold
            if entailment_threshold is not None
            else settings.nli_entailment_threshold
        )
        self.contradiction_threshold = (
            contradiction_threshold
            if contradiction_threshold is not None
            else settings.nli_contradiction_threshold
        )

    async def detect(
        self,
        response: str,
        memories: list[ScoredMemory],
    ) -> HallucinationResult:
        """
        Detect hallucinations in response.

        Args:
            response: Generated response text
            memories: Retrieved memories (context)

        Returns:
            HallucinationResult with claim verifications
        """
        # Extract claims from response
        claims = _extract_claims(response)

        if not claims:
            return HallucinationResult(
                claims=[],
                faithfulness_score=1.0,
                supported_count=0,
                unsupported_count=0,
                contradicted_count=0,
                uncertain_count=0,
            )

        # Format context
        context = self._format_context(memories)

        # Verify each claim
        verifications: list[ClaimVerification] = []

        for claim in claims:
            if self.use_model:
                verification = await self._verify_with_model(claim, memories)
            else:
                verification = await self._verify_with_llm(claim, context)

            verifications.append(verification)

        # Calculate statistics
        supported = sum(1 for v in verifications if v.status == ClaimStatus.SUPPORTED)
        unsupported = sum(1 for v in verifications if v.status == ClaimStatus.NOT_SUPPORTED)
        contradicted = sum(1 for v in verifications if v.status == ClaimStatus.CONTRADICTED)
        uncertain = sum(1 for v in verifications if v.status == ClaimStatus.UNCERTAIN)

        # Calculate faithfulness score
        total = len(verifications)
        if total > 0:
            # Supported and uncertain are OK, contradicted and unsupported are bad
            faithfulness = (supported + uncertain * 0.5) / total
        else:
            faithfulness = 1.0

        logger.info(
            f"Hallucination detection: {supported}/{total} supported, "
            f"{contradicted} contradicted, faithfulness={faithfulness:.2f}"
        )

        return HallucinationResult(
            claims=verifications,
            faithfulness_score=faithfulness,
            supported_count=supported,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            uncertain_count=uncertain,
        )

    async def _verify_with_model(
        self,
        claim: str,
        memories: list[ScoredMemory],
    ) -> ClaimVerification:
        """Verify claim using NLI model."""
        try:
            import torch

            model, tokenizer = _get_nli_model()

            best_entailment = 0.0
            best_contradiction = 0.0
            best_neutral = 1.0
            best_evidence = None
            best_source_id = None

            # Check against each memory
            for sm in memories:
                premise = sm.memory.content[:512]  # Limit length
                hypothesis = claim

                inputs = tokenizer(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]

                    # mDeBERTa labels: 0=entailment, 1=neutral, 2=contradiction
                    entailment = probs[0].item()
                    neutral = probs[1].item()
                    contradiction = probs[2].item()

                    if entailment > best_entailment:
                        best_entailment = entailment
                        best_neutral = neutral
                        best_contradiction = contradiction
                        best_evidence = premise[:200]
                        best_source_id = sm.memory.id

            # Determine status
            if best_entailment >= self.entailment_threshold:
                status = ClaimStatus.SUPPORTED
            elif best_contradiction >= self.contradiction_threshold:
                status = ClaimStatus.CONTRADICTED
            elif best_entailment > 0.3:
                status = ClaimStatus.UNCERTAIN
            else:
                status = ClaimStatus.NOT_SUPPORTED

            return ClaimVerification(
                claim=claim,
                status=status,
                entailment_prob=best_entailment,
                contradiction_prob=best_contradiction,
                neutral_prob=best_neutral,
                best_evidence=best_evidence,
                evidence_source_id=best_source_id,
            )

        except Exception as e:
            logger.warning(f"NLI model verification failed: {e}")
            # Fall back to LLM
            return await self._verify_with_llm(claim, self._format_context(memories))

    async def _verify_with_llm(
        self,
        claim: str,
        context: str,
    ) -> ClaimVerification:
        """Verify claim using LLM fallback."""
        try:
            prompt = CLAIM_VERIFICATION_PROMPT.format(
                context=context,
                claim=claim,
            )

            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
            )

            status, confidence, evidence = self._parse_verification_response(response)

            # Convert confidence to probabilities
            if status == ClaimStatus.SUPPORTED:
                entailment = confidence
                contradiction = 0.0
                neutral = 1 - confidence
            elif status == ClaimStatus.CONTRADICTED:
                entailment = 0.0
                contradiction = confidence
                neutral = 1 - confidence
            else:
                entailment = 0.3
                contradiction = 0.1
                neutral = 0.6

            return ClaimVerification(
                claim=claim,
                status=status,
                entailment_prob=entailment,
                contradiction_prob=contradiction,
                neutral_prob=neutral,
                best_evidence=evidence,
            )

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")

        # Default to uncertain on error
        return ClaimVerification(
            claim=claim,
            status=ClaimStatus.UNCERTAIN,
            entailment_prob=0.3,
            contradiction_prob=0.1,
            neutral_prob=0.6,
        )

    def _parse_verification_response(
        self,
        text: str,
    ) -> tuple[ClaimStatus, float, str | None]:
        """Parse pipe-delimited verification response."""
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("VERIFY|"):
                parts = line.split("|", 3)
                if len(parts) >= 3:
                    status_str = parts[1].strip().lower()
                    # Validate status
                    if status_str not in ("supported", "contradicted", "not_supported", "uncertain"):
                        status_str = "uncertain"
                    status = ClaimStatus(status_str)

                    try:
                        confidence = float(parts[2].strip()) / 100.0
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        confidence = 0.5

                    evidence = None
                    if len(parts) > 3:
                        evidence = parts[3].strip()
                        if evidence.lower() in ("нет", "null", "none", ""):
                            evidence = None

                    return status, confidence, evidence

        return ClaimStatus.UNCERTAIN, 0.5, None

    def _format_context(self, memories: list[ScoredMemory]) -> str:
        """Format memories as context string."""
        if not memories:
            return "Нет доступного контекста."

        lines = []
        for sm in memories[:10]:  # Limit to top 10
            lines.append(sm.memory.content[:500])

        return "\n\n".join(lines)


async def detect_hallucinations(
    response: str,
    memories: list[ScoredMemory],
    llm_client: LLMClient | None = None,
) -> HallucinationResult:
    """Convenience function for hallucination detection."""
    detector = HallucinationDetector(llm_client=llm_client)
    return await detector.detect(response, memories)
