"""Citation system for inline source references.

Adds inline citations [N] to responses and verifies them using NLI.
Citations link claims to source documents for transparency.
"""

import logging
import re
from dataclasses import dataclass, field

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.reasoning.hallucination_detector import (
    ClaimStatus,
    ClaimVerification,
    HallucinationDetector,
)
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single citation linking claim to source."""

    number: int  # Citation number [1], [2], etc.
    source_id: str
    source_title: str | None
    source_url: str | None
    claim_text: str
    verification: ClaimVerification | None = None

    @property
    def is_verified(self) -> bool:
        """Check if citation is verified as supported."""
        if self.verification is None:
            return False
        return self.verification.status == ClaimStatus.SUPPORTED

    @property
    def formatted_reference(self) -> str:
        """Format as reference entry."""
        if self.source_url:
            return f"[{self.number}] {self.source_title or 'Источник'}: {self.source_url}"
        return f"[{self.number}] {self.source_title or self.source_id}"


@dataclass
class CitedResponse:
    """Response with inline citations."""

    text: str  # Response text with [N] citations
    citations: list[Citation]
    references: str  # Formatted reference list

    # Verification stats
    verified_count: int = 0
    unverified_count: int = 0
    failed_count: int = 0

    @property
    def all_verified(self) -> bool:
        """Check if all citations are verified."""
        return self.failed_count == 0 and self.unverified_count == 0

    @property
    def verification_rate(self) -> float:
        """Ratio of verified citations."""
        total = len(self.citations)
        if total == 0:
            return 1.0
        return self.verified_count / total


# Pattern to match existing citations like [1], [2,3], [1-3]
CITATION_PATTERN = re.compile(r'\[(\d+(?:[-,]\d+)*)\]')


CITATION_GENERATION_PROMPT = """Добавь ссылки на источники в ответ.

Вопрос: {query}

Источники:
{sources}

Текущий ответ:
{response}

Перепиши ответ, добавив ссылки [N] после каждого факта, где N — номер источника.
Если факт основан на нескольких источниках, используй [N,M].
Не меняй содержание ответа, только добавь ссылки.

Ответ с цитатами:"""


class CitationManager:
    """
    Manages inline citations in responses.

    Adds [N] references to claims and optionally verifies them with NLI.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        verify_citations: bool | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.verify_citations = (
            verify_citations
            if verify_citations is not None
            else settings.citations_verify_nli
        )
        self.hallucination_detector = HallucinationDetector(llm_client=self.llm)

    async def add_citations(
        self,
        response: str,
        query: str,
        memories: list[ScoredMemory],
    ) -> CitedResponse:
        """
        Add inline citations to response.

        Args:
            response: Original response text
            query: User query
            memories: Retrieved memories (sources)

        Returns:
            CitedResponse with citations and references
        """
        if not memories:
            return CitedResponse(
                text=response,
                citations=[],
                references="",
            )

        # Build source mapping
        sources = self._build_sources(memories)

        # Generate cited response using LLM
        cited_text = await self._generate_cited_response(
            response=response,
            query=query,
            sources=sources,
        )

        # Extract citations from text
        citations = self._extract_citations(cited_text, memories)

        # Verify citations if enabled
        if self.verify_citations and citations:
            citations = await self._verify_citations(citations, memories)

        # Build references section
        references = self._build_references(citations)

        # Calculate stats
        verified = sum(1 for c in citations if c.is_verified)
        unverified = sum(1 for c in citations if c.verification is None)
        failed = sum(1 for c in citations if c.verification and not c.is_verified)

        logger.info(
            f"Citations: {len(citations)} total, "
            f"{verified} verified, {failed} failed"
        )

        return CitedResponse(
            text=cited_text,
            citations=citations,
            references=references,
            verified_count=verified,
            unverified_count=unverified,
            failed_count=failed,
        )

    def _build_sources(self, memories: list[ScoredMemory]) -> str:
        """Build numbered source list for LLM."""
        lines = []
        for i, sm in enumerate(memories, 1):
            title = None
            if sm.memory.metadata:
                title = sm.memory.metadata.get("title")

            source_name = title or sm.memory.id
            content_preview = sm.memory.content[:200]

            lines.append(f"[{i}] {source_name}:\n{content_preview}...")

        return "\n\n".join(lines)

    async def _generate_cited_response(
        self,
        response: str,
        query: str,
        sources: str,
    ) -> str:
        """Generate response with inline citations."""
        prompt = CITATION_GENERATION_PROMPT.format(
            query=query,
            sources=sources,
            response=response,
        )

        try:
            cited = await self.llm.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=2048,
            )
            return cited.strip()
        except Exception as e:
            logger.warning(f"Citation generation failed: {e}")
            return response

    def _extract_citations(
        self,
        text: str,
        memories: list[ScoredMemory],
    ) -> list[Citation]:
        """Extract citations from text."""
        citations: list[Citation] = []
        seen_numbers: set[int] = set()

        # Find all citation patterns
        for match in CITATION_PATTERN.finditer(text):
            citation_str = match.group(1)

            # Parse numbers (handles [1], [1,2], [1-3])
            numbers: list[int] = []
            for part in citation_str.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    numbers.extend(range(int(start), int(end) + 1))
                else:
                    numbers.append(int(part))

            # Create citation for each number
            for num in numbers:
                if num in seen_numbers:
                    continue
                seen_numbers.add(num)

                # Map to memory (1-indexed)
                if 1 <= num <= len(memories):
                    sm = memories[num - 1]

                    # Extract claim context (text before citation)
                    start = max(0, match.start() - 100)
                    claim_text = text[start:match.start()].strip()

                    # Get source metadata
                    title = None
                    url = None
                    if sm.memory.metadata:
                        title = sm.memory.metadata.get("title")
                        url = sm.memory.metadata.get("url")

                    citations.append(Citation(
                        number=num,
                        source_id=sm.memory.id,
                        source_title=title,
                        source_url=url,
                        claim_text=claim_text,
                    ))

        return citations

    async def _verify_citations(
        self,
        citations: list[Citation],
        memories: list[ScoredMemory],
    ) -> list[Citation]:
        """Verify each citation using NLI."""
        memory_map = {sm.memory.id: sm for sm in memories}

        for citation in citations:
            if not citation.claim_text:
                continue

            # Get source memory
            sm = memory_map.get(citation.source_id)
            if not sm:
                continue

            # Verify claim against specific source
            result = await self.hallucination_detector.detect(
                response=citation.claim_text,
                memories=[sm],
            )

            if result.claims:
                citation.verification = result.claims[0]

        return citations

    def _build_references(self, citations: list[Citation]) -> str:
        """Build references section."""
        if not citations:
            return ""

        # Deduplicate by source
        seen_sources: set[str] = set()
        unique_refs: list[str] = []

        for citation in sorted(citations, key=lambda c: c.number):
            if citation.source_id in seen_sources:
                continue
            seen_sources.add(citation.source_id)
            unique_refs.append(citation.formatted_reference)

        if not unique_refs:
            return ""

        return "\n\n---\nИсточники:\n" + "\n".join(unique_refs)


async def add_citations(
    response: str,
    query: str,
    memories: list[ScoredMemory],
    llm_client: LLMClient | None = None,
) -> CitedResponse:
    """Convenience function for adding citations."""
    manager = CitationManager(llm_client=llm_client)
    return await manager.add_citations(response, query, memories)
