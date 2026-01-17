"""Memory extraction from documents using LLM."""

import logging
from dataclasses import dataclass

from engram.ingestion.concept_extractor import ConceptExtractor, normalize_concept_name
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.ingestion.prompts import MEMORY_EXTRACTION_PROMPT
from engram.models import Concept, MemoryType, SemanticMemory

logger = logging.getLogger(__name__)


def validate_memory_type(type_str: str) -> MemoryType:
    """Validate and normalize memory type."""
    valid_types: set[MemoryType] = {"fact", "procedure", "relationship"}
    normalized = type_str.strip().lower()
    if normalized in valid_types:
        return normalized  # type: ignore[return-value]
    return "fact"


def validate_importance(value: float | int | str) -> float:
    """Validate importance score to 1-10 range."""
    try:
        score = float(value)
        return max(1.0, min(10.0, score))
    except (ValueError, TypeError):
        return 5.0


@dataclass
class MemoryExtractionResult:
    """Result of memory extraction."""

    memories: list[SemanticMemory]
    concepts: list[Concept]  # Concepts referenced by memories


class MemoryExtractor:
    """Extract semantic memories from documents using LLM."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        concept_extractor: ConceptExtractor | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.concept_extractor = concept_extractor or ConceptExtractor(self.llm)

    async def extract(
        self,
        content: str,
        title: str,
        doc_id: str | None = None,
    ) -> MemoryExtractionResult:
        """
        Extract semantic memories from document content.

        Args:
            content: Document text
            title: Document title for context
            doc_id: Optional document ID for provenance

        Returns:
            MemoryExtractionResult with memories and referenced concepts
        """
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            title=title,
            content=content[:8000],  # Limit content size
        )

        try:
            result = await self.llm.generate_json(prompt, temperature=0.3)
        except ValueError as e:
            logger.warning(f"Failed to extract memories: {e}")
            return MemoryExtractionResult(memories=[], concepts=[])

        if not isinstance(result, dict):
            logger.warning(f"Unexpected result type: {type(result)}")
            return MemoryExtractionResult(memories=[], concepts=[])

        raw_memories = result.get("memories", [])
        return self._parse_memories(raw_memories, doc_id)

    def _parse_memories(
        self,
        raw_memories: list[dict],
        doc_id: str | None,
    ) -> MemoryExtractionResult:
        """Parse raw memory data into SemanticMemory objects."""
        memories: list[SemanticMemory] = []
        all_concepts: dict[str, Concept] = {}  # name -> Concept

        for raw in raw_memories:
            if not isinstance(raw, dict):
                continue

            content = raw.get("content", "").strip()
            if not content or len(content) < 10:
                continue

            # Get or create concepts
            concept_names = raw.get("concepts", [])
            concept_ids: list[str] = []

            for name in concept_names:
                if not isinstance(name, str):
                    continue
                normalized = normalize_concept_name(name)
                if not normalized:
                    continue

                if normalized not in all_concepts:
                    concept = self.concept_extractor.get_or_create_concept(normalized)
                    all_concepts[normalized] = concept

                concept_ids.append(all_concepts[normalized].id)

            memory = SemanticMemory(
                id=generate_id(),
                content=content,
                concept_ids=concept_ids,
                source_doc_ids=[doc_id] if doc_id else [],
                memory_type=validate_memory_type(raw.get("type", "fact")),
                importance=validate_importance(raw.get("importance", 5)),
            )
            memories.append(memory)

        return MemoryExtractionResult(
            memories=memories,
            concepts=list(all_concepts.values()),
        )

    async def extract_with_concepts(
        self,
        content: str,
        title: str,
        doc_id: str | None = None,
    ) -> MemoryExtractionResult:
        """
        Extract both concepts and memories in one pass.

        First extracts concepts to build the concept graph,
        then extracts memories linked to those concepts.
        """
        # First extract concepts
        concept_result = await self.concept_extractor.extract(content)

        # Then extract memories
        memory_result = await self.extract(content, title, doc_id)

        # Merge concepts
        all_concepts = {c.name: c for c in concept_result.concepts}
        for c in memory_result.concepts:
            if c.name not in all_concepts:
                all_concepts[c.name] = c

        return MemoryExtractionResult(
            memories=memory_result.memories,
            concepts=list(all_concepts.values()),
        )


# Convenience function
async def extract_memories(
    content: str,
    title: str,
    doc_id: str | None = None,
) -> MemoryExtractionResult:
    """Extract memories from content using default extractor."""
    extractor = MemoryExtractor()
    return await extractor.extract(content, title, doc_id)
