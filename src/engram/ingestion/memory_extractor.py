"""Memory extraction from documents using LLM."""

import logging
from dataclasses import dataclass

from engram.ingestion.concept_extractor import ConceptExtractor, normalize_concept_name
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.ingestion.prompts import MEMORY_EXTRACTION_PROMPT, EXTRACTION_SYSTEM_PROMPT
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
    persons: list[tuple[str, str | None, str | None]] = None  # (name, role, team)

    def __post_init__(self) -> None:
        if self.persons is None:
            self.persons = []


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
            result = await self.llm.generate(
                prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning(f"Failed to extract memories: {e}")
            return MemoryExtractionResult(memories=[], concepts=[])

        return self._parse_memory_lines(result, doc_id)

    def _parse_memory_lines(
        self,
        text: str,
        doc_id: str | None,
    ) -> MemoryExtractionResult:
        """Parse MEMORY|... and PERSON|... lines.

        Supports three formats:
        - v4.5 (7 parts): MEMORY|search_summary|keywords|actual_content|type|concepts|importance
        - v3 (6 parts): MEMORY|summary|keywords|type|concepts|importance
        - v2 (5 parts): MEMORY|content|type|concepts|importance
        """
        memories: list[SemanticMemory] = []
        all_concepts: dict[str, Concept] = {}  # name -> Concept
        persons: list[tuple[str, str | None, str | None]] = []

        for line in text.split("\n"):
            line = line.strip()

            if line.startswith("MEMORY|"):
                parts = line.split("|")
                if len(parts) < 5:
                    continue

                content: str = ""
                search_content: str | None = None
                memory_type: MemoryType = "fact"
                concepts_str: str = ""
                importance: float = 5.0

                # Detect format by checking where the type field is
                # v4.5 (7 parts): type is at parts[4]
                # v3 (6 parts): type is at parts[3]
                # v2 (5 parts): type is at parts[2]

                is_v45_format = (
                    len(parts) >= 7 and
                    parts[4].strip().lower() in {"fact", "procedure", "relationship"}
                )
                is_v3_format = (
                    not is_v45_format and
                    len(parts) >= 6 and
                    parts[3].strip().lower() in {"fact", "procedure", "relationship"}
                )

                if is_v45_format:
                    # v4.5 format: search_summary|keywords|actual_content|type|concepts|importance
                    search_summary = parts[1].strip()
                    keywords = parts[2].strip()
                    actual_content = parts[3].strip()
                    memory_type = validate_memory_type(parts[4])
                    concepts_str = parts[5] if len(parts) > 5 else ""
                    importance = validate_importance(parts[6]) if len(parts) > 6 else 5.0

                    # search_content: summary + keywords (for embedding/BM25)
                    search_content = f"{search_summary} | {keywords}"
                    # content: actual data (for LLM context)
                    content = actual_content

                elif is_v3_format:
                    # v3 format: summary + keywords → both go to content (legacy)
                    summary = parts[1].strip()
                    keywords = parts[2].strip()
                    memory_type = validate_memory_type(parts[3])
                    concepts_str = parts[4] if len(parts) > 4 else ""
                    importance = validate_importance(parts[5]) if len(parts) > 5 else 5.0

                    # Legacy: combine summary and keywords for content
                    content = f"Summary: {summary} | Keywords: {keywords}"
                    search_content = None  # Use content for search

                else:
                    # v2 format: content only
                    content = parts[1].strip()
                    memory_type = validate_memory_type(parts[2]) if len(parts) > 2 else "fact"
                    concepts_str = parts[3] if len(parts) > 3 else ""
                    importance = validate_importance(parts[4]) if len(parts) > 4 else 5.0
                    search_content = None  # Use content for search

                if not content or len(content) < 10:
                    continue

                # Parse concepts (comma-separated)
                concept_ids: list[str] = []
                if concepts_str:
                    concept_names = [c.strip() for c in concepts_str.split(",")]
                    for name in concept_names:
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
                    search_content=search_content,
                    concept_ids=concept_ids,
                    source_doc_ids=[doc_id] if doc_id else [],
                    memory_type=memory_type,
                    importance=importance,
                )
                memories.append(memory)

            elif line.startswith("PERSON|"):
                parts = line.split("|")
                if len(parts) >= 2:
                    name = parts[1].strip()
                    role = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                    team = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
                    if name and len(name) >= 2:  # Skip empty or single-char
                        persons.append((name, role, team))

        return MemoryExtractionResult(
            memories=memories,
            concepts=list(all_concepts.values()),
            persons=persons,
        )

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
            persons=memory_result.persons,
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
