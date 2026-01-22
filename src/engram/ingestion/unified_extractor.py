"""Unified extraction - all knowledge extraction in 1 LLM call.

v3.6: Combines concept, memory, relation, and person extraction into a single prompt.
This reduces LLM calls from 3 to 1 per document, providing ~3x speedup.
"""

import logging
from dataclasses import dataclass, field

from engram.ingestion.concept_extractor import (
    normalize_concept_name,
    validate_concept_type,
    validate_relation_type,
)
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.memory_extractor import validate_importance, validate_memory_type
from engram.ingestion.parser import generate_id
from engram.ingestion.prompts import EXTRACTION_SYSTEM_PROMPT, UNIFIED_EXTRACTION_PROMPT
from engram.models import Concept, ConceptRelation, SemanticMemory

logger = logging.getLogger(__name__)


# Relationship type weights (based on semantic strength)
RELATION_TYPE_WEIGHTS: dict[str, float] = {
    "is_a": 0.95,
    "contains": 0.85,
    "uses": 0.75,
    "needs": 0.70,
    "causes": 0.65,
    "related_to": 0.50,
}


@dataclass
class UnifiedExtractionResult:
    """Result of unified extraction - all knowledge types in one."""

    concepts: list[Concept] = field(default_factory=list)
    relations: list[ConceptRelation] = field(default_factory=list)
    memories: list[SemanticMemory] = field(default_factory=list)
    persons: list[tuple[str, str | None, str | None]] = field(default_factory=list)


class UnifiedExtractor:
    """Extract concepts, relations, memories, and persons in a single LLM call.

    v3.6: Combines all extraction tasks into one prompt for maximum efficiency.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or get_llm_client()
        self._concept_cache: dict[str, Concept] = {}  # name -> Concept

    async def extract(
        self,
        content: str,
        title: str,
        doc_id: str | None = None,
    ) -> UnifiedExtractionResult:
        """
        Extract all knowledge types from document in a single LLM call.

        Args:
            content: Document text
            title: Document title for context
            doc_id: Optional document ID for provenance

        Returns:
            UnifiedExtractionResult with concepts, relations, memories, and persons
        """
        prompt = UNIFIED_EXTRACTION_PROMPT.format(
            title=title,
            content=content[:8000],  # Limit content size
        )

        try:
            result = await self.llm.generate(
                prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=8192,
            )
        except Exception as e:
            logger.warning(f"Failed unified extraction: {e}")
            return UnifiedExtractionResult()

        return self._parse_result(result, doc_id)

    def _parse_result(self, raw: str, doc_id: str | None) -> UnifiedExtractionResult:
        """Parse all extraction types from LLM output."""
        concepts: list[Concept] = []
        relations: list[ConceptRelation] = []
        memories: list[SemanticMemory] = []
        persons: list[tuple[str, str | None, str | None]] = []

        # Build name -> id mapping as we parse concepts
        concept_name_to_id: dict[str, str] = {}

        for line in raw.split("\n"):
            line = line.strip()

            if line.startswith("CONCEPT|"):
                concept = self._parse_concept_line(line)
                if concept:
                    concepts.append(concept)
                    concept_name_to_id[concept.name] = concept.id

            elif line.startswith("RELATION|"):
                relation = self._parse_relation_line(line, concept_name_to_id)
                if relation:
                    relations.append(relation)

            elif line.startswith("MEMORY|"):
                memory = self._parse_memory_line(line, doc_id, concept_name_to_id)
                if memory:
                    memories.append(memory)

            elif line.startswith("PERSON|"):
                person = self._parse_person_line(line)
                if person:
                    persons.append(person)

        logger.debug(
            f"Unified extraction: {len(concepts)} concepts, "
            f"{len(relations)} relations, {len(memories)} memories, "
            f"{len(persons)} persons"
        )

        return UnifiedExtractionResult(
            concepts=concepts,
            relations=relations,
            memories=memories,
            persons=persons,
        )

    def _parse_concept_line(self, line: str) -> Concept | None:
        """Parse CONCEPT|name|type|description line."""
        parts = line.split("|")
        if len(parts) < 3:
            return None

        name = normalize_concept_name(parts[1])
        if not name:
            return None

        # Check cache to avoid duplicates
        if name in self._concept_cache:
            return self._concept_cache[name]

        concept_type = validate_concept_type(parts[2]) if len(parts) > 2 else "general"
        description = parts[3].strip() if len(parts) > 3 else None

        concept = Concept(
            id=generate_id(),
            name=name,
            type=concept_type,
            description=description,
        )

        self._concept_cache[name] = concept
        return concept

    def _parse_relation_line(
        self,
        line: str,
        concept_name_to_id: dict[str, str],
    ) -> ConceptRelation | None:
        """Parse RELATION|source|target|type|strength line."""
        parts = line.split("|")
        if len(parts) < 4:
            return None

        source_name = normalize_concept_name(parts[1])
        target_name = normalize_concept_name(parts[2])

        # Skip if concepts not found
        if source_name not in concept_name_to_id or target_name not in concept_name_to_id:
            # Try to find in cache (for concepts from previous documents)
            source_id = concept_name_to_id.get(source_name)
            target_id = concept_name_to_id.get(target_name)

            if source_id is None and source_name in self._concept_cache:
                source_id = self._concept_cache[source_name].id
            if target_id is None and target_name in self._concept_cache:
                target_id = self._concept_cache[target_name].id

            if source_id is None or target_id is None:
                return None
        else:
            source_id = concept_name_to_id[source_name]
            target_id = concept_name_to_id[target_name]

        # Skip self-relations
        if source_id == target_id:
            return None

        relation_type = validate_relation_type(parts[3]) if len(parts) > 3 else "related_to"

        # Parse strength (0.1-1.0)
        try:
            raw_strength = float(parts[4]) if len(parts) > 4 else 0.5
        except ValueError:
            raw_strength = 0.5
        raw_strength = max(0.1, min(1.0, raw_strength))

        # Compute final weight: type_weight × raw_strength
        type_weight = RELATION_TYPE_WEIGHTS.get(relation_type, 0.5)
        final_weight = type_weight * raw_strength

        return ConceptRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=final_weight,
        )

    def _parse_memory_line(
        self,
        line: str,
        doc_id: str | None,
        concept_name_to_id: dict[str, str],
    ) -> SemanticMemory | None:
        """Parse MEMORY|summary|keywords|type|concepts|importance line."""
        parts = line.split("|")
        if len(parts) < 5:
            return None

        # New format: MEMORY|summary|keywords|type|concepts|importance (6 parts)
        summary = parts[1].strip()
        keywords = parts[2].strip()
        memory_type = validate_memory_type(parts[3]) if len(parts) > 3 else "fact"
        concepts_str = parts[4] if len(parts) > 4 else ""
        importance = validate_importance(parts[5]) if len(parts) > 5 else 5.0

        # Combine summary and keywords for content
        content = f"Summary: {summary} | Keywords: {keywords}"

        if not content or len(content) < 10:
            return None

        # Parse concept IDs
        concept_ids: list[str] = []
        if concepts_str:
            concept_names = [c.strip() for c in concepts_str.split(",")]
            for name in concept_names:
                normalized = normalize_concept_name(name)
                if not normalized:
                    continue

                # Look up concept ID
                concept_id = concept_name_to_id.get(normalized)
                if concept_id is None and normalized in self._concept_cache:
                    concept_id = self._concept_cache[normalized].id

                if concept_id:
                    concept_ids.append(concept_id)

        return SemanticMemory(
            id=generate_id(),
            content=content,
            concept_ids=concept_ids,
            source_doc_ids=[doc_id] if doc_id else [],
            memory_type=memory_type,
            importance=importance,
        )

    def _parse_person_line(
        self,
        line: str,
    ) -> tuple[str, str | None, str | None] | None:
        """Parse PERSON|name|role|team line."""
        parts = line.split("|")
        if len(parts) < 2:
            return None

        name = parts[1].strip()
        role = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        team = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None

        if not name or len(name) < 2:
            return None

        return (name, role, team)

    def get_or_create_concept(self, name: str, type_hint: str = "general") -> Concept:
        """Get existing concept or create new one."""
        normalized = normalize_concept_name(name)

        if normalized in self._concept_cache:
            return self._concept_cache[normalized]

        concept = Concept(
            id=generate_id(),
            name=normalized,
            type=validate_concept_type(type_hint),
        )
        self._concept_cache[normalized] = concept
        return concept

    def clear_cache(self) -> None:
        """Clear the concept cache."""
        self._concept_cache.clear()


# Convenience function
async def unified_extract(
    content: str,
    title: str,
    doc_id: str | None = None,
) -> UnifiedExtractionResult:
    """Extract all knowledge from content using default extractor."""
    extractor = UnifiedExtractor()
    return await extractor.extract(content, title, doc_id)
