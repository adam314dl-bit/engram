"""Concept extraction from documents using LLM."""

import logging
from dataclasses import dataclass

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.ingestion.prompts import CONCEPT_EXTRACTION_PROMPT
from engram.models import Concept, ConceptRelation, ConceptType

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of concept extraction."""

    concepts: list[Concept]
    relations: list[ConceptRelation]


def normalize_concept_name(name: str) -> str:
    """Normalize concept name to lowercase, trimmed."""
    return name.strip().lower()


def validate_concept_type(type_str: str) -> ConceptType:
    """Validate and normalize concept type."""
    valid_types: set[ConceptType] = {"tool", "resource", "action", "state", "config", "error", "general"}
    normalized = type_str.strip().lower()
    if normalized in valid_types:
        return normalized  # type: ignore[return-value]
    return "general"


def validate_relation_type(type_str: str) -> str:
    """Validate relation type."""
    valid_types = {"uses", "needs", "causes", "contains", "is_a", "related_to"}
    normalized = type_str.strip().lower()
    if normalized in valid_types:
        return normalized
    return "related_to"


class ConceptExtractor:
    """Extract concepts and relationships from text using LLM."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or get_llm_client()
        self._concept_cache: dict[str, Concept] = {}  # name -> Concept

    async def extract(self, content: str) -> ExtractionResult:
        """
        Extract concepts and relations from content.

        Args:
            content: Text to extract concepts from

        Returns:
            ExtractionResult with concepts and relations
        """
        prompt = CONCEPT_EXTRACTION_PROMPT.format(content=content[:8000])  # Limit content size

        try:
            result = await self.llm.generate_json(prompt, temperature=0.3)
        except ValueError as e:
            logger.warning(f"Failed to extract concepts: {e}")
            return ExtractionResult(concepts=[], relations=[])

        if not isinstance(result, dict):
            logger.warning(f"Unexpected result type: {type(result)}")
            return ExtractionResult(concepts=[], relations=[])

        concepts = self._parse_concepts(result.get("concepts", []))
        relations = self._parse_relations(result.get("relations", []), concepts)

        return ExtractionResult(concepts=concepts, relations=relations)

    def _parse_concepts(self, raw_concepts: list[dict]) -> list[Concept]:
        """Parse raw concept data into Concept objects."""
        concepts: list[Concept] = []

        for raw in raw_concepts:
            if not isinstance(raw, dict):
                continue

            name = raw.get("name", "").strip()
            if not name:
                continue

            normalized_name = normalize_concept_name(name)

            # Check cache to avoid duplicates
            if normalized_name in self._concept_cache:
                concepts.append(self._concept_cache[normalized_name])
                continue

            concept = Concept(
                id=generate_id(),
                name=normalized_name,
                type=validate_concept_type(raw.get("type", "general")),
                description=raw.get("description"),
            )

            self._concept_cache[normalized_name] = concept
            concepts.append(concept)

        return concepts

    def _parse_relations(
        self, raw_relations: list[dict], concepts: list[Concept]
    ) -> list[ConceptRelation]:
        """Parse raw relation data into ConceptRelation objects."""
        # Build name -> id mapping
        name_to_id: dict[str, str] = {}
        for concept in concepts:
            name_to_id[concept.name] = concept.id
        # Also include cached concepts
        for name, concept in self._concept_cache.items():
            name_to_id[name] = concept.id

        relations: list[ConceptRelation] = []

        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue

            source_name = normalize_concept_name(raw.get("source", ""))
            target_name = normalize_concept_name(raw.get("target", ""))

            if source_name not in name_to_id or target_name not in name_to_id:
                # Skip relations with unknown concepts
                continue

            relation = ConceptRelation(
                source_id=name_to_id[source_name],
                target_id=name_to_id[target_name],
                relation_type=validate_relation_type(raw.get("type", "related_to")),
                weight=0.7,  # Default weight, can be adjusted later
            )
            relations.append(relation)

        return relations

    def get_or_create_concept(self, name: str, type_hint: ConceptType = "general") -> Concept:
        """Get existing concept or create new one."""
        normalized = normalize_concept_name(name)

        if normalized in self._concept_cache:
            return self._concept_cache[normalized]

        concept = Concept(
            id=generate_id(),
            name=normalized,
            type=type_hint,
        )
        self._concept_cache[normalized] = concept
        return concept

    def clear_cache(self) -> None:
        """Clear the concept cache."""
        self._concept_cache.clear()


# Convenience function for one-off extraction
async def extract_concepts(content: str) -> ExtractionResult:
    """Extract concepts from content using default extractor."""
    extractor = ConceptExtractor()
    return await extractor.extract(content)
