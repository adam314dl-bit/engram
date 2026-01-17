"""Enhanced relationship extraction with edge embeddings and weights."""

import logging
from dataclasses import dataclass

from engram.ingestion.concept_extractor import ConceptExtractor
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.models import Concept, ConceptRelation
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


# Relationship type weights (based on semantic strength)
RELATION_TYPE_WEIGHTS: dict[str, float] = {
    "is_a": 0.95,      # Strong hierarchical relationship
    "contains": 0.85,  # Strong compositional relationship
    "uses": 0.75,      # Functional relationship
    "needs": 0.70,     # Dependency relationship
    "causes": 0.65,    # Causal relationship
    "related_to": 0.50,  # Weak/general relationship
}


RELATIONSHIP_EXTRACTION_PROMPT = """Проанализируй связи между концептами в тексте.

Текст:
{content}

Концепты для анализа: {concepts}

Для каждой пары связанных концептов определи:
- source: исходный концепт
- target: целевой концепт
- type: тип связи (is_a, contains, uses, needs, causes, related_to)
- description: краткое описание связи (5-10 слов)
- strength: сила связи 0.1-1.0 (1.0 = очень сильная, прямая связь)

Выведи ТОЛЬКО валидный JSON:
{{
  "relations": [
    {{
      "source": "docker",
      "target": "container",
      "type": "uses",
      "description": "Docker creates and manages containers",
      "strength": 0.9
    }}
  ]
}}

Извлеки только явные связи, присутствующие в тексте.
Не добавляй комментарии, только JSON."""


@dataclass
class EnhancedRelation:
    """Relationship with embedding and computed weight."""

    relation: ConceptRelation
    description: str
    raw_strength: float


class RelationshipExtractor:
    """Extract and enhance concept relationships with embeddings."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.embeddings = embedding_service or get_embedding_service()

    async def extract_relationships(
        self,
        content: str,
        concepts: list[Concept],
    ) -> list[ConceptRelation]:
        """
        Extract relationships between concepts from content.

        Args:
            content: Text to analyze
            concepts: List of concepts to find relationships between

        Returns:
            List of ConceptRelation with embeddings and weights
        """
        if len(concepts) < 2:
            return []

        # Build concept name list and mapping
        concept_names = [c.name for c in concepts]
        name_to_id = {c.name: c.id for c in concepts}

        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            content=content[:6000],
            concepts=", ".join(concept_names[:30]),  # Limit concept list
        )

        try:
            result = await self.llm.generate_json(prompt, temperature=0.3)
        except ValueError as e:
            logger.warning(f"Failed to extract relationships: {e}")
            return []

        if not isinstance(result, dict):
            return []

        raw_relations = result.get("relations", [])
        relations: list[ConceptRelation] = []
        descriptions: list[str] = []

        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue

            source_name = raw.get("source", "").strip().lower()
            target_name = raw.get("target", "").strip().lower()

            # Skip if concepts not found
            if source_name not in name_to_id or target_name not in name_to_id:
                continue

            # Skip self-relations
            if source_name == target_name:
                continue

            relation_type = raw.get("type", "related_to").strip().lower()
            if relation_type not in RELATION_TYPE_WEIGHTS:
                relation_type = "related_to"

            description = raw.get("description", f"{source_name} {relation_type} {target_name}")
            raw_strength = float(raw.get("strength", 0.5))
            raw_strength = max(0.1, min(1.0, raw_strength))

            # Compute final weight: type_weight × raw_strength
            type_weight = RELATION_TYPE_WEIGHTS.get(relation_type, 0.5)
            final_weight = type_weight * raw_strength

            relation = ConceptRelation(
                source_id=name_to_id[source_name],
                target_id=name_to_id[target_name],
                relation_type=relation_type,
                weight=final_weight,
            )
            relations.append(relation)
            descriptions.append(description)

        # Generate edge embeddings for query-oriented filtering
        if relations and descriptions:
            embeddings = await self.embeddings.embed_batch(descriptions)
            for relation, emb in zip(relations, embeddings, strict=True):
                relation.edge_embedding = emb

        logger.debug(f"Extracted {len(relations)} relationships with embeddings")
        return relations

    async def enhance_existing_relations(
        self,
        relations: list[ConceptRelation],
        concepts: list[Concept],
    ) -> list[ConceptRelation]:
        """
        Add embeddings to existing relations based on concept info.

        Used when relations were extracted without embeddings.
        """
        if not relations:
            return relations

        # Build ID to concept mapping
        id_to_concept = {c.id: c for c in concepts}

        descriptions: list[str] = []
        for rel in relations:
            source = id_to_concept.get(rel.source_id)
            target = id_to_concept.get(rel.target_id)

            if source and target:
                desc = f"{source.name} {rel.relation_type} {target.name}"
                if source.description:
                    desc += f" ({source.description})"
            else:
                desc = f"{rel.source_id} {rel.relation_type} {rel.target_id}"

            descriptions.append(desc)

        # Generate embeddings
        embeddings = await self.embeddings.embed_batch(descriptions)

        for relation, emb in zip(relations, embeddings, strict=True):
            relation.edge_embedding = emb

        return relations


async def extract_relationships(
    content: str,
    concepts: list[Concept],
) -> list[ConceptRelation]:
    """Convenience function to extract relationships."""
    extractor = RelationshipExtractor()
    return await extractor.extract_relationships(content, concepts)
