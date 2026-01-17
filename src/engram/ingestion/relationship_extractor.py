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


RELATIONSHIP_EXTRACTION_PROMPT = """Извлеки связи между концептами из этого текста.

Текст:
{content}

Концепты для анализа: {concepts}

Выведи каждую связь на новой строке в формате:
RELATION|источник|цель|тип|описание|сила

Типы: is_a, contains, uses, needs, causes, related_to
Сила: 0.1-1.0 (1.0 = очень сильная прямая связь)

Пример:
RELATION|docker|контейнер|uses|Docker создаёт и управляет контейнерами|0.9
RELATION|kubernetes|docker|needs|Kubernetes требует среду выполнения контейнеров|0.8

Выводи ТОЛЬКО строки RELATION. Без объяснений, без другого текста.

Вывод:"""


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
            result = await self.llm.generate(
                prompt,
                system_prompt="Ты извлекаешь связи в простой строчный формат. Выводи ТОЛЬКО запрошенные строки. Без объяснений, без markdown, без лишнего текста.",
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning(f"Failed to extract relationships: {e}")
            return []

        relations: list[ConceptRelation] = []
        descriptions: list[str] = []

        for line in result.split("\n"):
            line = line.strip()
            if not line.startswith("RELATION|"):
                continue

            parts = line.split("|")
            if len(parts) < 4:
                continue

            source_name = parts[1].strip().lower()
            target_name = parts[2].strip().lower()

            # Skip if concepts not found
            if source_name not in name_to_id or target_name not in name_to_id:
                continue

            # Skip self-relations
            if source_name == target_name:
                continue

            relation_type = parts[3].strip().lower() if len(parts) > 3 else "related_to"
            if relation_type not in RELATION_TYPE_WEIGHTS:
                relation_type = "related_to"

            description = parts[4].strip() if len(parts) > 4 else f"{source_name} {relation_type} {target_name}"

            try:
                raw_strength = float(parts[5]) if len(parts) > 5 else 0.5
            except ValueError:
                raw_strength = 0.5
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
