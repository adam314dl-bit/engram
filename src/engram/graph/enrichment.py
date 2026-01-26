"""Semantic enrichment for concepts and edges.

Adds world knowledge through LLM-generated definitions and relations,
and classifies edges as semantic vs contextual.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.config import settings
from engram.graph.config import EnrichmentConfig
from engram.graph.models import EdgeSourceType, EnrichedEdge

if TYPE_CHECKING:
    from engram.models import Concept, ConceptRelation
    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# Semantic relation types that represent universal knowledge
SEMANTIC_RELATION_TYPES = {
    "is_a",
    "part_of",
    "has_part",
    "instance_of",
    "subclass_of",
    "contains",
    "uses",
    "requires",
    "produces",
    "enables",
}


@dataclass
class ConceptDefinition:
    """LLM-generated definition for a concept."""

    concept_id: str
    concept_name: str
    definition: str
    is_a: list[str]  # Hypernyms (e.g., Docker is_a containerization_tool)
    contains: list[str]  # Parts/components
    uses: list[str]  # Things it uses
    needs: list[str]  # Requirements/dependencies


@dataclass
class EnrichmentResult:
    """Result of enrichment process."""

    concepts_enriched: int
    definitions_generated: int
    edges_classified: int
    world_knowledge_edges: int
    errors: list[str]


class SemanticEnricher:
    """Enriches concepts with world knowledge and semantic relations."""

    def __init__(
        self,
        db: "Neo4jClient",
        config: EnrichmentConfig | None = None,
    ) -> None:
        self.db = db
        self.config = config or EnrichmentConfig()
        self._definition_cache: dict[str, ConceptDefinition] = {}

    async def extract_definition(self, concept: "Concept") -> ConceptDefinition:
        """Extract definition and relations from LLM for a concept.

        Args:
            concept: Concept to define

        Returns:
            ConceptDefinition with extracted knowledge
        """
        # Check cache first
        if concept.id in self._definition_cache:
            return self._definition_cache[concept.id]

        import requests

        prompt = f"""Дай краткое определение понятия "{concept.name}" и его связи.

Ответь в JSON формате:
{{
    "definition": "краткое определение (1-2 предложения)",
    "is_a": ["категория1", "категория2"],
    "contains": ["компонент1", "компонент2"],
    "uses": ["технология1", "ресурс1"],
    "needs": ["требование1", "зависимость1"]
}}

Если понятие неизвестно или слишком специфично, верни пустые списки.
Используй только общеизвестные факты, не выдумывай."""

        try:
            response = requests.post(
                f"{settings.llm_base_url}/chat/completions",
                json={
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                timeout=settings.llm_timeout,
            )
            response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"]

            # Parse JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            definition = ConceptDefinition(
                concept_id=concept.id,
                concept_name=concept.name,
                definition=data.get("definition", ""),
                is_a=data.get("is_a", []),
                contains=data.get("contains", []),
                uses=data.get("uses", []),
                needs=data.get("needs", []),
            )

            self._definition_cache[concept.id] = definition
            return definition

        except Exception as e:
            logger.warning(f"Failed to extract definition for {concept.name}: {e}")
            return ConceptDefinition(
                concept_id=concept.id,
                concept_name=concept.name,
                definition="",
                is_a=[],
                contains=[],
                uses=[],
                needs=[],
            )

    async def generate_semantic_relations(
        self,
        concept: "Concept",
        definition: ConceptDefinition,
    ) -> list[EnrichedEdge]:
        """Generate world knowledge edges from concept definition.

        Args:
            concept: Source concept
            definition: LLM-generated definition

        Returns:
            List of enriched edges representing world knowledge
        """
        edges: list[EnrichedEdge] = []

        # Create edges for is_a relations
        for target_name in definition.is_a[: self.config.max_relations_per_concept]:
            # Find or note the target concept
            target = await self.db.get_concept_by_name(target_name)
            if target:
                edges.append(
                    EnrichedEdge(
                        source_id=concept.id,
                        target_id=target.id,
                        relation_type="is_a",
                        weight=0.9,
                        is_semantic=True,
                        is_universal=True,
                        source_type=EdgeSourceType.WORLD_KNOWLEDGE,
                    )
                )

        # Create edges for contains relations
        for target_name in definition.contains[: self.config.max_relations_per_concept]:
            target = await self.db.get_concept_by_name(target_name)
            if target:
                edges.append(
                    EnrichedEdge(
                        source_id=concept.id,
                        target_id=target.id,
                        relation_type="contains",
                        weight=0.8,
                        is_semantic=True,
                        is_universal=True,
                        source_type=EdgeSourceType.WORLD_KNOWLEDGE,
                    )
                )

        # Create edges for uses relations
        for target_name in definition.uses[: self.config.max_relations_per_concept]:
            target = await self.db.get_concept_by_name(target_name)
            if target:
                edges.append(
                    EnrichedEdge(
                        source_id=concept.id,
                        target_id=target.id,
                        relation_type="uses",
                        weight=0.7,
                        is_semantic=True,
                        is_universal=True,
                        source_type=EdgeSourceType.WORLD_KNOWLEDGE,
                    )
                )

        # Create edges for needs relations
        for target_name in definition.needs[: self.config.max_relations_per_concept]:
            target = await self.db.get_concept_by_name(target_name)
            if target:
                edges.append(
                    EnrichedEdge(
                        source_id=concept.id,
                        target_id=target.id,
                        relation_type="needs",
                        weight=0.7,
                        is_semantic=True,
                        is_universal=True,
                        source_type=EdgeSourceType.WORLD_KNOWLEDGE,
                    )
                )

        return edges

    async def classify_existing_edges(self) -> int:
        """Classify existing edges as semantic vs contextual.

        Returns:
            Number of edges classified
        """
        # Mark edges with semantic relation types
        query = """
        MATCH ()-[r:RELATED_TO]->()
        WHERE r.type IN $semantic_types
        SET r.is_semantic = true, r.is_universal = true
        RETURN count(r) as count
        """
        result = await self.db.execute_query(
            query, semantic_types=list(SEMANTIC_RELATION_TYPES)
        )
        return result[0]["count"] if result else 0

    async def enrich_concept(self, concept: "Concept") -> list[EnrichedEdge]:
        """Enrich a single concept with world knowledge.

        Args:
            concept: Concept to enrich

        Returns:
            List of new edges created
        """
        definition = await self.extract_definition(concept)
        edges = await self.generate_semantic_relations(concept, definition)

        # Save edges to database
        for edge in edges:
            await self._save_enriched_edge(edge)

        # Update concept with definition
        if definition.definition:
            query = """
            MATCH (c:Concept {id: $id})
            SET c.enriched_definition = $definition,
                c.enriched_at = datetime()
            """
            await self.db.execute_query(
                query,
                id=concept.id,
                definition=definition.definition,
            )

        return edges

    async def enrich_all_concepts(
        self,
        batch_size: int | None = None,
    ) -> EnrichmentResult:
        """Enrich all concepts with world knowledge.

        Args:
            batch_size: Number of concepts to process per batch

        Returns:
            EnrichmentResult with statistics
        """
        batch_size = batch_size or self.config.max_definitions_per_batch
        errors: list[str] = []
        concepts_enriched = 0
        definitions_generated = 0
        world_knowledge_edges = 0

        # Get concepts that haven't been enriched
        query = """
        MATCH (c:Concept)
        WHERE c.enriched_at IS NULL
          AND (c.status IS NULL OR c.status = 'active')
        RETURN c
        """
        results = await self.db.execute_query(query)

        from engram.models import Concept

        concepts = [Concept.from_dict(dict(r["c"])) for r in results]
        logger.info(f"Enriching {len(concepts)} concepts")

        for i in range(0, len(concepts), batch_size):
            batch = concepts[i : i + batch_size]
            for concept in batch:
                try:
                    edges = await self.enrich_concept(concept)
                    concepts_enriched += 1
                    if self._definition_cache.get(concept.id, ConceptDefinition(
                        "", "", "", [], [], [], []
                    )).definition:
                        definitions_generated += 1
                    world_knowledge_edges += len(edges)
                except Exception as e:
                    errors.append(f"Failed to enrich {concept.name}: {e}")

            logger.info(f"Enriched {min(i + batch_size, len(concepts))}/{len(concepts)} concepts")

        # Classify existing edges
        edges_classified = await self.classify_existing_edges()

        return EnrichmentResult(
            concepts_enriched=concepts_enriched,
            definitions_generated=definitions_generated,
            edges_classified=edges_classified,
            world_knowledge_edges=world_knowledge_edges,
            errors=errors,
        )

    async def _save_enriched_edge(self, edge: EnrichedEdge) -> None:
        """Save an enriched edge to the database."""
        query = """
        MATCH (source:Concept {id: $source_id})
        MATCH (target:Concept {id: $target_id})
        MERGE (source)-[r:RELATED_TO]->(target)
        SET r.type = $relation_type,
            r.weight = $weight,
            r.is_semantic = $is_semantic,
            r.is_universal = $is_universal,
            r.source_type = $source_type,
            r.provenance_doc_id = $provenance_doc_id,
            r.updated_at = datetime()
        """
        await self.db.execute_query(
            query,
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            is_semantic=edge.is_semantic,
            is_universal=edge.is_universal,
            source_type=edge.source_type.value,
            provenance_doc_id=edge.provenance_doc_id,
        )


class EdgeClassifier:
    """Classifies edges using LLM for semantic vs contextual determination."""

    def __init__(self, db: "Neo4jClient") -> None:
        self.db = db

    async def classify_all_edges(
        self,
        batch_size: int = 50,
    ) -> int:
        """Classify all unclassified edges.

        Args:
            batch_size: Number of edges to process per LLM call

        Returns:
            Number of edges classified
        """
        import requests

        # Get unclassified edges
        query = """
        MATCH (a:Concept)-[r:RELATED_TO]->(b:Concept)
        WHERE r.is_semantic IS NULL
        RETURN a.name as source, b.name as target, r.type as type, id(r) as edge_id
        """
        results = await self.db.execute_query(query)

        if not results:
            return 0

        logger.info(f"Classifying {len(results)} edges")
        classified = 0

        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]

            # Format edges for LLM
            edges_text = "\n".join(
                f"- {r['source']} --{r['type']}--> {r['target']}"
                for r in batch
            )

            prompt = f"""Классифицируй каждую связь как семантическую (универсально истинную) или контекстную (зависящую от документа).

Связи:
{edges_text}

Ответь в JSON формате:
{{
    "classifications": [
        {{"source": "...", "target": "...", "is_semantic": true/false, "is_universal": true/false}}
    ]
}}

Семантические связи - это общеизвестные факты (Docker использует контейнеры).
Контекстные связи - это специфичные для документа (команда X использует инструмент Y)."""

            try:
                response = requests.post(
                    f"{settings.llm_base_url}/chat/completions",
                    json={
                        "model": settings.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                    },
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    timeout=settings.llm_timeout,
                )
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"]

                # Parse JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                # Update edges
                for classification in data.get("classifications", []):
                    update_query = """
                    MATCH (a:Concept {name: $source})-[r:RELATED_TO]->(b:Concept {name: $target})
                    SET r.is_semantic = $is_semantic,
                        r.is_universal = $is_universal
                    """
                    await self.db.execute_query(
                        update_query,
                        source=classification["source"],
                        target=classification["target"],
                        is_semantic=classification.get("is_semantic", False),
                        is_universal=classification.get("is_universal", False),
                    )
                    classified += 1

            except Exception as e:
                logger.warning(f"Failed to classify batch: {e}")

        return classified
