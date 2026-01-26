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

    def _get_llm_config(self) -> tuple[str, str, str, float]:
        """Get LLM configuration for enrichment.

        Returns:
            Tuple of (base_url, model, api_key, timeout)
        """
        if settings.enrichment_llm_enabled:
            return (
                settings.enrichment_llm_base_url,
                settings.enrichment_llm_model,
                settings.enrichment_llm_api_key,
                settings.enrichment_llm_timeout,
            )
        return (
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_timeout,
        )

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

        base_url, model, api_key, timeout = self._get_llm_config()

        prompt = f"""Дай краткое определение понятия "{concept.name}" и его связи.

Формат ответа (разделитель |):
определение|is_a:категория1,категория2|contains:компонент1,компонент2|uses:технология1|needs:требование1

Пример для "Docker":
Платформа контейнеризации для упаковки приложений|is_a:контейнеризация,виртуализация|contains:образ,контейнер|uses:Linux|needs:ядро_Linux

Если понятие неизвестно, верни только: неизвестно
Используй только общеизвестные факты."""

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"].strip()

            # Parse pipe-delimited format
            definition_text = ""
            is_a: list[str] = []
            contains: list[str] = []
            uses: list[str] = []
            needs: list[str] = []

            if content and content != "неизвестно":
                parts = content.split("|")
                if parts:
                    definition_text = parts[0].strip()
                    for part in parts[1:]:
                        part = part.strip()
                        if part.startswith("is_a:"):
                            is_a = [x.strip() for x in part[5:].split(",") if x.strip()]
                        elif part.startswith("contains:"):
                            contains = [x.strip() for x in part[9:].split(",") if x.strip()]
                        elif part.startswith("uses:"):
                            uses = [x.strip() for x in part[5:].split(",") if x.strip()]
                        elif part.startswith("needs:"):
                            needs = [x.strip() for x in part[6:].split(",") if x.strip()]

            definition = ConceptDefinition(
                concept_id=concept.id,
                concept_name=concept.name,
                definition=definition_text,
                is_a=is_a,
                contains=contains,
                uses=uses,
                needs=needs,
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
        max_concurrent: int | None = None,
    ) -> EnrichmentResult:
        """Enrich all concepts with world knowledge.

        Args:
            batch_size: Number of concepts to process per batch
            max_concurrent: Max parallel LLM requests (default: from settings)

        Returns:
            EnrichmentResult with statistics
        """
        import asyncio

        batch_size = batch_size or self.config.max_definitions_per_batch
        if max_concurrent is None:
            if settings.enrichment_llm_enabled:
                max_concurrent = settings.enrichment_llm_max_concurrent
            else:
                max_concurrent = settings.llm_max_concurrent
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

        # Log LLM configuration
        if settings.enrichment_llm_enabled:
            logger.info(
                f"Using enrichment LLM: {settings.enrichment_llm_model} "
                f"at {settings.enrichment_llm_base_url}"
            )
        else:
            logger.info(f"Using main LLM: {settings.llm_model} at {settings.llm_base_url}")

        logger.info(f"Enriching {len(concepts)} concepts with {max_concurrent} parallel requests")

        # Semaphore to limit concurrent LLM requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def enrich_with_semaphore(concept: Concept) -> tuple[int, int, list[str]]:
            """Enrich a concept with semaphore-controlled concurrency."""
            async with semaphore:
                try:
                    edges = await self.enrich_concept(concept)
                    has_def = 1 if self._definition_cache.get(concept.id, ConceptDefinition(
                        "", "", "", [], [], [], []
                    )).definition else 0
                    return (1, has_def, len(edges), [])
                except Exception as e:
                    return (0, 0, 0, [f"Failed to enrich {concept.name}: {e}"])

        # Process in batches with parallel requests within each batch
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i : i + batch_size]

            # Run batch in parallel
            tasks = [enrich_with_semaphore(c) for c in batch]
            results_batch = await asyncio.gather(*tasks)

            # Aggregate results
            for enriched, has_def, edge_count, errs in results_batch:
                concepts_enriched += enriched
                definitions_generated += has_def
                world_knowledge_edges += edge_count
                errors.extend(errs)

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

    def _get_llm_config(self) -> tuple[str, str, str, float]:
        """Get LLM configuration for edge classification.

        Returns:
            Tuple of (base_url, model, api_key, timeout)
        """
        if settings.enrichment_llm_enabled:
            return (
                settings.enrichment_llm_base_url,
                settings.enrichment_llm_model,
                settings.enrichment_llm_api_key,
                settings.enrichment_llm_timeout,
            )
        return (
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_timeout,
        )

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

        base_url, model, api_key, timeout = self._get_llm_config()

        # Get unclassified edges
        query = """
        MATCH (a:Concept)-[r:RELATED_TO]->(b:Concept)
        WHERE r.is_semantic IS NULL
        RETURN a.name as source, b.name as target, r.type as type, id(r) as edge_id
        """
        results = await self.db.execute_query(query)

        if not results:
            return 0

        # Log LLM configuration
        if settings.enrichment_llm_enabled:
            logger.info(
                f"Using enrichment LLM: {settings.enrichment_llm_model} "
                f"at {settings.enrichment_llm_base_url}"
            )

        logger.info(f"Classifying {len(results)} edges")
        classified = 0

        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]

            # Format edges for LLM
            edges_text = "\n".join(
                f"- {r['source']} --{r['type']}--> {r['target']}"
                for r in batch
            )

            prompt = f"""Классифицируй каждую связь как семантическую (S) или контекстную (C).

Связи:
{edges_text}

Формат ответа (одна связь на строку):
source|target|S или C

Пример:
docker|контейнер|S
команда_X|инструмент_Y|C

S = семантическая (общеизвестный факт)
C = контекстная (специфично для документа)"""

            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=timeout,
                )
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"].strip()

                # Parse pipe-delimited format (one per line)
                for line in content.split("\n"):
                    line = line.strip()
                    if not line or "|" not in line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 3:
                        source = parts[0].strip()
                        target = parts[1].strip()
                        classification_type = parts[2].strip().upper()
                        is_semantic = classification_type == "S"

                        update_query = """
                        MATCH (a:Concept {name: $source})-[r:RELATED_TO]->(b:Concept {name: $target})
                        SET r.is_semantic = $is_semantic,
                            r.is_universal = $is_semantic
                        """
                        await self.db.execute_query(
                            update_query,
                            source=source,
                            target=target,
                            is_semantic=is_semantic,
                        )
                        classified += 1

            except Exception as e:
                logger.warning(f"Failed to classify batch: {e}")

        return classified
