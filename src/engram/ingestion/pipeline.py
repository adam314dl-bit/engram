"""Ingestion pipeline - orchestrates document processing."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from engram.ingestion.concept_extractor import ConceptExtractor
from engram.ingestion.memory_extractor import MemoryExtractor
from engram.ingestion.parser import DocumentParser
from engram.ingestion.relationship_extractor import RelationshipExtractor
from engram.models import Concept, Document
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document: Document
    concepts_created: int
    memories_created: int
    relations_created: int
    errors: list[str]


class IngestionPipeline:
    """
    Pipeline for ingesting documents into Engram.

    Steps:
    1. Parse document
    2. Extract concepts
    3. Extract memories
    4. Generate embeddings
    5. Store in Neo4j
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        parser: DocumentParser | None = None,
        concept_extractor: ConceptExtractor | None = None,
        memory_extractor: MemoryExtractor | None = None,
        relationship_extractor: RelationshipExtractor | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.db = neo4j_client
        self.parser = parser or DocumentParser()
        self.concept_extractor = concept_extractor or ConceptExtractor()
        self.memory_extractor = memory_extractor or MemoryExtractor(
            concept_extractor=self.concept_extractor
        )
        self.embeddings = embedding_service or get_embedding_service()
        self.relationship_extractor = relationship_extractor or RelationshipExtractor(
            embedding_service=self.embeddings
        )

    async def ingest_file(self, path: Path | str) -> IngestionResult:
        """Ingest a single file."""
        if isinstance(path, str):
            path = Path(path)

        logger.info(f"Ingesting file: {path}")

        # Parse document
        document = await self.parser.parse_file(path)
        return await self.ingest_document(document)

    async def ingest_content(self, content: str, title: str) -> IngestionResult:
        """Ingest raw content."""
        document = self.parser.parse_content(content, title)
        return await self.ingest_document(document)

    async def ingest_document(self, document: Document) -> IngestionResult:
        """Ingest a parsed document."""
        errors: list[str] = []
        concepts_created = 0
        memories_created = 0
        relations_created = 0

        try:
            # Update document status
            document.status = "processing"
            await self.db.save_document(document)

            # Extract concepts and relations
            logger.debug(f"Extracting concepts from: {document.title}")
            concept_result = await self.concept_extractor.extract(document.content)

            # Extract memories
            logger.debug(f"Extracting memories from: {document.title}")
            memory_result = await self.memory_extractor.extract(
                document.content,
                document.title,
                document.id,
            )

            # Merge concepts
            all_concepts: dict[str, Concept] = {}
            for c in concept_result.concepts:
                all_concepts[c.name] = c
            for c in memory_result.concepts:
                if c.name not in all_concepts:
                    all_concepts[c.name] = c

            concepts = list(all_concepts.values())
            memories = memory_result.memories

            # Extract enhanced relationships with embeddings
            logger.debug(f"Extracting relationships for {len(concepts)} concepts")
            relations = await self.relationship_extractor.extract_relationships(
                document.content, concepts
            )

            # Fall back to basic relations if enhanced extraction fails
            if not relations:
                relations = await self.relationship_extractor.enhance_existing_relations(
                    concept_result.relations, concepts
                )

            # Generate embeddings for concepts
            if concepts:
                logger.debug(f"Generating embeddings for {len(concepts)} concepts")
                concept_texts = [c.name + (f": {c.description}" if c.description else "") for c in concepts]
                embeddings = await self.embeddings.embed_batch(concept_texts)
                for concept, emb in zip(concepts, embeddings):
                    concept.embedding = emb

            # Generate embeddings for memories
            if memories:
                logger.debug(f"Generating embeddings for {len(memories)} memories")
                memory_texts = [m.content for m in memories]
                embeddings = await self.embeddings.embed_batch(memory_texts)
                for memory, emb in zip(memories, embeddings):
                    memory.embedding = emb

            # Save concepts
            for concept in concepts:
                await self.db.save_concept(concept)
                concepts_created += 1

            # Save concept relations
            for relation in relations:
                await self.db.save_concept_relation(relation)
                relations_created += 1

            # Save memories and link to concepts
            for memory in memories:
                await self.db.save_semantic_memory(memory)
                memories_created += 1

                # Link to concepts
                for concept_id in memory.concept_ids:
                    await self.db.link_memory_to_concept(memory.id, concept_id)

                # Link to document
                await self.db.link_memory_to_document(memory.id, document.id)

            # Update document status
            document.status = "completed"
            document.processed_at = datetime.utcnow()
            document.extracted_concept_ids = [c.id for c in concepts]
            document.extracted_memory_ids = [m.id for m in memories]
            await self.db.save_document(document)

            logger.info(
                f"Ingested '{document.title}': "
                f"{concepts_created} concepts, {memories_created} memories, "
                f"{relations_created} relations"
            )

        except Exception as e:
            error_msg = f"Error ingesting document: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            document.status = "failed"
            document.error_message = str(e)
            await self.db.save_document(document)

        return IngestionResult(
            document=document,
            concepts_created=concepts_created,
            memories_created=memories_created,
            relations_created=relations_created,
            errors=errors,
        )

    async def ingest_directory(
        self,
        directory: Path | str,
        extensions: list[str] | None = None,
    ) -> list[IngestionResult]:
        """Ingest all matching files in a directory."""
        if isinstance(directory, str):
            directory = Path(directory)

        documents = await self.parser.parse_directory(directory, extensions)
        results: list[IngestionResult] = []

        for doc in documents:
            result = await self.ingest_document(doc)
            results.append(result)

        total_concepts = sum(r.concepts_created for r in results)
        total_memories = sum(r.memories_created for r in results)
        logger.info(
            f"Directory ingestion complete: {len(results)} documents, "
            f"{total_concepts} concepts, {total_memories} memories"
        )

        return results
