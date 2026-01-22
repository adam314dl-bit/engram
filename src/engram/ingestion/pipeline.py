"""Ingestion pipeline - orchestrates document processing.

v3.6: Combined extraction - 1 LLM call per document (3x faster than v3.5).
      Unified prompt extracts concepts, memories, relations, and persons together.

v3.5: Simplified pipeline - removed table/list extraction (handled by LLM directly).
      Added batch Neo4j writes for 30-40% speedup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import asyncio

from engram.config import settings
from engram.ingestion.parser import DocumentParser, generate_id
from engram.ingestion.person_extractor import PersonExtractor
from engram.ingestion.unified_extractor import UnifiedExtractor
from engram.models import Concept, Document, SemanticMemory
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.quality_filter import is_quality_chunk
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

    v3.6 Combined extraction (1 LLM call per document):
    1. Parse document
    2. Extract persons (regex/Natasha - backup, no LLM)
    3. Unified extraction (1 LLM call - concepts, memories, relations, persons)
    4. Merge persons from regex and LLM
    5. Apply quality filtering
    6. Generate embeddings (batch)
    7. Store in Neo4j (batch)
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        parser: DocumentParser | None = None,
        unified_extractor: UnifiedExtractor | None = None,
        embedding_service: EmbeddingService | None = None,
        person_extractor: PersonExtractor | None = None,
    ) -> None:
        self.db = neo4j_client
        self.parser = parser or DocumentParser()
        self.unified_extractor = unified_extractor or UnifiedExtractor()
        self.embeddings = embedding_service or get_embedding_service()
        self.person_extractor = person_extractor or PersonExtractor()

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
        """Ingest a parsed document.

        v3.6 Combined: 1 LLM call (unified extraction).
        All extraction tasks in single prompt for maximum efficiency.
        """
        errors: list[str] = []
        concepts_created = 0
        memories_created = 0
        relations_created = 0

        try:
            # Update document status
            document.status = "processing"
            await self.db.save_document(document)

            text_content = document.content

            # --- Phase 1: Extract persons (regex/Natasha - backup, no LLM) ---
            person_result = self.person_extractor.extract(text_content, document.id)
            person_memories: list[SemanticMemory] = []

            if person_result.persons:
                logger.debug(f"Found {len(person_result.persons)} persons via regex/Natasha in: {document.title}")

                for person in person_result.persons:
                    if person.role:
                        content = f"{person.canonical_name} — {person.role}"
                        if person.team:
                            content += f" (команда {person.team})"

                        person_memories.append(SemanticMemory(
                            id=generate_id(),
                            content=content,
                            concept_ids=[],
                            source_doc_ids=[document.id],
                            memory_type="fact",
                            importance=6.0,
                            metadata={
                                "source_type": "person",
                                "person_name": person.canonical_name,
                                "person_variants": person.variants,
                                "person_role": person.role,
                                "person_team": person.team,
                            },
                        ))

            # --- Phase 2: Unified extraction (1 LLM call) ---
            logger.debug(f"Unified extraction from: {document.title}")
            extraction_result = await self.unified_extractor.extract(
                text_content,
                document.title,
                document.id,
            )

            concepts = extraction_result.concepts
            relations = extraction_result.relations

            # Merge LLM-extracted persons with regex/Natasha
            if extraction_result.persons:
                logger.debug(f"Found {len(extraction_result.persons)} persons via LLM in: {document.title}")
                existing_names = {
                    m.metadata.get("person_name", "").lower()
                    for m in person_memories
                    if m.metadata
                }

                for name, role, team in extraction_result.persons:
                    if name.lower() not in existing_names:
                        if role:
                            content = f"{name} — {role}"
                            if team:
                                content += f" (команда {team})"

                            person_memories.append(SemanticMemory(
                                id=generate_id(),
                                content=content,
                                concept_ids=[],
                                source_doc_ids=[document.id],
                                memory_type="fact",
                                importance=6.0,
                                metadata={
                                    "source_type": "person",
                                    "person_name": name,
                                    "person_variants": [],
                                    "person_role": role,
                                    "person_team": team,
                                },
                            ))
                            existing_names.add(name.lower())

            # Combine memories: text + person
            memories = extraction_result.memories + person_memories

            # Apply quality filtering to text memories
            filtered_memories: list[SemanticMemory] = []
            for memory in memories:
                source_type = memory.metadata.get("source_type") if memory.metadata else None
                if source_type == "person":
                    filtered_memories.append(memory)
                elif is_quality_chunk(memory.content, settings.chunk_quality_threshold, settings.min_chunk_words):
                    filtered_memories.append(memory)
                else:
                    logger.debug(f"Filtered low-quality memory: {memory.content[:50]}...")

            memories = filtered_memories
            logger.debug(f"After quality filtering: {len(memories)} memories")

            # --- Phase 3: Generate embeddings (combined batch) ---
            all_texts: list[str] = []
            concept_count = len(concepts)

            # Collect texts for concepts
            for c in concepts:
                all_texts.append(c.name + (f": {c.description}" if c.description else ""))

            # Collect texts for memories
            for m in memories:
                all_texts.append(m.content)

            # Single batch embedding call
            if all_texts:
                logger.debug(f"Generating embeddings for {len(all_texts)} items (batch)")
                all_embeddings = await self.embeddings.embed_batch(all_texts)

                # Assign embeddings to concepts
                for i, concept in enumerate(concepts):
                    concept.embedding = all_embeddings[i]

                # Assign embeddings to memories
                for i, memory in enumerate(memories):
                    memory.embedding = all_embeddings[concept_count + i]

            # --- Phase 4: Save to Neo4j (batch) ---
            # Save concepts batch
            if concepts:
                await self.db.save_concepts_batch(concepts)
                concepts_created = len(concepts)

            # Save relations batch
            if relations:
                await self.db.save_relations_batch(relations)
                relations_created = len(relations)

            # Save memories batch
            if memories:
                await self.db.save_memories_batch(memories)
                memories_created = len(memories)

                # Collect memory-concept and memory-document links
                memory_concept_links: list[tuple[str, str]] = []
                memory_doc_links: list[tuple[str, str]] = []

                for memory in memories:
                    for concept_id in memory.concept_ids:
                        memory_concept_links.append((memory.id, concept_id))
                    memory_doc_links.append((memory.id, document.id))

                # Batch link operations
                if memory_concept_links:
                    await self.db.link_memories_to_concepts_batch(memory_concept_links)
                if memory_doc_links:
                    await self.db.link_memories_to_documents_batch(memory_doc_links)

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
        max_concurrent: int | None = None,
        progress_callback: "Callable[[IngestionResult], None] | None" = None,
    ) -> list[IngestionResult]:
        """Ingest all matching files in a directory with parallel processing.

        Args:
            directory: Directory to process
            extensions: File extensions to include
            max_concurrent: Max parallel documents
            progress_callback: Called after each document completes (for progress bars)
        """
        if isinstance(directory, str):
            directory = Path(directory)

        documents = await self.parser.parse_directory(directory, extensions)

        if not documents:
            return []

        # Use configured or provided concurrency limit
        concurrency = max_concurrent or settings.ingestion_max_concurrent
        semaphore = asyncio.Semaphore(concurrency)

        final_results: list[IngestionResult] = []
        results_lock = asyncio.Lock()

        async def ingest_with_semaphore(doc: Document) -> IngestionResult:
            async with semaphore:
                result = await self.ingest_document(doc)
                async with results_lock:
                    final_results.append(result)
                    if progress_callback:
                        progress_callback(result)
                return result

        # Process documents in parallel
        logger.info(f"Ingesting {len(documents)} documents with max {concurrency} concurrent")
        results = await asyncio.gather(
            *[ingest_with_semaphore(doc) for doc in documents],
            return_exceptions=True,
        )

        # Handle any exceptions that weren't caught
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error ingesting document {documents[i].title}: {result}")
                error_result = IngestionResult(
                    document=documents[i],
                    concepts_created=0,
                    memories_created=0,
                    relations_created=0,
                    errors=[str(result)],
                )
                # Check if already added via callback
                if error_result.document.id not in [r.document.id for r in final_results]:
                    final_results.append(error_result)
                    if progress_callback:
                        progress_callback(error_result)

        total_concepts = sum(r.concepts_created for r in final_results)
        total_memories = sum(r.memories_created for r in final_results)
        logger.info(
            f"Directory ingestion complete: {len(final_results)} documents, "
            f"{total_concepts} concepts, {total_memories} memories"
        )

        return final_results
