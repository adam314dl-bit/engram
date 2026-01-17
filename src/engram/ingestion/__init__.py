"""Ingestion layer for Engram - parsing and extraction."""

from engram.ingestion.concept_extractor import ConceptExtractor, ExtractionResult, extract_concepts
from engram.ingestion.llm_client import LLMClient, close_llm_client, get_llm_client
from engram.ingestion.memory_extractor import (
    MemoryExtractionResult,
    MemoryExtractor,
    extract_memories,
)
from engram.ingestion.parser import (
    DocumentParser,
    create_chunks,
    generate_id,
    parse_content,
    parse_file,
)
from engram.ingestion.pipeline import IngestionPipeline, IngestionResult
from engram.ingestion.relationship_extractor import (
    RelationshipExtractor,
    extract_relationships,
)

__all__ = [
    # Parser
    "DocumentParser",
    "parse_file",
    "parse_content",
    "create_chunks",
    "generate_id",
    # LLM
    "LLMClient",
    "get_llm_client",
    "close_llm_client",
    # Concept extraction
    "ConceptExtractor",
    "ExtractionResult",
    "extract_concepts",
    # Memory extraction
    "MemoryExtractor",
    "MemoryExtractionResult",
    "extract_memories",
    # Relationship extraction
    "RelationshipExtractor",
    "extract_relationships",
    # Pipeline
    "IngestionPipeline",
    "IngestionResult",
]
