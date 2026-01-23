"""Semantic chunking service for raw document retrieval.

Uses Chonkie SemanticChunker for intelligent document segmentation
based on semantic similarity rather than fixed character counts.
"""

import logging
import uuid
from dataclasses import dataclass

from chonkie import SemanticChunker

from engram.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document."""

    id: str
    text: str
    doc_id: str
    position: int  # Position in document (0-indexed)


class ChunkingService:
    """
    Semantic chunking service using Chonkie.

    Creates semantically coherent chunks by detecting topic boundaries
    rather than splitting at fixed character counts.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """
        Initialize the chunking service.

        Args:
            chunk_size: Target chunk size in tokens (default from settings)
            chunk_overlap: Overlap between chunks in tokens (default from settings)
            similarity_threshold: Semantic similarity threshold for boundaries
        """
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_tokens
        self.similarity_threshold = similarity_threshold or settings.chunk_semantic_threshold

        # Initialize Chonkie SemanticChunker
        # Uses sentence-transformers for semantic boundary detection
        self._chunker: SemanticChunker | None = None

    def _get_chunker(self) -> SemanticChunker:
        """Lazy initialization of the chunker."""
        if self._chunker is None:
            logger.info(
                f"Initializing SemanticChunker: "
                f"size={self.chunk_size}, overlap={self.chunk_overlap}, "
                f"threshold={self.similarity_threshold}"
            )
            self._chunker = SemanticChunker(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                chunk_size=self.chunk_size,
                similarity_threshold=self.similarity_threshold,
            )
        return self._chunker

    def chunk_document(
        self,
        text: str,
        doc_id: str,
    ) -> list[DocumentChunk]:
        """
        Chunk a document into semantically coherent pieces.

        Args:
            text: Document text to chunk
            doc_id: ID of the source document

        Returns:
            List of DocumentChunk objects with IDs and positions
        """
        if not text or not text.strip():
            logger.debug(f"Empty text for doc {doc_id}, returning no chunks")
            return []

        chunker = self._get_chunker()

        try:
            # Chonkie returns a list of Chunk objects with .text attribute
            chonkie_chunks = chunker.chunk(text)

            chunks: list[DocumentChunk] = []
            for i, chunk in enumerate(chonkie_chunks):
                chunk_text = chunk.text.strip()
                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            id=f"chunk_{uuid.uuid4().hex[:12]}",
                            text=chunk_text,
                            doc_id=doc_id,
                            position=i,
                        )
                    )

            logger.debug(f"Chunked doc {doc_id} into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document {doc_id}: {e}")
            # Fallback: return single chunk with entire text
            return [
                DocumentChunk(
                    id=f"chunk_{uuid.uuid4().hex[:12]}",
                    text=text.strip(),
                    doc_id=doc_id,
                    position=0,
                )
            ]

    def chunk_documents(
        self,
        documents: list[tuple[str, str]],
    ) -> list[DocumentChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of (doc_id, text) tuples

        Returns:
            List of all chunks across all documents
        """
        all_chunks: list[DocumentChunk] = []
        for doc_id, text in documents:
            chunks = self.chunk_document(text, doc_id)
            all_chunks.extend(chunks)
        return all_chunks


# Singleton instance
_chunking_service: ChunkingService | None = None


def get_chunking_service() -> ChunkingService:
    """Get or create the global chunking service."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service
