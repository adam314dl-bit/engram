"""Document model - source documents for knowledge extraction."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

DocumentType = Literal["markdown", "text", "html", "pdf"]
ProcessingStatus = Literal["pending", "processing", "completed", "failed"]


@dataclass
class Document:
    """
    Represents a source document for knowledge extraction.

    Tracks provenance and processing status.
    """

    id: str
    title: str
    content: str  # Raw content
    doc_type: DocumentType = "markdown"

    # Source information
    source_path: str | None = None  # File path or URL
    source_hash: str | None = None  # For deduplication

    # Processing status
    status: ProcessingStatus = "pending"
    error_message: str | None = None

    # Extracted data references
    extracted_concept_ids: list[str] = field(default_factory=list)
    extracted_memory_ids: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None

    # Optional metadata from document
    author: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "doc_type": self.doc_type,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "status": self.status,
            "error_message": self.error_message,
            "extracted_concept_ids": self.extracted_concept_ids,
            "extracted_memory_ids": self.extracted_memory_ids,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create from dictionary (Neo4j record)."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            doc_type=data.get("doc_type", "markdown"),
            source_path=data.get("source_path"),
            source_hash=data.get("source_hash"),
            status=data.get("status", "pending"),
            error_message=data.get("error_message"),
            extracted_concept_ids=data.get("extracted_concept_ids", []),
            extracted_memory_ids=data.get("extracted_memory_ids", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.utcnow()
            ),
            processed_at=(
                datetime.fromisoformat(data["processed_at"])
                if data.get("processed_at")
                else None
            ),
            author=data.get("author"),
            version=data.get("version"),
            tags=data.get("tags", []),
        )


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document for processing.

    Used when documents are too large to process at once.
    """

    id: str
    document_id: str
    content: str
    chunk_index: int
    start_offset: int
    end_offset: int

    # Processing
    processed: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "processed": self.processed,
        }
