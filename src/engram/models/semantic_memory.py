"""Semantic memory model - facts, procedures, and relationships."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

MemoryType = Literal["fact", "procedure", "relationship"]
# Extended status for ACT-R forgetting system:
# - active: Normal retrieval, high activation
# - deprioritized: Reduced retrieval weight, low activation
# - archived: Excluded from retrieval, very low activation (can be restored)
# - superseded: Replaced by newer information
# - invalid: Marked as incorrect
MemoryStatus = Literal["active", "deprioritized", "archived", "superseded", "invalid"]


def parse_datetime(value: Any) -> datetime | None:
    """Parse datetime from various formats (string, Neo4j DateTime, or native datetime)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    # Neo4j DateTime object - convert to Python datetime
    if hasattr(value, "to_native"):
        return value.to_native()
    # Try string conversion as fallback
    return datetime.fromisoformat(str(value))


@dataclass
class SemanticMemory:
    """
    Represents a unit of knowledge: facts, procedures, or relationships.

    Examples:
    - Fact: "Docker is a container platform"
    - Procedure: "To free disk space, run docker system prune"
    - Relationship: "Containers require images to run"
    """

    id: str
    content: str  # The knowledge itself

    # Optional structure for structured facts (SPO triples)
    subject: str | None = None  # "Docker"
    predicate: str | None = None  # "uses"
    object: str | None = None  # "containers"

    # Links to concepts and sources
    concept_ids: list[str] = field(default_factory=list)  # Connected concepts
    source_doc_ids: list[str] = field(default_factory=list)  # Document provenance
    source_episode_ids: list[str] = field(default_factory=list)  # If crystallized

    # Type classification
    memory_type: MemoryType = "fact"

    # Confidence and strength
    importance: float = 5.0  # 1-10 scale, LLM-assigned
    confidence: float = 0.8  # 0-1
    strength: float = 2.5  # SM-2 easiness factor (1.3-2.5)

    # Temporal metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Validity window (optional, for time-sensitive knowledge)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    status: MemoryStatus = "active"

    # ACT-R base-level activation fields
    base_level_activation: float | None = None  # Computed B_i value
    activation_updated_at: datetime | None = None  # When activation was last computed
    access_history: list[datetime] = field(default_factory=list)  # Full access history for ACT-R

    # Supersession tracking (for contradiction resolution)
    superseded_by: str | None = None  # ID of memory that supersedes this one
    superseded_at: datetime | None = None

    # Embedding for semantic search
    embedding: list[float] | None = None

    # v3.3: Flexible metadata for source type, person info, table links, etc.
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.id,
            "content": self.content,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "concept_ids": self.concept_ids,
            "source_doc_ids": self.source_doc_ids,
            "source_episode_ids": self.source_episode_ids,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "confidence": self.confidence,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "status": self.status,
            "base_level_activation": self.base_level_activation,
            "activation_updated_at": (
                self.activation_updated_at.isoformat() if self.activation_updated_at else None
            ),
            "access_history": [t.isoformat() for t in self.access_history],
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at.isoformat() if self.superseded_at else None,
            "embedding": self.embedding,
            # Serialize metadata to JSON string for Neo4j (doesn't support nested maps)
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticMemory":
        """Create from dictionary (Neo4j record)."""
        # Parse access_history
        access_history_raw = data.get("access_history", [])
        access_history = []
        if access_history_raw:
            for t in access_history_raw:
                parsed = parse_datetime(t)
                if parsed:
                    access_history.append(parsed)

        return cls(
            id=data["id"],
            content=data["content"],
            subject=data.get("subject"),
            predicate=data.get("predicate"),
            object=data.get("object"),
            concept_ids=data.get("concept_ids", []),
            source_doc_ids=data.get("source_doc_ids", []),
            source_episode_ids=data.get("source_episode_ids", []),
            memory_type=data.get("memory_type", "fact"),
            importance=data.get("importance", 5.0),
            confidence=data.get("confidence", 0.8),
            strength=data.get("strength", 2.5),
            created_at=parse_datetime(data.get("created_at")) or datetime.utcnow(),
            last_accessed=parse_datetime(data.get("last_accessed")) or datetime.utcnow(),
            access_count=data.get("access_count", 0),
            valid_from=parse_datetime(data.get("valid_from")),
            valid_until=parse_datetime(data.get("valid_until")),
            status=data.get("status", "active"),
            base_level_activation=data.get("base_level_activation"),
            activation_updated_at=parse_datetime(data.get("activation_updated_at")),
            access_history=access_history,
            superseded_by=data.get("superseded_by"),
            superseded_at=parse_datetime(data.get("superseded_at")),
            embedding=data.get("embedding"),
            # Deserialize metadata from JSON string
            metadata=json.loads(data["metadata"]) if data.get("metadata") else None,
        )

    def is_valid(self) -> bool:
        """Check if memory is currently valid based on status and time window."""
        if self.status != "active":
            return False
        now = datetime.utcnow()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True
