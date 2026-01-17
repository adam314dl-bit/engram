"""Concept node model - atomic ideas/entities in the knowledge graph."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

ConceptType = Literal["tool", "resource", "action", "state", "config", "error", "general"]


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
class Concept:
    """
    Represents an atomic idea or entity in the concept network.

    Examples: "Docker", "disk space", "prune", "container"
    """

    id: str
    name: str  # Normalized, lowercase
    type: ConceptType  # tool, resource, action, state, config, error, general

    # Optional description if available from source
    description: str | None = None

    # Embedding vector for semantic similarity
    embedding: list[float] | None = None

    # Hierarchy (optional, for is-a relationships)
    parent_id: str | None = None
    level: int = 2  # 0=root, 1=domain, 2=concept, 3=instance

    # Usage statistics
    activation_count: int = 0
    last_activated: datetime | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "embedding": self.embedding,
            "parent_id": self.parent_id,
            "level": self.level,
            "activation_count": self.activation_count,
            "last_activated": self.last_activated.isoformat() if self.last_activated else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Concept":
        """Create from dictionary (Neo4j record)."""
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            description=data.get("description"),
            embedding=data.get("embedding"),
            parent_id=data.get("parent_id"),
            level=data.get("level", 2),
            activation_count=data.get("activation_count", 0),
            last_activated=parse_datetime(data.get("last_activated")),
            created_at=parse_datetime(data.get("created_at")) or datetime.utcnow(),
            updated_at=parse_datetime(data.get("updated_at")) or datetime.utcnow(),
        )


@dataclass
class ConceptRelation:
    """
    Represents a weighted, typed edge between two concepts.

    Example: Docker --uses--> container (weight: 0.9)
    """

    source_id: str
    target_id: str
    relation_type: str  # uses, needs, causes, contains, is_a, related_to
    weight: float = 0.5  # 0.0 - 1.0

    # Optional embedding for query-oriented edge filtering
    edge_embedding: list[float] | None = None

    # Statistics
    co_occurrence_count: int = 1
    last_used: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "edge_embedding": self.edge_embedding,
            "co_occurrence_count": self.co_occurrence_count,
            "last_used": self.last_used.isoformat(),
        }
