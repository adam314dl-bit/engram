"""Data models for graph quality optimization."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConceptStatus(str, Enum):
    """Status of a concept in the graph."""

    ACTIVE = "active"  # Normal, usable concept
    MERGED = "merged"  # Merged into another concept (canonical)
    ALIAS = "alias"  # Alias pointing to canonical
    PENDING_REVIEW = "pending_review"  # Needs human review


class MatchConfidence(str, Enum):
    """Confidence level for duplicate detection."""

    HIGH = "high"  # Auto-merge (>= 0.95)
    MEDIUM = "medium"  # Create POSSIBLE_DUPLICATE edge (>= 0.80)
    LOW = "low"  # Track but don't act (>= 0.60)
    NONE = "none"  # No match (< 0.60)


class EdgeSourceType(str, Enum):
    """Source/provenance of an edge."""

    ONTOLOGY = "ontology"  # From external ontology (e.g., WordNet)
    WORLD_KNOWLEDGE = "world_knowledge"  # LLM-generated world knowledge
    INFERENCE = "inference"  # Inferred from patterns
    DOCUMENT = "document"  # Extracted from document
    USER = "user"  # User-defined


@dataclass
class DuplicateCandidate:
    """A potential duplicate concept pair."""

    source_id: str
    source_name: str
    target_id: str
    target_name: str

    # Similarity scores (0.0 - 1.0)
    labse_similarity: float
    phonetic_similarity: float
    string_similarity: float
    combined_similarity: float

    # Classification
    confidence: MatchConfidence

    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "labse_similarity": self.labse_similarity,
            "phonetic_similarity": self.phonetic_similarity,
            "string_similarity": self.string_similarity,
            "combined_similarity": self.combined_similarity,
            "confidence": self.confidence.value,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class CanonicalConcept:
    """A canonical concept with its aliases."""

    id: str
    name: str  # Canonical name
    aliases: list[str] = field(default_factory=list)  # Alternative names
    merged_from: list[str] = field(default_factory=list)  # IDs of merged concepts

    def add_alias(self, alias: str) -> None:
        """Add an alias if not already present."""
        if alias.lower() != self.name.lower() and alias.lower() not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)


@dataclass
class EnrichedEdge:
    """An edge with semantic classification and provenance."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.5

    # Semantic properties
    is_semantic: bool = False  # True if represents semantic relationship
    is_universal: bool = False  # True if universally true (not context-dependent)

    # Provenance
    source_type: EdgeSourceType = EdgeSourceType.DOCUMENT
    provenance_doc_id: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "is_semantic": self.is_semantic,
            "is_universal": self.is_universal,
            "source_type": self.source_type.value,
            "provenance_doc_id": self.provenance_doc_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
