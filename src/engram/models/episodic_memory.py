"""Episodic memory model - reasoning traces with outcomes."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

FeedbackType = Literal["positive", "negative", "correction"]


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
class EpisodicMemory:
    """
    Represents a past reasoning episode: query, behavior, answer, and outcome.

    Stores abstracted behavior patterns for metacognitive reuse.
    Successful episodes can crystallize into semantic memories.
    """

    id: str

    # The episode context
    query: str  # Original user query
    concepts_activated: list[str] = field(default_factory=list)  # Concept IDs that fired
    memories_used: list[str] = field(default_factory=list)  # SemanticMemory IDs used

    # Reasoning pattern (metacognitive reuse format)
    behavior_name: str = ""  # "check_disk_usage", "explain_concept"
    behavior_instruction: str = ""  # One-line reusable pattern
    domain: str = "general"  # "docker", "kubernetes", etc.

    # Outcome
    answer_summary: str = ""  # Brief summary of response given
    feedback: FeedbackType | None = None
    correction_text: str | None = None  # If user provided correction

    # Statistics
    importance: float = 5.0  # 1-10
    repetition_count: int = 1  # Times similar query succeeded
    success_count: int = 0
    failure_count: int = 0

    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)

    # Consolidation tracking
    consolidated: bool = False  # Has become semantic memory?
    consolidated_memory_id: str | None = None

    # Embedding on behavior_instruction for finding similar strategies
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.id,
            "query": self.query,
            "concepts_activated": self.concepts_activated,
            "memories_used": self.memories_used,
            "behavior_name": self.behavior_name,
            "behavior_instruction": self.behavior_instruction,
            "domain": self.domain,
            "answer_summary": self.answer_summary,
            "feedback": self.feedback,
            "correction_text": self.correction_text,
            "importance": self.importance,
            "repetition_count": self.repetition_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "consolidated": self.consolidated,
            "consolidated_memory_id": self.consolidated_memory_id,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemory":
        """Create from dictionary (Neo4j record)."""
        return cls(
            id=data["id"],
            query=data["query"],
            concepts_activated=data.get("concepts_activated", []),
            memories_used=data.get("memories_used", []),
            behavior_name=data.get("behavior_name", ""),
            behavior_instruction=data.get("behavior_instruction", ""),
            domain=data.get("domain", "general"),
            answer_summary=data.get("answer_summary", ""),
            feedback=data.get("feedback"),
            correction_text=data.get("correction_text"),
            importance=data.get("importance", 5.0),
            repetition_count=data.get("repetition_count", 1),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            created_at=parse_datetime(data.get("created_at")) or datetime.utcnow(),
            last_used=parse_datetime(data.get("last_used")) or datetime.utcnow(),
            consolidated=data.get("consolidated", False),
            consolidated_memory_id=data.get("consolidated_memory_id"),
            embedding=data.get("embedding"),
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def is_successful(self) -> bool:
        """Check if episode has positive outcome."""
        return self.success_count > self.failure_count
