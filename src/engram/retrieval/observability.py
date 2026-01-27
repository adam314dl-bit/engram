"""Retrieval observability for tracing chunks through the pipeline.

v4.5: Provides detailed tracing of where chunks appear and disappear
at each stage of the retrieval pipeline.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from engram.config import settings


@dataclass
class ChunkTrace:
    """Trace for a single chunk through the pipeline."""

    memory_id: str
    content_preview: str  # First ~100 chars of content
    full_content: str = ""  # Full content for detailed inspection

    # Scores at each stage: step_name -> score
    stage_scores: dict[str, float] = field(default_factory=dict)

    # Ranks at each stage: step_name -> rank (1-indexed)
    stage_ranks: dict[str, int] = field(default_factory=dict)

    # Sources that found this chunk (V, B, G, P, etc.)
    sources: list[str] = field(default_factory=list)

    # Whether included in final result
    included: bool = False

    # Final rank if included (1-indexed)
    final_rank: int | None = None

    def appeared_in(self, step: str) -> bool:
        """Check if chunk appeared in a specific step."""
        return step in self.stage_scores

    def first_appearance(self) -> str | None:
        """Get the first step where this chunk appeared."""
        if not self.stage_scores:
            return None
        # Return the step with the lowest rank (appeared first)
        return min(self.stage_ranks.keys(), key=lambda k: self.stage_ranks.get(k, 9999))

    def dropped_at(self) -> str | None:
        """Get the step where this chunk was dropped (if not included)."""
        if self.included:
            return None
        if not self.stage_scores:
            return None
        # Return the last step where it appeared
        steps = list(self.stage_scores.keys())
        return steps[-1] if steps else None


@dataclass
class StepTrace:
    """Trace for a single pipeline step."""

    step_name: str
    duration_ms: float

    # Counts
    input_count: int
    output_count: int

    # Scores for chunks in this step: memory_id -> score
    chunk_scores: dict[str, float] = field(default_factory=dict)

    # Additional metadata (e.g., extracted concepts, thresholds used)
    metadata: dict = field(default_factory=dict)

    @property
    def dropped_count(self) -> int:
        """Number of chunks dropped in this step."""
        return self.input_count - self.output_count


@dataclass
class RetrievalTrace:
    """Complete trace for a retrieval operation."""

    trace_id: str
    query: str
    timestamp: datetime

    # Pipeline steps in order
    steps: list[StepTrace] = field(default_factory=list)

    # Per-chunk traces: memory_id -> ChunkTrace
    chunk_traces: dict[str, ChunkTrace] = field(default_factory=dict)

    # Query concepts extracted
    extracted_concepts: list[str] = field(default_factory=list)

    # Total pipeline duration
    total_duration_ms: float = 0.0

    def find_chunk_journey(self, memory_id: str) -> dict:
        """Track where a chunk appeared and disappeared.

        Returns:
            Dict with step info for each step where chunk was present
        """
        chunk = self.chunk_traces.get(memory_id)
        if not chunk:
            return {"error": f"Chunk {memory_id} not found in trace"}

        journey = []
        for step in self.steps:
            if memory_id in step.chunk_scores:
                journey.append({
                    "step": step.step_name,
                    "present": True,
                    "score": step.chunk_scores[memory_id],
                    "rank": chunk.stage_ranks.get(step.step_name),
                })
            else:
                journey.append({
                    "step": step.step_name,
                    "present": False,
                })

        return {
            "memory_id": memory_id,
            "content_preview": chunk.content_preview,
            "sources": chunk.sources,
            "included": chunk.included,
            "final_rank": chunk.final_rank,
            "journey": journey,
        }

    def summary(self) -> str:
        """Human-readable pipeline summary."""
        lines = [
            f"Retrieval Trace: {self.trace_id}",
            f"Query: {self.query[:80]}{'...' if len(self.query) > 80 else ''}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Total duration: {self.total_duration_ms:.1f}ms",
            f"Extracted concepts: {', '.join(self.extracted_concepts) or 'none'}",
            "",
            "Pipeline Steps:",
        ]

        for step in self.steps:
            lines.append(
                f"  {step.step_name}: "
                f"{step.input_count} -> {step.output_count} "
                f"({step.duration_ms:.1f}ms)"
            )
            if step.dropped_count > 0:
                lines.append(f"    dropped: {step.dropped_count}")

        # Count chunks by source
        source_counts: dict[str, int] = {}
        for chunk in self.chunk_traces.values():
            for source in chunk.sources:
                source_counts[source] = source_counts.get(source, 0) + 1

        lines.append("")
        lines.append("Sources:")
        for source, count in sorted(source_counts.items()):
            source_name = {
                "V": "Vector",
                "B": "BM25",
                "G": "Graph (spreading)",
                "P": "Path",
                "BE": "BM25 Expanded",
                "S": "Semantic rewrite",
                "H": "HyDE",
            }.get(source, source)
            lines.append(f"  {source_name}: {count}")

        # Final stats
        included_count = sum(1 for c in self.chunk_traces.values() if c.included)
        lines.append("")
        lines.append(f"Final: {included_count} chunks included")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "extracted_concepts": self.extracted_concepts,
            "steps": [
                {
                    "step_name": s.step_name,
                    "duration_ms": s.duration_ms,
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "chunk_traces": {
                mid: {
                    "memory_id": c.memory_id,
                    "content_preview": c.content_preview,
                    "full_content": c.full_content,
                    "stage_scores": c.stage_scores,
                    "stage_ranks": c.stage_ranks,
                    "sources": c.sources,
                    "included": c.included,
                    "final_rank": c.final_rank,
                }
                for mid, c in self.chunk_traces.items()
            },
        }

    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, directory: str | None = None) -> Path:
        """Save trace to file.

        Args:
            directory: Directory to save to (default from settings)

        Returns:
            Path to saved file
        """
        directory = directory or settings.observability_trace_dir
        trace_dir = Path(directory)
        trace_dir.mkdir(parents=True, exist_ok=True)

        # Filename: trace_YYYYMMDD_HHMMSS_ID.json
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        short_id = self.trace_id[:8]
        filename = f"trace_{timestamp_str}_{short_id}.json"

        filepath = trace_dir / filename
        filepath.write_text(self.to_json())

        return filepath


def create_trace(query: str) -> RetrievalTrace:
    """Create a new retrieval trace."""
    return RetrievalTrace(
        trace_id=str(uuid.uuid4()),
        query=query,
        timestamp=datetime.utcnow(),
    )
