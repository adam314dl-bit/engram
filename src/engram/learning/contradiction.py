"""Contradiction detection and resolution for memories.

Uses LLM-based analysis with Russian prompts to detect contradictions
between memories and automatically resolve them when confidence gap
exceeds the configured threshold.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.config import settings
from engram.ingestion.llm_client import get_llm_client

if TYPE_CHECKING:
    from engram.models import SemanticMemory
    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# Russian prompt for contradiction detection
CONTRADICTION_DETECTION_PROMPT = """Проанализируй два факта и определи, противоречат ли они друг другу.

Факт 1: {fact1}
Факт 2: {fact2}

Оцени по следующим критериям:
1. Прямое противоречие: факты утверждают взаимоисключающие вещи
2. Частичное противоречие: факты частично несовместимы
3. Нет противоречия: факты могут сосуществовать

Верни JSON:
{{
    "contradiction_type": "direct" | "partial" | "none",
    "confidence": 0.0-1.0,
    "explanation": "краткое объяснение",
    "newer_is_correct": true | false | null
}}

Если contradiction_type == "none", то newer_is_correct должен быть null.
Если информация устарела (например, "X был Y" vs "X теперь Z"), newer_is_correct = true.
"""

# Russian prompt for resolution recommendation
RESOLUTION_PROMPT = """Два факта противоречат друг другу:

Старый факт (создан: {old_date}): {old_fact}
Новый факт (создан: {new_date}): {new_fact}

Тип противоречия: {contradiction_type}

Рекомендуй действие:
1. "keep_old" - сохранить старый факт, отклонить новый
2. "keep_new" - сохранить новый факт, пометить старый как superseded
3. "keep_both" - сохранить оба (если они относятся к разным контекстам)
4. "merge" - объединить в один факт

Верни JSON:
{{
    "action": "keep_old" | "keep_new" | "keep_both" | "merge",
    "confidence": 0.0-1.0,
    "merged_content": "объединённый факт, если action=merge, иначе null",
    "explanation": "краткое объяснение решения"
}}
"""


@dataclass
class ContradictionResult:
    """Result of contradiction detection between two memories."""

    memory1_id: str
    memory2_id: str
    contradiction_type: str  # "direct", "partial", "none"
    confidence: float
    explanation: str
    newer_is_correct: bool | None


@dataclass
class ResolutionResult:
    """Result of contradiction resolution."""

    action: str  # "keep_old", "keep_new", "keep_both", "merge"
    confidence: float
    merged_content: str | None
    explanation: str
    auto_applied: bool


async def detect_contradiction(
    memory1: "SemanticMemory",
    memory2: "SemanticMemory",
) -> ContradictionResult:
    """
    Detect if two memories contradict each other using LLM.

    Args:
        memory1: First memory
        memory2: Second memory

    Returns:
        ContradictionResult with detection details
    """
    client = get_llm_client()

    prompt = CONTRADICTION_DETECTION_PROMPT.format(
        fact1=memory1.content,
        fact2=memory2.content,
    )

    try:
        result = await client.generate_json(
            prompt=prompt,
            temperature=0.3,
            fallback={"contradiction_type": "none", "confidence": 0.0, "explanation": "", "newer_is_correct": None},
        )

        return ContradictionResult(
            memory1_id=memory1.id,
            memory2_id=memory2.id,
            contradiction_type=result.get("contradiction_type", "none"),
            confidence=float(result.get("confidence", 0.0)),
            explanation=result.get("explanation", ""),
            newer_is_correct=result.get("newer_is_correct"),
        )
    except Exception as e:
        logger.warning(f"Contradiction detection failed: {e}")
        return ContradictionResult(
            memory1_id=memory1.id,
            memory2_id=memory2.id,
            contradiction_type="none",
            confidence=0.0,
            explanation=f"Detection failed: {e}",
            newer_is_correct=None,
        )


async def resolve_contradiction(
    old_memory: "SemanticMemory",
    new_memory: "SemanticMemory",
    contradiction_type: str,
) -> ResolutionResult:
    """
    Get resolution recommendation for contradicting memories.

    Args:
        old_memory: Older memory
        new_memory: Newer memory
        contradiction_type: Type of contradiction detected

    Returns:
        ResolutionResult with recommended action
    """
    client = get_llm_client()

    prompt = RESOLUTION_PROMPT.format(
        old_date=old_memory.created_at.isoformat(),
        old_fact=old_memory.content,
        new_date=new_memory.created_at.isoformat(),
        new_fact=new_memory.content,
        contradiction_type=contradiction_type,
    )

    try:
        result = await client.generate_json(
            prompt=prompt,
            temperature=0.3,
            fallback={"action": "keep_both", "confidence": 0.0, "merged_content": None, "explanation": ""},
        )

        return ResolutionResult(
            action=result.get("action", "keep_both"),
            confidence=float(result.get("confidence", 0.0)),
            merged_content=result.get("merged_content"),
            explanation=result.get("explanation", ""),
            auto_applied=False,
        )
    except Exception as e:
        logger.warning(f"Resolution recommendation failed: {e}")
        return ResolutionResult(
            action="keep_both",
            confidence=0.0,
            merged_content=None,
            explanation=f"Resolution failed: {e}",
            auto_applied=False,
        )


async def check_and_resolve(
    db: "Neo4jClient",
    memory1: "SemanticMemory",
    memory2: "SemanticMemory",
    auto_resolve_gap: float | None = None,
) -> tuple[ContradictionResult, ResolutionResult | None]:
    """
    Check for contradiction and optionally auto-resolve.

    Auto-resolution occurs when:
    1. A direct or partial contradiction is detected
    2. The confidence gap between old and new exceeds auto_resolve_gap

    Args:
        db: Neo4j client
        memory1: First memory
        memory2: Second memory
        auto_resolve_gap: Confidence gap for auto-resolution
            (default: settings.contradiction_auto_resolve_gap)

    Returns:
        Tuple of (ContradictionResult, ResolutionResult or None)
    """
    auto_resolve_gap = auto_resolve_gap or settings.contradiction_auto_resolve_gap

    # Detect contradiction
    detection = await detect_contradiction(memory1, memory2)

    if detection.contradiction_type == "none":
        return detection, None

    # Determine which is older
    if memory1.created_at <= memory2.created_at:
        old_memory, new_memory = memory1, memory2
    else:
        old_memory, new_memory = memory2, memory1

    # Get resolution recommendation
    resolution = await resolve_contradiction(old_memory, new_memory, detection.contradiction_type)

    # Check if we should auto-apply
    confidence_gap = abs(new_memory.confidence - old_memory.confidence)
    should_auto_apply = (
        detection.contradiction_type == "direct"
        and resolution.confidence >= 0.7
        and (
            confidence_gap >= auto_resolve_gap
            or detection.newer_is_correct is True
        )
    )

    if should_auto_apply and resolution.action in ("keep_new", "keep_old"):
        # Auto-apply resolution
        if resolution.action == "keep_new":
            await _mark_superseded(db, old_memory.id, new_memory.id)
        else:
            await _mark_superseded(db, new_memory.id, old_memory.id)
        resolution.auto_applied = True
        logger.info(
            f"Auto-resolved contradiction: {resolution.action} "
            f"({old_memory.id} vs {new_memory.id})"
        )

    return detection, resolution


async def _mark_superseded(
    db: "Neo4jClient",
    superseded_id: str,
    supersedes_id: str,
) -> None:
    """
    Mark a memory as superseded by another.

    Args:
        db: Neo4j client
        superseded_id: ID of memory being superseded
        supersedes_id: ID of memory that supersedes it
    """
    await db.execute_query(
        """
        MATCH (old:SemanticMemory {id: $superseded_id})
        MATCH (new:SemanticMemory {id: $supersedes_id})
        SET old.status = 'superseded',
            old.superseded_by = $supersedes_id,
            old.superseded_at = datetime()
        MERGE (new)-[:SUPERSEDES]->(old)
        """,
        superseded_id=superseded_id,
        supersedes_id=supersedes_id,
    )


async def find_potential_contradictions(
    db: "Neo4jClient",
    memory: "SemanticMemory",
    similarity_threshold: float = 0.7,
    limit: int = 10,
) -> list["SemanticMemory"]:
    """
    Find memories that might contradict a given memory.

    Uses embedding similarity to find semantically related memories,
    which are more likely to contain contradictions.

    Args:
        db: Neo4j client
        memory: Memory to check for contradictions
        similarity_threshold: Minimum similarity to consider
        limit: Maximum memories to return

    Returns:
        List of potentially contradicting memories
    """
    from engram.models import SemanticMemory

    if not memory.embedding:
        logger.warning(f"Memory {memory.id} has no embedding, cannot find contradictions")
        return []

    # Find similar memories (excluding itself and already superseded)
    results = await db.execute_query(
        """
        CALL db.index.vector.queryNodes('memory_embedding', $k, $embedding)
        YIELD node, score
        WHERE node.id <> $memory_id
          AND node.status = 'active'
          AND score >= $threshold
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """,
        embedding=memory.embedding,
        memory_id=memory.id,
        threshold=similarity_threshold,
        limit=limit,
        k=limit * 2,  # Get extra to account for filtering
    )

    return [SemanticMemory.from_dict(dict(r["node"])) for r in results]


async def batch_check_contradictions(
    db: "Neo4jClient",
    new_memory: "SemanticMemory",
    auto_resolve: bool = True,
) -> list[tuple[ContradictionResult, ResolutionResult | None]]:
    """
    Check a new memory against existing memories for contradictions.

    This should be called after ingesting new memories.

    Args:
        db: Neo4j client
        new_memory: Newly created memory
        auto_resolve: Whether to auto-resolve contradictions

    Returns:
        List of (ContradictionResult, ResolutionResult) tuples
    """
    # Find potential contradictions
    candidates = await find_potential_contradictions(db, new_memory)

    results = []
    for candidate in candidates:
        if auto_resolve:
            detection, resolution = await check_and_resolve(db, new_memory, candidate)
        else:
            detection = await detect_contradiction(new_memory, candidate)
            resolution = None

        if detection.contradiction_type != "none":
            results.append((detection, resolution))

    return results
