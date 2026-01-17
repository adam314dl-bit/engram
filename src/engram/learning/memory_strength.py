"""SM-2 algorithm for memory strength updates.

Based on the SuperMemo SM-2 spaced repetition algorithm.
Strength (easiness factor) ranges from 1.3 to 2.5.
"""

import logging
from datetime import datetime

from engram.models import SemanticMemory
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# SM-2 constants
MIN_EASINESS = 1.3
MAX_EASINESS = 2.5
DEFAULT_EASINESS = 2.5


async def update_memory_strength(
    db: Neo4jClient,
    memory_id: str,
    quality: int,
) -> SemanticMemory | None:
    """
    Update memory strength using SM-2 algorithm.

    Args:
        db: Database client
        memory_id: Memory to update
        quality: 0-5 rating (0=complete failure, 5=perfect recall)
            - 5: Perfect response
            - 4: Correct response after hesitation
            - 3: Correct response with serious difficulty
            - 2: Incorrect response but remembered upon seeing correct
            - 1: Incorrect response; correct seemed easy to recall
            - 0: Complete blackout

    Returns:
        Updated memory or None if not found
    """
    memory = await db.get_semantic_memory(memory_id)
    if not memory:
        logger.warning(f"Memory not found for strength update: {memory_id}")
        return None

    old_strength = memory.strength

    if quality >= 3:  # Successful recall
        memory.access_count += 1
        # SM-2 formula for easiness factor update
        memory.strength = max(
            MIN_EASINESS,
            memory.strength + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )
        memory.strength = min(MAX_EASINESS, memory.strength)
    else:  # Failed recall
        # Decrease easiness factor
        memory.strength = max(MIN_EASINESS, memory.strength - 0.2)

    memory.last_accessed = datetime.utcnow()

    await db.save_semantic_memory(memory)

    logger.debug(
        f"Updated memory {memory_id} strength: {old_strength:.2f} -> {memory.strength:.2f} "
        f"(quality={quality})"
    )

    return memory


async def strengthen_memory(
    db: Neo4jClient,
    memory_id: str,
    boost: float = 0.1,
) -> SemanticMemory | None:
    """
    Strengthen a memory (Hebbian-style reinforcement).

    Used when memory is successfully used in reasoning.

    Args:
        db: Database client
        memory_id: Memory to strengthen
        boost: Amount to increase strength (default 0.1)

    Returns:
        Updated memory or None if not found
    """
    memory = await db.get_semantic_memory(memory_id)
    if not memory:
        return None

    old_strength = memory.strength
    memory.strength = min(MAX_EASINESS, memory.strength + boost)
    memory.access_count += 1
    memory.last_accessed = datetime.utcnow()

    await db.save_semantic_memory(memory)

    logger.debug(
        f"Strengthened memory {memory_id}: {old_strength:.2f} -> {memory.strength:.2f}"
    )

    return memory


async def weaken_memory(
    db: Neo4jClient,
    memory_id: str,
    factor: float = 0.95,
) -> SemanticMemory | None:
    """
    Weaken a memory slightly (used after negative feedback).

    Args:
        db: Database client
        memory_id: Memory to weaken
        factor: Multiplicative factor (default 0.95 = 5% reduction)

    Returns:
        Updated memory or None if not found
    """
    memory = await db.get_semantic_memory(memory_id)
    if not memory:
        return None

    old_strength = memory.strength
    memory.strength = max(MIN_EASINESS, memory.strength * factor)
    memory.last_accessed = datetime.utcnow()

    await db.save_semantic_memory(memory)

    logger.debug(
        f"Weakened memory {memory_id}: {old_strength:.2f} -> {memory.strength:.2f}"
    )

    return memory


async def batch_strengthen_memories(
    db: Neo4jClient,
    memory_ids: list[str],
    boost: float = 0.1,
) -> int:
    """
    Strengthen multiple memories at once.

    Args:
        db: Database client
        memory_ids: Memories to strengthen
        boost: Amount to increase strength

    Returns:
        Number of memories successfully updated
    """
    count = 0
    for memory_id in memory_ids:
        result = await strengthen_memory(db, memory_id, boost)
        if result:
            count += 1

    logger.info(f"Strengthened {count}/{len(memory_ids)} memories")
    return count


async def batch_weaken_memories(
    db: Neo4jClient,
    memory_ids: list[str],
    factor: float = 0.95,
) -> int:
    """
    Weaken multiple memories at once.

    Args:
        db: Database client
        memory_ids: Memories to weaken
        factor: Multiplicative factor

    Returns:
        Number of memories successfully updated
    """
    count = 0
    for memory_id in memory_ids:
        result = await weaken_memory(db, memory_id, factor)
        if result:
            count += 1

    logger.info(f"Weakened {count}/{len(memory_ids)} memories")
    return count
