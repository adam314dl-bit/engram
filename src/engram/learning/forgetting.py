"""ACT-R inspired memory forgetting and archival system.

Implements base-level activation from ACT-R cognitive architecture to model
memory decay and retrieval probability. Memories are never deleted, only
deprioritized or archived based on activation levels.

Memory Lifecycle:
- ACTIVE: Normal retrieval, high activation
- DEPRIORITIZED: Reduced retrieval weight, low activation
- ARCHIVED: Excluded from retrieval, very low activation (can be restored)
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from engram.config import settings

if TYPE_CHECKING:
    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Memory status constants (match MemoryStatus Literal in models)
STATUS_ACTIVE = "active"
STATUS_DEPRIORITIZED = "deprioritized"
STATUS_ARCHIVED = "archived"


@dataclass
class AccessRecord:
    """Record of a memory access event."""

    timestamp: datetime
    context: str | None = None  # Optional: what triggered the access


@dataclass
class ActivationInfo:
    """Base-level activation information for a memory."""

    memory_id: str
    base_level_activation: float
    status: str
    access_count: int
    last_accessed: datetime
    created_at: datetime


def compute_base_level_activation(
    access_times: list[datetime],
    current_time: datetime | None = None,
    decay_d: float | None = None,
) -> float:
    """
    Compute ACT-R base-level activation.

    B_i = ln(sum_{j=1}^{n} t_j^{-d})

    Where:
    - t_j is the time since the j-th access (in seconds)
    - d is the decay parameter (default 0.5)
    - n is the number of accesses

    Higher activation = more likely to be retrieved.
    Lower activation = more likely to be forgotten.

    Args:
        access_times: List of datetime when memory was accessed
        current_time: Current time (default: now)
        decay_d: Decay parameter d (default: settings.actr_decay_d)

    Returns:
        Base-level activation (log scale, typically -5 to +5)
    """
    if not access_times:
        return float("-inf")

    current_time = current_time or datetime.utcnow()
    decay_d = decay_d if decay_d is not None else settings.actr_decay_d

    # Compute sum of t^{-d} for each access
    activation_sum = 0.0
    for access_time in access_times:
        # Handle timezone-aware datetimes
        if access_time.tzinfo is not None:
            access_time = access_time.replace(tzinfo=None)

        # Time since access in seconds (minimum 1 second to avoid division by zero)
        time_diff = (current_time - access_time).total_seconds()
        time_diff = max(time_diff, 1.0)

        # Add t^{-d}
        activation_sum += time_diff ** (-decay_d)

    # Return log of sum
    if activation_sum <= 0:
        return float("-inf")

    return math.log(activation_sum)


def compute_retrieval_probability(
    base_level_activation: float,
    threshold_tau: float | None = None,
    noise_s: float = 0.4,
) -> float:
    """
    Compute probability of retrieval given base-level activation.

    P(retrieval) = 1 / (1 + exp((tau - B_i) / s))

    Where:
    - tau is the retrieval threshold
    - B_i is the base-level activation
    - s is the noise parameter

    Args:
        base_level_activation: Base-level activation B_i
        threshold_tau: Retrieval threshold (default: settings.actr_threshold_tau)
        noise_s: Noise parameter (default 0.4)

    Returns:
        Probability of retrieval (0-1)
    """
    threshold_tau = threshold_tau if threshold_tau is not None else settings.actr_threshold_tau

    if base_level_activation == float("-inf"):
        return 0.0

    # Logistic function
    exponent = (threshold_tau - base_level_activation) / noise_s
    exponent = max(min(exponent, 100), -100)  # Clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(exponent))


def determine_memory_status(
    base_level_activation: float,
    deprioritize_threshold: float | None = None,
    archive_threshold: float | None = None,
) -> str:
    """
    Determine memory status based on activation level.

    Args:
        base_level_activation: Current base-level activation
        deprioritize_threshold: Activation below which memory is deprioritized
        archive_threshold: Activation below which memory is archived

    Returns:
        Memory status: "active", "deprioritized", or "archived"
    """
    deprioritize_threshold = (
        deprioritize_threshold
        if deprioritize_threshold is not None
        else settings.forgetting_deprioritize_threshold
    )
    archive_threshold = (
        archive_threshold
        if archive_threshold is not None
        else settings.forgetting_archive_threshold
    )

    if base_level_activation < archive_threshold:
        return STATUS_ARCHIVED
    elif base_level_activation < deprioritize_threshold:
        return STATUS_DEPRIORITIZED
    else:
        return STATUS_ACTIVE


async def update_memory_activation(
    db: "Neo4jClient",
    memory_id: str,
    access_times: list[datetime] | None = None,
) -> ActivationInfo:
    """
    Update a memory's base-level activation and status.

    Args:
        db: Neo4j client
        memory_id: Memory ID to update
        access_times: List of access times (if None, fetched from DB)

    Returns:
        Updated activation info
    """
    # If access times not provided, fetch from database
    if access_times is None:
        result = await db.execute_query(
            """
            MATCH (m:SemanticMemory {id: $memory_id})
            RETURN m.created_at as created_at,
                   m.last_accessed as last_accessed,
                   m.access_count as access_count,
                   m.access_history as access_history
            """,
            memory_id=memory_id,
        )
        if not result:
            raise ValueError(f"Memory not found: {memory_id}")

        record = result[0]
        # Build access times from history or approximate from count
        access_history = record.get("access_history", [])
        if access_history:
            access_times = [datetime.fromisoformat(t) for t in access_history]
        else:
            # Approximate: distribute accesses evenly between creation and last access
            created_at = record["created_at"]
            last_accessed = record["last_accessed"]
            access_count = record.get("access_count", 1)

            if hasattr(created_at, "to_native"):
                created_at = created_at.to_native()
            if hasattr(last_accessed, "to_native"):
                last_accessed = last_accessed.to_native()

            # Create approximate access times
            access_times = [created_at]
            if access_count > 1 and last_accessed:
                time_span = (last_accessed - created_at).total_seconds()
                for i in range(1, access_count):
                    offset = time_span * (i / access_count)
                    access_times.append(
                        created_at + __import__("datetime").timedelta(seconds=offset)
                    )

    # Compute activation
    current_time = datetime.utcnow()
    activation = compute_base_level_activation(access_times, current_time)
    new_status = determine_memory_status(activation)

    # Update in database
    await db.execute_query(
        """
        MATCH (m:SemanticMemory {id: $memory_id})
        SET m.base_level_activation = $activation,
            m.status = $status,
            m.activation_updated_at = datetime()
        RETURN m.access_count as access_count,
               m.last_accessed as last_accessed,
               m.created_at as created_at
        """,
        memory_id=memory_id,
        activation=activation,
        status=new_status,
    )

    result = await db.execute_query(
        """
        MATCH (m:SemanticMemory {id: $memory_id})
        RETURN m.access_count as access_count,
               m.last_accessed as last_accessed,
               m.created_at as created_at
        """,
        memory_id=memory_id,
    )
    record = result[0]

    last_accessed = record["last_accessed"]
    created_at = record["created_at"]
    if hasattr(last_accessed, "to_native"):
        last_accessed = last_accessed.to_native()
    if hasattr(created_at, "to_native"):
        created_at = created_at.to_native()

    return ActivationInfo(
        memory_id=memory_id,
        base_level_activation=activation,
        status=new_status,
        access_count=record.get("access_count", 1),
        last_accessed=last_accessed,
        created_at=created_at,
    )


async def batch_update_activations(
    db: "Neo4jClient",
    memory_ids: list[str] | None = None,
    batch_size: int = 100,
) -> list[ActivationInfo]:
    """
    Batch update activations for multiple memories.

    Args:
        db: Neo4j client
        memory_ids: List of memory IDs (if None, update all active memories)
        batch_size: Number of memories to update per batch

    Returns:
        List of updated activation info
    """
    # Get memories to update
    if memory_ids is None:
        result = await db.execute_query(
            """
            MATCH (m:SemanticMemory)
            WHERE m.status = 'active' OR m.status = 'deprioritized'
            RETURN m.id as id
            """,
        )
        memory_ids = [r["id"] for r in result]

    results = []
    for i in range(0, len(memory_ids), batch_size):
        batch = memory_ids[i : i + batch_size]
        for memory_id in batch:
            try:
                info = await update_memory_activation(db, memory_id)
                results.append(info)
            except Exception as e:
                logger.warning(f"Failed to update activation for {memory_id}: {e}")

    return results


async def restore_memory(
    db: "Neo4jClient",
    memory_id: str,
) -> ActivationInfo:
    """
    Restore an archived memory to active status.

    This records a new access, boosting the activation level.

    Args:
        db: Neo4j client
        memory_id: Memory ID to restore

    Returns:
        Updated activation info
    """
    # Record new access
    await db.execute_query(
        """
        MATCH (m:SemanticMemory {id: $memory_id})
        SET m.last_accessed = datetime(),
            m.access_count = coalesce(m.access_count, 0) + 1,
            m.status = 'active'
        """,
        memory_id=memory_id,
    )

    # Recompute activation
    return await update_memory_activation(db, memory_id)


async def get_memories_by_status(
    db: "Neo4jClient",
    status: str,
    limit: int = 100,
) -> list[str]:
    """
    Get memory IDs by status.

    Args:
        db: Neo4j client
        status: Memory status to filter by
        limit: Maximum memories to return

    Returns:
        List of memory IDs
    """
    result = await db.execute_query(
        """
        MATCH (m:SemanticMemory {status: $status})
        RETURN m.id as id
        ORDER BY m.base_level_activation DESC
        LIMIT $limit
        """,
        status=status,
        limit=limit,
    )
    return [r["id"] for r in result]


async def run_forgetting_cycle(
    db: "Neo4jClient",
    batch_size: int = 500,
) -> dict[str, int]:
    """
    Run a forgetting cycle: update all memory activations and statuses.

    This should be run periodically (e.g., daily) to maintain memory health.

    Args:
        db: Neo4j client
        batch_size: Batch size for updates

    Returns:
        Statistics about status changes
    """
    logger.info("Starting forgetting cycle...")

    # Count statuses before
    before = await db.execute_query(
        """
        MATCH (m:SemanticMemory)
        RETURN m.status as status, count(*) as count
        """
    )
    before_counts = {r["status"]: r["count"] for r in before}

    # Update all activations
    results = await batch_update_activations(db, batch_size=batch_size)

    # Count statuses after
    after = await db.execute_query(
        """
        MATCH (m:SemanticMemory)
        RETURN m.status as status, count(*) as count
        """
    )
    after_counts = {r["status"]: r["count"] for r in after}

    stats = {
        "total_updated": len(results),
        "active_before": before_counts.get("active", 0),
        "active_after": after_counts.get("active", 0),
        "deprioritized_before": before_counts.get("deprioritized", 0),
        "deprioritized_after": after_counts.get("deprioritized", 0),
        "archived_before": before_counts.get("archived", 0),
        "archived_after": after_counts.get("archived", 0),
    }

    logger.info(f"Forgetting cycle complete: {stats}")
    return stats
