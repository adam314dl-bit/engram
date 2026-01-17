"""Hebbian learning for concept link strengthening.

"Neurons that fire together wire together" - strengthen connections
between concepts that are activated together during successful reasoning.
"""

import logging
from datetime import datetime
from itertools import combinations

from engram.models import ConceptRelation
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Hebbian constants
MAX_WEIGHT = 1.0
MIN_WEIGHT = 0.1
STRENGTHEN_AMOUNT = 0.05
WEAKEN_AMOUNT = 0.02


async def strengthen_concept_links(
    db: Neo4jClient,
    concept_ids: list[str],
    boost: float = STRENGTHEN_AMOUNT,
) -> int:
    """
    Strengthen connections between co-activated concepts.

    When concepts are activated together during successful reasoning,
    we strengthen the edges between them (Hebbian learning).

    Args:
        db: Database client
        concept_ids: Concepts that were activated together
        boost: Amount to increase edge weights

    Returns:
        Number of edges strengthened
    """
    if len(concept_ids) < 2:
        return 0

    count = 0

    # Strengthen edges between all pairs of activated concepts
    for source_id, target_id in combinations(concept_ids, 2):
        updated = await _strengthen_edge(db, source_id, target_id, boost)
        if updated:
            count += 1
        # Also strengthen reverse direction
        updated = await _strengthen_edge(db, target_id, source_id, boost)
        if updated:
            count += 1

    logger.info(
        f"Strengthened {count} concept links between {len(concept_ids)} concepts"
    )
    return count


async def _strengthen_edge(
    db: Neo4jClient,
    source_id: str,
    target_id: str,
    boost: float,
) -> bool:
    """Strengthen a single edge between two concepts."""
    query = """
    MATCH (source:Concept {id: $source_id})-[r:RELATED_TO]->(target:Concept {id: $target_id})
    SET r.weight = CASE
        WHEN r.weight + $boost > $max_weight THEN $max_weight
        ELSE r.weight + $boost
    END,
    r.co_occurrence_count = coalesce(r.co_occurrence_count, 0) + 1,
    r.last_used = datetime()
    RETURN r.weight as new_weight
    """

    try:
        results = await db.execute_query(
            query,
            source_id=source_id,
            target_id=target_id,
            boost=boost,
            max_weight=MAX_WEIGHT,
        )
        if results:
            new_weight = results[0]["new_weight"]
            logger.debug(
                f"Strengthened edge {source_id} -> {target_id}: weight={new_weight:.3f}"
            )
            return True
    except Exception as e:
        logger.debug(f"No edge to strengthen: {source_id} -> {target_id}: {e}")

    return False


async def weaken_concept_links(
    db: Neo4jClient,
    concept_ids: list[str],
    decay: float = WEAKEN_AMOUNT,
) -> int:
    """
    Weaken connections between concepts (used after failures).

    Args:
        db: Database client
        concept_ids: Concepts that were activated but led to failure
        decay: Amount to decrease edge weights

    Returns:
        Number of edges weakened
    """
    if len(concept_ids) < 2:
        return 0

    count = 0

    for source_id, target_id in combinations(concept_ids, 2):
        updated = await _weaken_edge(db, source_id, target_id, decay)
        if updated:
            count += 1
        updated = await _weaken_edge(db, target_id, source_id, decay)
        if updated:
            count += 1

    logger.info(f"Weakened {count} concept links")
    return count


async def _weaken_edge(
    db: Neo4jClient,
    source_id: str,
    target_id: str,
    decay: float,
) -> bool:
    """Weaken a single edge between two concepts."""
    query = """
    MATCH (source:Concept {id: $source_id})-[r:RELATED_TO]->(target:Concept {id: $target_id})
    SET r.weight = CASE
        WHEN r.weight - $decay < $min_weight THEN $min_weight
        ELSE r.weight - $decay
    END,
    r.last_used = datetime()
    RETURN r.weight as new_weight
    """

    try:
        results = await db.execute_query(
            query,
            source_id=source_id,
            target_id=target_id,
            decay=decay,
            min_weight=MIN_WEIGHT,
        )
        if results:
            return True
    except Exception as e:
        logger.debug(f"No edge to weaken: {source_id} -> {target_id}: {e}")

    return False


async def update_concept_activation(
    db: Neo4jClient,
    concept_ids: list[str],
) -> int:
    """
    Update activation counts for concepts.

    Args:
        db: Database client
        concept_ids: Concepts that were activated

    Returns:
        Number of concepts updated
    """
    count = 0
    for concept_id in concept_ids:
        try:
            await db.update_concept_activation(concept_id)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to update activation for {concept_id}: {e}")

    return count


async def create_or_strengthen_link(
    db: Neo4jClient,
    source_id: str,
    target_id: str,
    relation_type: str = "related_to",
    initial_weight: float = 0.5,
    boost: float = STRENGTHEN_AMOUNT,
) -> bool:
    """
    Create a new link or strengthen existing one.

    Used when concepts co-occur but may not have an existing edge.

    Args:
        db: Database client
        source_id: Source concept ID
        target_id: Target concept ID
        relation_type: Type of relationship
        initial_weight: Weight for new edges
        boost: Amount to strengthen existing edges

    Returns:
        True if edge was created or updated
    """
    # Try to strengthen existing edge first
    if await _strengthen_edge(db, source_id, target_id, boost):
        return True

    # Create new edge if none exists
    query = """
    MATCH (source:Concept {id: $source_id})
    MATCH (target:Concept {id: $target_id})
    MERGE (source)-[r:RELATED_TO]->(target)
    ON CREATE SET
        r.type = $relation_type,
        r.weight = $initial_weight,
        r.co_occurrence_count = 1,
        r.last_used = datetime()
    ON MATCH SET
        r.weight = CASE
            WHEN r.weight + $boost > $max_weight THEN $max_weight
            ELSE r.weight + $boost
        END,
        r.co_occurrence_count = coalesce(r.co_occurrence_count, 0) + 1,
        r.last_used = datetime()
    RETURN r
    """

    try:
        results = await db.execute_query(
            query,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            initial_weight=initial_weight,
            boost=boost,
            max_weight=MAX_WEIGHT,
        )
        if results:
            logger.debug(f"Created/strengthened link {source_id} -> {target_id}")
            return True
    except Exception as e:
        logger.warning(f"Failed to create link {source_id} -> {target_id}: {e}")

    return False
