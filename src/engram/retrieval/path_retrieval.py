"""Path-based retrieval for finding memories that bridge multiple query concepts.

v4.5: Finds memories that connect multiple query concepts through graph paths,
complementing spreading activation which spreads from each concept independently.
"""

import logging
from dataclasses import dataclass, field

from engram.config import settings
from engram.models import SemanticMemory
from engram.retrieval.hybrid_search import ScoredMemory
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class PathRetrievalResult:
    """Result from path-based retrieval."""

    # Paths found between concept pairs
    paths: list[dict] = field(default_factory=list)

    # Bridge concepts connecting multiple query concepts
    bridge_concepts: list[dict] = field(default_factory=list)

    # Memories linked to multiple query concepts (highest value)
    shared_memories: list[ScoredMemory] = field(default_factory=list)

    # Memories from intermediate path concepts
    path_memories: list[ScoredMemory] = field(default_factory=list)

    @property
    def all_memories(self) -> list[ScoredMemory]:
        """Get all memories from path retrieval (shared + path)."""
        return self.shared_memories + self.path_memories

    @property
    def memory_count(self) -> int:
        """Total number of memories found."""
        return len(self.shared_memories) + len(self.path_memories)


class PathBasedRetriever:
    """
    Path-based retrieval for finding memories that bridge query concepts.

    Strategies:
    1. Shared memories: Find memories linked to 2+ query concepts (highest value)
    2. Shortest paths: Find paths between concept pairs, get intermediate memories
    3. Bridge concepts: Find concepts connected to multiple query concepts

    These complement spreading activation by finding cross-cutting memories
    that wouldn't be found by spreading from each concept independently.
    """

    def __init__(self, db: Neo4jClient) -> None:
        self.db = db

    async def retrieve(
        self,
        concept_ids: list[str],
        max_path_length: int | None = None,
        max_hops: int | None = None,
        min_links: int | None = None,
        memory_limit: int | None = None,
    ) -> PathRetrievalResult:
        """
        Main entry point - combines all path-based retrieval strategies.

        Args:
            concept_ids: Query concept IDs to find paths between
            max_path_length: Max path length (default from settings)
            max_hops: Max hops for bridge detection (default from settings)
            min_links: Min concept links for shared memories (default from settings)
            memory_limit: Max memories to return (default from settings)

        Returns:
            PathRetrievalResult with paths, bridges, and scored memories
        """
        # Need at least 2 concepts for path-based retrieval
        if len(concept_ids) < 2:
            logger.debug("Path retrieval skipped: need at least 2 concepts")
            return PathRetrievalResult()

        # Apply defaults from settings
        max_path_length = max_path_length or settings.path_max_length
        max_hops = max_hops or settings.path_bridge_max_hops
        min_links = min_links or settings.path_shared_min_links
        memory_limit = memory_limit or settings.path_memory_limit

        logger.debug(
            f"Path retrieval for {len(concept_ids)} concepts "
            f"(max_length={max_path_length}, min_links={min_links})"
        )

        # 1. Find shared memories (highest value - directly linked to multiple concepts)
        shared_results = await self.db.find_shared_memories(
            concept_ids=concept_ids,
            min_links=min_links,
            limit=memory_limit,
        )
        shared_memories = self._score_shared_memories(shared_results, concept_ids)
        logger.debug(f"Found {len(shared_memories)} shared memories")

        # 2. Find shortest paths between concept pairs
        paths = await self.db.find_shortest_paths(
            concept_ids=concept_ids,
            max_length=max_path_length,
        )
        logger.debug(f"Found {len(paths)} paths between concepts")

        # 3. Get memories from intermediate path concepts
        intermediate_ids = self._extract_intermediate_ids(paths, concept_ids)
        path_mems: list[SemanticMemory] = []
        if intermediate_ids:
            # Exclude already-found shared memory IDs
            exclude_ids = [sm.memory.id for sm in shared_memories]
            path_mems = await self.db.get_memories_for_path_concepts(
                path_concept_ids=intermediate_ids,
                exclude_ids=exclude_ids,
                limit=memory_limit,
            )
        path_memories = self._score_path_memories(path_mems, intermediate_ids)
        logger.debug(f"Found {len(path_memories)} path memories")

        # 4. Find bridge concepts (for debugging/analysis)
        bridges = await self.db.find_bridge_concepts(
            concept_ids=concept_ids,
            max_hops=max_hops,
            limit=20,
        )
        logger.debug(f"Found {len(bridges)} bridge concepts")

        return PathRetrievalResult(
            paths=paths,
            bridge_concepts=bridges,
            shared_memories=shared_memories,
            path_memories=path_memories,
        )

    def _extract_intermediate_ids(
        self,
        paths: list[dict],
        query_concept_ids: list[str],
    ) -> list[str]:
        """Extract intermediate concept IDs from paths (excluding query concepts)."""
        intermediate: set[str] = set()
        query_set = set(query_concept_ids)

        for path in paths:
            path_ids = path.get("path_ids", [])
            for concept_id in path_ids:
                if concept_id not in query_set:
                    intermediate.add(concept_id)

        return list(intermediate)

    def _score_shared_memories(
        self,
        results: list[tuple[SemanticMemory, list[str], int]],
        concept_ids: list[str],
    ) -> list[ScoredMemory]:
        """Score shared memories based on how many query concepts they link.

        Higher score for memories linking more concepts.
        """
        scored: list[ScoredMemory] = []
        num_concepts = len(concept_ids)

        for memory, linked, link_count in results:
            # Score based on fraction of query concepts linked
            link_ratio = link_count / num_concepts if num_concepts > 0 else 0

            # Boost for linking more concepts (quadratic boost)
            link_boost = 1.0 + (link_count - 1) * 0.5

            # Combine with importance
            importance_score = memory.importance / 10.0

            # Final score: link_ratio weighted heavily + importance
            score = (link_ratio * 0.6 + importance_score * 0.4) * link_boost

            scored.append(ScoredMemory(
                memory=memory,
                score=score,
                sources=["P"],  # Path-based
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def _score_path_memories(
        self,
        memories: list[SemanticMemory],
        intermediate_ids: list[str],
    ) -> list[ScoredMemory]:
        """Score memories from intermediate path concepts.

        Lower score than shared memories since they're one hop away.
        """
        scored: list[ScoredMemory] = []
        intermediate_set = set(intermediate_ids)

        for memory in memories:
            # Count how many intermediate concepts this memory links to
            memory_concept_ids = set(memory.concept_ids) if memory.concept_ids else set()
            overlap = len(memory_concept_ids & intermediate_set)

            # Base score from importance
            importance_score = memory.importance / 10.0

            # Boost for linking multiple path concepts
            overlap_boost = 1.0 + (overlap - 1) * 0.3 if overlap > 0 else 0.5

            # Final score (lower than shared memories)
            score = importance_score * overlap_boost * 0.7  # 0.7 discount vs shared

            scored.append(ScoredMemory(
                memory=memory,
                score=score,
                sources=["P"],  # Path-based
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored
