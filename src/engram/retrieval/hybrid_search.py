"""Hybrid search combining BM25, vector search, and graph traversal.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple sources.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from engram.config import settings
from engram.models import EpisodicMemory, SemanticMemory
from engram.retrieval.embeddings import cosine_similarity
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """A memory with its retrieval score and source info."""

    memory: SemanticMemory
    score: float
    sources: list[str] = field(default_factory=list)  # Which retrieval methods found it


@dataclass
class ScoredEpisode:
    """An episode with its retrieval score."""

    episode: EpisodicMemory
    score: float


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> dict[str, float]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each list where item appears.

    Args:
        ranked_lists: List of [(item_id, score)] lists, each sorted by score desc
        k: Constant to prevent high scores for top-ranked items (default 60)

    Returns:
        Dictionary of item_id -> fused_score
    """
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked_list):
            rrf_score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
            fused_scores[item_id] = fused_scores.get(item_id, 0) + rrf_score

    return fused_scores


def hours_since(dt: datetime | None) -> float:
    """Calculate hours since a datetime."""
    if dt is None:
        return 1000.0  # Large value for never-accessed items
    now = datetime.utcnow()
    # Handle timezone-aware datetimes from Neo4j
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    delta = now - dt
    return delta.total_seconds() / 3600.0


class HybridSearch:
    """
    Hybrid search combining multiple retrieval signals.

    Combines:
    1. Graph-based retrieval (from spreading activation)
    2. Vector similarity search
    3. BM25 full-text search

    Then reranks using: recency + importance + relevance
    """

    def __init__(
        self,
        db: Neo4jClient,
        vector_k: int | None = None,
        bm25_k: int | None = None,
        final_k: int | None = None,
        recency_decay: float = 0.995,
    ) -> None:
        self.db = db
        self.vector_k = vector_k or settings.retrieval_vector_k
        self.bm25_k = bm25_k or settings.retrieval_bm25_k
        self.final_k = final_k or settings.retrieval_top_k
        self.recency_decay = recency_decay

    async def search_memories(
        self,
        query: str,
        query_embedding: list[float],
        graph_memories: list[SemanticMemory] | None = None,
        graph_memory_scores: dict[str, float] | None = None,
        force_include_ids: list[str] | None = None,
        force_exclude_ids: set[str] | None = None,
    ) -> list[ScoredMemory]:
        """
        Search for relevant memories using hybrid approach.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            graph_memories: Memories from spreading activation (optional)
            graph_memory_scores: Scores from graph traversal (optional)
            force_include_ids: Memory IDs to force include (with "forced" source)
            force_exclude_ids: Memory IDs to exclude from results

        Returns:
            List of ScoredMemory sorted by final score
        """
        force_include_ids = force_include_ids or []
        force_exclude_ids = force_exclude_ids or set()
        # 1. Vector search
        vector_results = await self.db.vector_search_memories(
            embedding=query_embedding, k=self.vector_k
        )
        vector_ranked = [(m.id, score) for m, score in vector_results]

        # 2. BM25 full-text search
        bm25_results = await self.db.fulltext_search_memories(
            query_text=query, k=self.bm25_k
        )
        bm25_ranked = [(m.id, score) for m, score in bm25_results]

        # 3. Graph-based results (if provided)
        graph_ranked: list[tuple[str, float]] = []
        if graph_memories and graph_memory_scores:
            graph_ranked = [
                (m.id, graph_memory_scores.get(m.id, 0))
                for m in graph_memories
            ]
            graph_ranked.sort(key=lambda x: x[1], reverse=True)

        # 4. Force-included memories
        forced_memories: list[SemanticMemory] = []
        forced_ranked: list[tuple[str, float]] = []
        if force_include_ids:
            for mem_id in force_include_ids:
                mem = await self.db.get_semantic_memory(mem_id)
                if mem:
                    forced_memories.append(mem)
                    # Give forced memories a high score to ensure inclusion
                    forced_ranked.append((mem.id, 1.0))

        # Combine using RRF
        ranked_lists = [r for r in [vector_ranked, bm25_ranked, graph_ranked, forced_ranked] if r]
        fused_scores = reciprocal_rank_fusion(ranked_lists)

        # Build memory lookup
        all_memories: dict[str, SemanticMemory] = {}
        memory_sources: dict[str, list[str]] = {}

        for m, _ in vector_results:
            if m.id not in force_exclude_ids:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("vector")

        for m, _ in bm25_results:
            if m.id not in force_exclude_ids:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("bm25")

        if graph_memories:
            for m in graph_memories:
                if m.id not in force_exclude_ids:
                    all_memories[m.id] = m
                    memory_sources.setdefault(m.id, []).append("graph")

        # Add forced memories
        for m in forced_memories:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("forced")

        # Rerank with recency + importance + relevance
        scored_memories = self._rerank_memories(
            memories=all_memories,
            fused_scores=fused_scores,
            query_embedding=query_embedding,
            sources=memory_sources,
        )

        return scored_memories[: self.final_k]

    def _rerank_memories(
        self,
        memories: dict[str, SemanticMemory],
        fused_scores: dict[str, float],
        query_embedding: list[float],
        sources: dict[str, list[str]],
    ) -> list[ScoredMemory]:
        """
        Rerank memories using Generative Agents formula.

        Score = recency + importance + relevance

        Where:
        - recency = decay^hours_since_access
        - importance = importance_score / 10
        - relevance = cosine_similarity(query, memory)
        """
        scored: list[ScoredMemory] = []

        for memory_id, memory in memories.items():
            # Recency score (exponential decay)
            recency = self.recency_decay ** hours_since(memory.last_accessed)

            # Importance score (normalized to 0-1)
            importance = memory.importance / 10.0

            # Relevance score (cosine similarity)
            relevance = 0.5  # Default if no embedding
            if memory.embedding:
                relevance = cosine_similarity(query_embedding, memory.embedding)

            # RRF score (normalized)
            rrf_score = fused_scores.get(memory_id, 0)

            # Combined score
            # Weight RRF higher since it already combines multiple signals
            final_score = (recency * 0.2) + (importance * 0.2) + (relevance * 0.3) + (rrf_score * 10 * 0.3)

            scored.append(ScoredMemory(
                memory=memory,
                score=final_score,
                sources=sources.get(memory_id, []),
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    async def search_similar_episodes(
        self,
        query_embedding: list[float],
        k: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[ScoredEpisode]:
        """
        Find similar past episodes for reasoning templates.

        Args:
            query_embedding: Query embedding vector
            k: Number of episodes to return
            exclude_ids: Episode IDs to exclude

        Returns:
            List of ScoredEpisode sorted by similarity
        """
        exclude_ids = exclude_ids or []

        # Vector search on episode embeddings (behavior_instruction)
        results = await self.db.vector_search_episodes(
            embedding=query_embedding, k=k * 2  # Get more to filter
        )

        scored: list[ScoredEpisode] = []
        for episode, similarity in results:
            if episode.id in exclude_ids:
                continue

            # Boost successful episodes
            success_boost = 1.0
            if episode.success_count > 0:
                success_rate = episode.success_count / max(
                    episode.success_count + episode.failure_count, 1
                )
                success_boost = 1.0 + (success_rate * 0.5)

            # Recency boost
            recency = self.recency_decay ** hours_since(episode.last_used)

            final_score = similarity * success_boost * (0.8 + recency * 0.2)

            scored.append(ScoredEpisode(episode=episode, score=final_score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    async def find_alternative_episodes(
        self,
        concepts: list[str],
        exclude_behavior: str,
        k: int = 3,
    ) -> list[ScoredEpisode]:
        """
        Find successful episodes with different approaches.

        Used for re-reasoning when initial approach fails.

        Args:
            concepts: Concept IDs to search within
            exclude_behavior: Behavior name to exclude
            k: Number of alternatives to return

        Returns:
            List of alternative episodes
        """
        # Query for episodes that activated these concepts but used different behavior
        query = """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        WHERE c.id IN $concept_ids
          AND e.behavior_name <> $exclude_behavior
          AND e.success_count > e.failure_count
        WITH e, count(c) as concept_overlap
        ORDER BY concept_overlap DESC, e.success_count DESC
        LIMIT $limit
        RETURN e
        """
        results = await self.db.execute_query(
            query,
            concept_ids=concepts,
            exclude_behavior=exclude_behavior,
            limit=k,
        )

        episodes: list[ScoredEpisode] = []
        for record in results:
            episode = EpisodicMemory.from_dict(dict(record["e"]))
            # Score based on success rate
            success_rate = episode.success_count / max(
                episode.success_count + episode.failure_count, 1
            )
            episodes.append(ScoredEpisode(episode=episode, score=success_rate))

        return episodes


async def hybrid_search(
    db: Neo4jClient,
    query: str,
    query_embedding: list[float],
    graph_memories: list[SemanticMemory] | None = None,
    k: int = 10,
) -> list[ScoredMemory]:
    """Convenience function for hybrid search."""
    searcher = HybridSearch(db=db, final_k=k)
    return await searcher.search_memories(
        query=query,
        query_embedding=query_embedding,
        graph_memories=graph_memories,
    )
