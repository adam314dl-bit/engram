"""Hybrid search combining BM25, vector search, and graph traversal.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple sources.
Includes MMR (Maximal Marginal Relevance) for diversity and dynamic top_k.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from engram.config import settings
from engram.models import EpisodicMemory, SemanticMemory
from engram.retrieval.embeddings import cosine_similarity
from engram.retrieval.fusion import rrf_scores_only, weighted_rrf
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# Russian patterns for query complexity classification
SIMPLE_QUERY_PATTERNS = [
    r"^(кто|что|где|когда)\s+",  # Simple factoid questions
    r"^как\s+называется",
    r"^какой\s+(номер|телефон|адрес|email)",
    r"^(дата|время|место)\s+",
    r"контакт(ы)?\s+",
]

COMPLEX_QUERY_PATTERNS = [
    r"(сравни|сравнение|отличия|разница)",
    r"(почему|зачем|для\s+чего)",
    r"(как\s+работает|принцип\s+работы)",
    r"(преимущества|недостатки|плюсы|минусы)",
    r"(объясни|расскажи\s+подробно)",
    r"(все|полный\s+список|перечисли)",
]


def classify_query_complexity(query: str) -> tuple[str, int]:
    """
    Classify query complexity and recommend top_k.

    Args:
        query: User query text

    Returns:
        Tuple of (complexity_type, recommended_k)
        - "simple": factoid queries, k=4
        - "moderate": standard queries, k=6
        - "complex": analytical queries, k=8
    """
    query_lower = query.lower().strip()

    # Check for simple patterns
    for pattern in SIMPLE_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return ("simple", settings.topk_simple)

    # Check for complex patterns
    for pattern in COMPLEX_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return ("complex", settings.topk_complex)

    # Default: moderate complexity
    return ("moderate", settings.topk_moderate)


def mmr_rerank(
    query_embedding: list[float],
    candidates: list[tuple[str, list[float], float]],
    k: int,
    lambda_mult: float = 0.5,
    fetch_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance reranking for diversity.

    MMR balances relevance to query with diversity among selected results.
    Score = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected))

    Args:
        query_embedding: Query embedding vector
        candidates: List of (id, embedding, original_score) tuples
        k: Number of results to return
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        fetch_k: Number of candidates to consider (should be > k)

    Returns:
        List of (id, mmr_score) tuples sorted by MMR score
    """
    if not candidates:
        return []

    # Limit candidates to fetch_k
    candidates = candidates[:fetch_k]

    # Convert to numpy for efficiency
    query_vec = np.array(query_embedding)
    doc_vecs = np.array([c[1] for c in candidates])
    doc_ids = [c[0] for c in candidates]
    original_scores = [c[2] for c in candidates]

    # Normalize vectors for cosine similarity
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

    # Compute query-document similarities
    query_sims = np.dot(doc_norms, query_norm)

    # Compute document-document similarity matrix
    doc_sims = np.dot(doc_norms, doc_norms.T)

    # MMR selection
    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        if not remaining_indices:
            break

        mmr_scores = []
        for idx in remaining_indices:
            # Relevance to query
            relevance = query_sims[idx]

            # Maximum similarity to already selected documents
            if selected_indices:
                max_sim_to_selected = max(doc_sims[idx, s] for s in selected_indices)
            else:
                max_sim_to_selected = 0.0

            # MMR score
            mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
            mmr_scores.append((idx, mmr))

        # Select document with highest MMR score
        best_idx, best_mmr = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # Return selected documents with their MMR scores
    results = []
    for i, idx in enumerate(selected_indices):
        # Combine MMR position with original score for final ranking
        position_score = 1.0 - (i / k) if k > 0 else 1.0
        combined_score = (original_scores[idx] + position_score) / 2
        results.append((doc_ids[idx], combined_score))

    return results


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
        query_embedding: list[float] | None = None,
        graph_memories: list[SemanticMemory] | None = None,
        graph_memory_scores: dict[str, float] | None = None,
        path_memories: list[SemanticMemory] | None = None,
        path_memory_scores: dict[str, float] | None = None,
        transliteration_variants: list[str] | None = None,
        use_dynamic_k: bool = True,
    ) -> list[ScoredMemory]:
        """
        Search for relevant memories using hybrid approach.

        Args:
            query: User query text
            query_embedding: Query embedding vector (optional in bm25_graph mode)
            graph_memories: Memories from spreading activation (optional)
            graph_memory_scores: Scores from graph traversal (optional)
            path_memories: Memories from path-based retrieval (v4.5)
            path_memory_scores: Scores from path retrieval (v4.5)
            transliteration_variants: Query variants (e.g., ["роутинг", "routing"])
            use_dynamic_k: Whether to use dynamic top_k based on query complexity

        Returns:
            List of ScoredMemory sorted by final score
        """
        # Determine final_k based on query complexity
        final_k = self.final_k
        if use_dynamic_k and settings.dynamic_topk_enabled:
            complexity, recommended_k = classify_query_complexity(query)
            final_k = recommended_k
            logger.debug(f"Query complexity: {complexity}, using k={final_k}")

        # Check retrieval mode
        use_vector = settings.retrieval_mode != "bm25_graph" and query_embedding is not None

        # 1. Vector search (skip in bm25_graph mode)
        vector_results: list[tuple[SemanticMemory, float]] = []
        if use_vector:
            vector_results = await self.db.vector_search_memories(
                embedding=query_embedding, k=self.vector_k
            )
        vector_ranked = [(m.id, score) for m, score in vector_results]

        # 2. BM25 full-text search (always)
        # Use transliteration variants if provided (v4.5.1)
        bm25_queries = transliteration_variants if transliteration_variants else [query]
        bm25_results: list[tuple[SemanticMemory, float]] = []
        bm25_seen_ids: set[str] = set()

        for bm25_query in bm25_queries:
            variant_results = await self.db.fulltext_search_memories(
                query_text=bm25_query, k=self.bm25_k
            )
            for m, score in variant_results:
                if m.id not in bm25_seen_ids:
                    bm25_results.append((m, score))
                    bm25_seen_ids.add(m.id)
                else:
                    # Update score if this variant found it with higher score
                    for i, (existing_m, existing_score) in enumerate(bm25_results):
                        if existing_m.id == m.id and score > existing_score:
                            bm25_results[i] = (m, score)
                            break

        # Sort by score after merging
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_ranked = [(m.id, score) for m, score in bm25_results]

        # 3. Graph-based results (if provided)
        graph_ranked: list[tuple[str, float]] = []
        if graph_memories and graph_memory_scores:
            graph_ranked = [
                (m.id, graph_memory_scores.get(m.id, 0))
                for m in graph_memories
            ]
            graph_ranked.sort(key=lambda x: x[1], reverse=True)

        # 4. Path-based results (v4.5)
        path_ranked: list[tuple[str, float]] = []
        if path_memories and path_memory_scores:
            path_ranked = [
                (m.id, path_memory_scores.get(m.id, 0))
                for m in path_memories
            ]
            path_ranked.sort(key=lambda x: x[1], reverse=True)

        # Combine using weighted RRF (v4.6)
        # Order: vector, bm25, graph (spreading), path (weights applied in same order)
        ranked_lists: list[list[tuple[str, float]]] = []
        weights: list[float] = []

        if vector_ranked:
            ranked_lists.append(vector_ranked)
            weights.append(settings.rrf_vector_weight)
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)
            weights.append(settings.rrf_bm25_weight)
        if graph_ranked:
            ranked_lists.append(graph_ranked)
            weights.append(settings.rrf_graph_weight)
        if path_ranked:
            ranked_lists.append(path_ranked)
            weights.append(settings.rrf_path_weight)

        fused_scores = weighted_rrf(ranked_lists, weights, k=settings.rrf_k)

        # Build memory lookup
        all_memories: dict[str, SemanticMemory] = {}
        memory_sources: dict[str, list[str]] = {}

        for m, _ in vector_results:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("V")  # Vector

        for m, _ in bm25_results:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("B")  # BM25

        if graph_memories:
            for m in graph_memories:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("G")  # Graph

        if path_memories:
            for m in path_memories:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("P")  # Path

        # Rerank with recency + importance + relevance
        scored_memories = self._rerank_memories(
            memories=all_memories,
            fused_scores=fused_scores,
            query_embedding=query_embedding,
            sources=memory_sources,
        )

        # Apply cross-encoder reranking if enabled
        if settings.reranker_enabled:
            scored_memories = self._apply_reranker(
                query=query,
                scored_memories=scored_memories,
                candidates_k=settings.reranker_candidates,
            )

        # Apply MMR for diversity if enabled (requires embeddings)
        if settings.mmr_enabled and use_vector and len(scored_memories) > final_k:
            scored_memories = self._apply_mmr(
                query_embedding=query_embedding,  # type: ignore
                scored_memories=scored_memories,
                k=final_k,
            )

        return scored_memories[:final_k]

    def _apply_reranker(
        self,
        query: str,
        scored_memories: list[ScoredMemory],
        candidates_k: int = 30,
    ) -> list[ScoredMemory]:
        """
        Apply cross-encoder reranking to top candidates.

        Args:
            query: Query text
            scored_memories: Pre-ranked memories
            candidates_k: Number of candidates to rerank

        Returns:
            Reranked list of ScoredMemory
        """
        from engram.retrieval.reranker import rerank_with_fallback

        # Only rerank top candidates
        to_rerank = scored_memories[:candidates_k]
        rest = scored_memories[candidates_k:]

        # Prepare candidates for reranker: (id, content, score)
        candidates = [
            (sm.memory.id, sm.memory.content, sm.score)
            for sm in to_rerank
        ]

        # Rerank
        reranked = rerank_with_fallback(query, candidates, top_k=candidates_k)

        # Build result with reranked scores
        memory_lookup = {sm.memory.id: sm for sm in to_rerank}
        result: list[ScoredMemory] = []

        for item in reranked:
            sm = memory_lookup.get(item.id)
            if sm:
                # Update score with reranker score (blend original and rerank)
                blended_score = (sm.score + item.rerank_score) / 2
                result.append(ScoredMemory(
                    memory=sm.memory,
                    score=blended_score,
                    sources=sm.sources,
                ))

        # Add back remaining memories
        result.extend(rest)
        return result

    def _apply_mmr(
        self,
        query_embedding: list[float],
        scored_memories: list[ScoredMemory],
        k: int,
    ) -> list[ScoredMemory]:
        """
        Apply MMR reranking for diversity.

        Args:
            query_embedding: Query embedding vector
            scored_memories: Pre-ranked memories
            k: Number of results to return

        Returns:
            MMR-reranked list of ScoredMemory
        """
        # Prepare candidates: (id, embedding, score)
        candidates = []
        for sm in scored_memories[:settings.mmr_fetch_k]:
            if sm.memory.embedding:
                candidates.append((sm.memory.id, sm.memory.embedding, sm.score))

        if not candidates:
            return scored_memories[:k]

        # Apply MMR
        mmr_results = mmr_rerank(
            query_embedding=query_embedding,
            candidates=candidates,
            k=k,
            lambda_mult=settings.mmr_lambda,
            fetch_k=settings.mmr_fetch_k,
        )

        # Build result maintaining original ScoredMemory objects
        memory_lookup = {sm.memory.id: sm for sm in scored_memories}
        result: list[ScoredMemory] = []

        for doc_id, mmr_score in mmr_results:
            sm = memory_lookup.get(doc_id)
            if sm:
                result.append(ScoredMemory(
                    memory=sm.memory,
                    score=mmr_score,
                    sources=sm.sources,
                ))

        return result

    def _rerank_memories(
        self,
        memories: dict[str, SemanticMemory],
        fused_scores: dict[str, float],
        query_embedding: list[float] | None,
        sources: dict[str, list[str]],
    ) -> list[ScoredMemory]:
        """
        Rerank memories using Generative Agents formula.

        Score = recency + importance + relevance

        Where:
        - recency = decay^hours_since_access
        - importance = importance_score / 10
        - relevance = cosine_similarity(query, memory) or 0.5 if no embedding
        """
        scored: list[ScoredMemory] = []

        for memory_id, memory in memories.items():
            # Recency score (exponential decay)
            recency = self.recency_decay ** hours_since(memory.last_accessed)

            # Importance score (normalized to 0-1)
            importance = memory.importance / 10.0

            # Relevance score (cosine similarity) - skip if no query embedding
            relevance = 0.5  # Default if no embedding
            if query_embedding and memory.embedding:
                relevance = cosine_similarity(query_embedding, memory.embedding)

            # RRF score (normalized)
            rrf_score = fused_scores.get(memory_id, 0)

            # Combined score
            # In bm25_graph mode (no embeddings), give more weight to RRF
            if query_embedding is None:
                # No relevance signal, boost RRF and importance
                final_score = (recency * 0.2) + (importance * 0.3) + (rrf_score * 10 * 0.5)
            else:
                # Standard hybrid with all signals
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


    async def search_memories_multi_query(
        self,
        original_query: str,
        original_embedding: list[float] | None = None,
        bm25_expanded: str | None = None,
        semantic_rewrite: str | None = None,
        semantic_rewrite_embedding: list[float] | None = None,
        hyde_document: str | None = None,
        hyde_embedding: list[float] | None = None,
        graph_memories: list[SemanticMemory] | None = None,
        graph_memory_scores: dict[str, float] | None = None,
        path_memories: list[SemanticMemory] | None = None,
        path_memory_scores: dict[str, float] | None = None,
        transliteration_variants: list[str] | None = None,
    ) -> list[ScoredMemory]:
        """
        Search using multiple query variants with RRF fusion.

        Retrieval strategy (in hybrid mode):
        - Original query → Full hybrid (BM25 + vector + graph + path)
        - BM25 expanded → BM25 only
        - Semantic rewrite → Vector only
        - HyDE document → Vector only (if present)

        In bm25_graph mode:
        - Original query → BM25 + graph + path only
        - BM25 expanded → BM25 only
        - Semantic rewrite and HyDE → Skipped

        All results fused with RRF, then reranked against ORIGINAL query.

        Args:
            original_query: Original user query
            original_embedding: Embedding of original query (optional in bm25_graph mode)
            bm25_expanded: BM25-expanded query (synonyms, lemmas)
            semantic_rewrite: Semantically rewritten query
            semantic_rewrite_embedding: Embedding of semantic rewrite
            hyde_document: Hypothetical document for HyDE
            hyde_embedding: Embedding of HyDE document
            graph_memories: Memories from spreading activation
            graph_memory_scores: Scores from graph traversal
            path_memories: Memories from path-based retrieval (v4.5)
            path_memory_scores: Scores from path retrieval (v4.5)
            transliteration_variants: Query variants (e.g., ["роутинг", "routing"])

        Returns:
            List of ScoredMemory fused and reranked
        """
        all_ranked_lists: list[list[tuple[str, float]]] = []
        all_weights: list[float] = []  # v4.6: Track weights for each ranked list
        all_memories: dict[str, SemanticMemory] = {}
        memory_sources: dict[str, list[str]] = {}

        # Check retrieval mode
        use_vector = settings.retrieval_mode != "bm25_graph" and original_embedding is not None

        # 1. Original query → Hybrid search (vector only in hybrid mode)
        if use_vector:
            orig_vector_results = await self.db.vector_search_memories(
                embedding=original_embedding, k=self.vector_k
            )
            # Vector ranked list
            orig_vector_ranked = [(m.id, score) for m, score in orig_vector_results]
            all_ranked_lists.append(orig_vector_ranked)
            all_weights.append(settings.rrf_vector_weight)
            for m, _ in orig_vector_results:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("V")  # Vector

        # BM25 always runs - use transliteration variants if provided (v4.5.1)
        bm25_queries = transliteration_variants if transliteration_variants else [original_query]
        orig_bm25_results: list[tuple[SemanticMemory, float]] = []
        bm25_seen_ids: set[str] = set()

        for bm25_query in bm25_queries:
            variant_results = await self.db.fulltext_search_memories(
                query_text=bm25_query, k=self.bm25_k
            )
            for m, score in variant_results:
                if m.id not in bm25_seen_ids:
                    orig_bm25_results.append((m, score))
                    bm25_seen_ids.add(m.id)
                else:
                    # Update score if this variant found it with higher score
                    for i, (existing_m, existing_score) in enumerate(orig_bm25_results):
                        if existing_m.id == m.id and score > existing_score:
                            orig_bm25_results[i] = (m, score)
                            break

        # Sort by score after merging
        orig_bm25_results.sort(key=lambda x: x[1], reverse=True)
        orig_bm25_ranked = [(m.id, score) for m, score in orig_bm25_results]
        all_ranked_lists.append(orig_bm25_ranked)
        all_weights.append(settings.rrf_bm25_weight)
        for m, _ in orig_bm25_results:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("B")  # BM25

        # Graph ranked list (always if provided)
        if graph_memories and graph_memory_scores:
            graph_ranked = [
                (m.id, graph_memory_scores.get(m.id, 0))
                for m in graph_memories
            ]
            graph_ranked.sort(key=lambda x: x[1], reverse=True)
            all_ranked_lists.append(graph_ranked)
            all_weights.append(settings.rrf_graph_weight)
            for m in graph_memories:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("G")  # Graph

        # Path ranked list (v4.5)
        if path_memories and path_memory_scores:
            path_ranked = [
                (m.id, path_memory_scores.get(m.id, 0))
                for m in path_memories
            ]
            path_ranked.sort(key=lambda x: x[1], reverse=True)
            all_ranked_lists.append(path_ranked)
            all_weights.append(settings.rrf_path_weight)
            for m in path_memories:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("P")  # Path

        # 2. BM25 expanded → BM25 only (uses BM25 weight)
        if bm25_expanded and bm25_expanded != original_query:
            bm25_exp_results = await self._retrieve_bm25_only(bm25_expanded)
            bm25_exp_ranked = [(m.id, score) for m, score in bm25_exp_results]
            all_ranked_lists.append(bm25_exp_ranked)
            all_weights.append(settings.rrf_bm25_weight)
            for m, _ in bm25_exp_results:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("BE")  # BM25 Expanded

        # 3. Semantic rewrite → Vector only (skip in bm25_graph mode)
        if use_vector and semantic_rewrite_embedding:
            sem_results = await self._retrieve_vector_only(semantic_rewrite_embedding)
            sem_ranked = [(m.id, score) for m, score in sem_results]
            all_ranked_lists.append(sem_ranked)
            all_weights.append(settings.rrf_vector_weight)
            for m, _ in sem_results:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("S")  # Semantic

        # 4. HyDE → Vector only (skip in bm25_graph mode)
        if use_vector and hyde_embedding:
            hyde_results = await self._retrieve_vector_only(hyde_embedding)
            hyde_ranked = [(m.id, score) for m, score in hyde_results]
            all_ranked_lists.append(hyde_ranked)
            all_weights.append(settings.rrf_vector_weight)
            for m, _ in hyde_results:
                all_memories[m.id] = m
                memory_sources.setdefault(m.id, []).append("H")  # HyDE

        # Weighted RRF fusion of all ranked lists (v4.6)
        fused_scores = weighted_rrf(all_ranked_lists, all_weights, k=settings.rrf_k)

        # Build scored memories with composite scores
        scored_memories = self._rerank_memories(
            memories=all_memories,
            fused_scores=fused_scores,
            query_embedding=original_embedding,  # May be None in bm25_graph mode
            sources=memory_sources,
        )

        # Apply cross-encoder reranking against ORIGINAL query
        if settings.reranker_enabled:
            scored_memories = self._apply_reranker(
                query=original_query,  # Rerank against original
                scored_memories=scored_memories,
                candidates_k=settings.reranker_candidates,
            )

        # Apply MMR for diversity (requires embeddings)
        if settings.mmr_enabled and use_vector and len(scored_memories) > self.final_k:
            scored_memories = self._apply_mmr(
                query_embedding=original_embedding,  # type: ignore
                scored_memories=scored_memories,
                k=self.final_k,
            )

        return scored_memories[:self.final_k]

    async def _retrieve_bm25_only(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[SemanticMemory, float]]:
        """BM25-only retrieval for expanded queries."""
        k = k or self.bm25_k
        return await self.db.fulltext_search_memories(query_text=query, k=k)

    async def _retrieve_vector_only(
        self,
        embedding: list[float],
        k: int | None = None,
    ) -> list[tuple[SemanticMemory, float]]:
        """Vector-only retrieval for semantic queries."""
        k = k or self.vector_k
        return await self.db.vector_search_memories(embedding=embedding, k=k)


async def hybrid_search(
    db: Neo4jClient,
    query: str,
    query_embedding: list[float] | None = None,
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
