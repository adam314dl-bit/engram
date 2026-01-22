"""Full retrieval pipeline combining all retrieval components.

Orchestrates:
1. Concept extraction from query
2. Query classification (person/role, complexity)
3. Transliteration expansion for mixed-script queries
4. Spreading activation through concept network
5. Hybrid search (vector + BM25 + graph)
6. Source weighting and quality filtering
7. MMR reranking for diversity
8. Raw table fetching for summaries
9. Similar episode retrieval
10. Final ranking and result assembly
"""

import logging
from dataclasses import dataclass, field

from engram.config import settings
from engram.ingestion.concept_extractor import ConceptExtractor
from engram.ingestion.person_extractor import PersonQueryType, classify_person_query
from engram.models import Concept, EpisodicMemory, SemanticMemory
from engram.preprocessing.transliteration import expand_query_transliteration
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.hybrid_search import (
    HybridSearch,
    ScoredEpisode,
    ScoredMemory,
    classify_query_complexity,
)
from engram.retrieval.quality_filter import apply_source_weight
from engram.retrieval.spreading_activation import (
    ActivationResult,
    SpreadingActivation,
)
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Complete result from the retrieval pipeline."""

    # Query info
    query: str
    query_embedding: list[float]

    # Extracted concepts from query
    query_concepts: list[Concept]

    # Spreading activation results
    activated_concepts: dict[str, float]  # concept_id -> activation
    activation_result: ActivationResult | None = None

    # Retrieved memories (ranked)
    memories: list[ScoredMemory] = field(default_factory=list)

    # Similar episodes for reasoning templates
    episodes: list[ScoredEpisode] = field(default_factory=list)

    # Debug info
    retrieval_sources: dict[str, int] = field(default_factory=dict)  # source -> count

    # v3.3 additions
    query_complexity: str = "moderate"  # simple, moderate, complex
    person_query_type: PersonQueryType = PersonQueryType.GENERAL
    transliteration_variants: list[str] = field(default_factory=list)
    raw_tables: dict[str, SemanticMemory] = field(default_factory=dict)  # summary_id -> raw

    @property
    def top_memories(self) -> list[SemanticMemory]:
        """Get just the memory objects from scored results."""
        return [sm.memory for sm in self.memories]

    @property
    def top_episodes(self) -> list[EpisodicMemory]:
        """Get just the episode objects from scored results."""
        return [se.episode for se in self.episodes]

    @property
    def has_person_query(self) -> bool:
        """Check if this was a person-related query."""
        return self.person_query_type != PersonQueryType.GENERAL


class RetrievalPipeline:
    """
    Full retrieval pipeline for Engram.

    Pipeline steps:
    1. Embed query
    2. Extract concepts from query
    3. Find matching concepts in graph
    4. Spread activation through concept network
    5. Get memories connected to activated concepts
    6. Hybrid search (combining graph + vector + BM25)
    7. Find similar episodes (reasoning templates)
    8. Return unified result
    """

    def __init__(
        self,
        db: Neo4jClient,
        embedding_service: EmbeddingService | None = None,
        concept_extractor: ConceptExtractor | None = None,
        spreading_activation: SpreadingActivation | None = None,
        hybrid_search: HybridSearch | None = None,
    ) -> None:
        self.db = db
        self.embeddings = embedding_service or get_embedding_service()
        self.concept_extractor = concept_extractor or ConceptExtractor()
        self.spreading = spreading_activation or SpreadingActivation(db=db)
        self.hybrid = hybrid_search or HybridSearch(db=db)

    async def retrieve(
        self,
        query: str,
        top_k_memories: int | None = None,
        top_k_episodes: int | None = None,
        include_episodes: bool = True,
        use_transliteration: bool = True,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.

        Args:
            query: User query text
            top_k_memories: Number of memories to return (default from settings)
            top_k_episodes: Number of episodes to return (default 3)
            include_episodes: Whether to include similar episodes
            use_transliteration: Whether to expand query with transliteration variants

        Returns:
            RetrievalResult with all retrieved information
        """
        # 1. Classify query (person/role, complexity)
        person_query_type, person_entity = classify_person_query(query)
        query_complexity, recommended_k = classify_query_complexity(query)

        # Use dynamic top_k if enabled
        if settings.dynamic_topk_enabled and top_k_memories is None:
            top_k_memories = recommended_k
        else:
            top_k_memories = top_k_memories or settings.retrieval_top_k

        top_k_episodes = top_k_episodes or 3

        logger.debug(
            f"Query classification: complexity={query_complexity}, "
            f"person_type={person_query_type.value}, k={top_k_memories}"
        )

        # 2. Expand query with transliteration variants
        transliteration_variants: list[str] = []
        if use_transliteration:
            transliteration_variants = expand_query_transliteration(query)
            if len(transliteration_variants) > 1:
                logger.debug(f"Transliteration variants: {transliteration_variants}")

        # 3. Embed query (and variants if needed)
        logger.debug(f"Embedding query: {query[:50]}...")
        query_embedding = await self.embeddings.embed(query)

        # 4. Extract concepts from query
        logger.debug("Extracting concepts from query")
        concept_result = await self.concept_extractor.extract(query)
        query_concepts = concept_result.concepts

        # 5. Find matching concepts in graph
        seed_concept_ids: list[str] = []
        matched_concepts: list[Concept] = []

        for concept in query_concepts:
            # Try to find existing concept by name
            existing = await self.db.get_concept_by_name(concept.name)
            if existing:
                seed_concept_ids.append(existing.id)
                matched_concepts.append(existing)

        # If no concepts matched, use vector search to find similar concepts
        if not seed_concept_ids:
            logger.debug("No exact concept matches, using vector search")
            vector_concepts = await self.db.vector_search_concepts(
                embedding=query_embedding, k=5
            )
            for concept, score in vector_concepts:
                if score > 0.5:  # Similarity threshold
                    seed_concept_ids.append(concept.id)
                    matched_concepts.append(concept)

        # 6. Spread activation through concept network
        logger.debug(f"Spreading activation from {len(seed_concept_ids)} seed concepts")
        activation_result: ActivationResult | None = None
        activated_concepts: dict[str, float] = {}

        if seed_concept_ids:
            activation_result = await self.spreading.activate(
                seed_concept_ids, query_embedding
            )
            activated_concepts = activation_result.activations

        # 7. Get memories connected to activated concepts
        graph_memories: list[SemanticMemory] = []
        graph_memory_scores: dict[str, float] = {}

        if activated_concepts:
            # Get concepts with activation above threshold
            active_concept_ids = [
                cid for cid, act in activated_concepts.items()
                if act >= settings.activation_threshold
            ]

            if active_concept_ids:
                graph_memories = await self.db.get_memories_for_concepts(
                    concept_ids=active_concept_ids,
                    limit=top_k_memories * 2,
                )

                # Score memories by sum of activations of their concepts
                for memory in graph_memories:
                    score = sum(
                        activated_concepts.get(cid, 0)
                        for cid in memory.concept_ids
                    )
                    graph_memory_scores[memory.id] = score

        # 8. Hybrid search (combining all signals)
        # Note: MMR and reranker are applied inside search_memories
        logger.debug("Performing hybrid search")
        scored_memories = await self.hybrid.search_memories(
            query=query,
            query_embedding=query_embedding,
            graph_memories=graph_memories,
            graph_memory_scores=graph_memory_scores,
            use_dynamic_k=False,  # Already applied dynamic k above
        )

        # 9. Apply source weights to scores
        for sm in scored_memories:
            source_type = None
            if sm.memory.metadata:
                source_type = sm.memory.metadata.get("source_type")
            sm.score = apply_source_weight(sm.score, source_type)

        # Re-sort after weighting
        scored_memories.sort(key=lambda x: x.score, reverse=True)

        # Limit to top_k
        scored_memories = scored_memories[:top_k_memories]

        # 10. Fetch raw tables for summary memories
        raw_tables: dict[str, SemanticMemory] = {}
        summary_ids = [
            sm.memory.id for sm in scored_memories
            if sm.memory.memory_type == "table_summary"
        ]
        if summary_ids:
            from engram.preprocessing.table_enricher import fetch_raw_tables_for_summaries
            raw_tables = await fetch_raw_tables_for_summaries(self.db, summary_ids)
            if raw_tables:
                logger.debug(f"Fetched {len(raw_tables)} raw tables for generation")

        # Count retrieval sources
        retrieval_sources: dict[str, int] = {}
        for sm in scored_memories:
            for source in sm.sources:
                retrieval_sources[source] = retrieval_sources.get(source, 0) + 1

        # 11. Find similar episodes (reasoning templates)
        episodes: list[ScoredEpisode] = []
        if include_episodes:
            logger.debug("Finding similar episodes")
            episodes = await self.hybrid.search_similar_episodes(
                query_embedding=query_embedding,
                k=top_k_episodes,
            )

        # Update access counts for retrieved memories
        for sm in scored_memories:
            await self.db.update_memory_access(sm.memory.id)

        logger.info(
            f"Retrieved {len(scored_memories)} memories, {len(episodes)} episodes "
            f"from {len(activated_concepts)} activated concepts "
            f"(complexity={query_complexity}, person={person_query_type.value})"
        )

        return RetrievalResult(
            query=query,
            query_embedding=query_embedding,
            query_concepts=matched_concepts,
            activated_concepts=activated_concepts,
            activation_result=activation_result,
            memories=scored_memories,
            episodes=episodes,
            retrieval_sources=retrieval_sources,
            query_complexity=query_complexity,
            person_query_type=person_query_type,
            transliteration_variants=transliteration_variants,
            raw_tables=raw_tables,
        )

    async def retrieve_candidates(
        self,
        query: str,
        top_k_memories: int = 100,
    ) -> RetrievalResult:
        """
        Retrieve many candidates for LLM selection (two-phase retrieval).

        Simplified retrieval that:
        - Gets many more candidates (default 100)
        - Skips reranker and MMR (LLM will select)
        - Returns raw candidates with scores

        Args:
            query: User query text
            top_k_memories: Number of memories to return (default 100)

        Returns:
            RetrievalResult with many candidate memories
        """
        logger.debug(f"Retrieving {top_k_memories} candidates for LLM selection")

        # 1. Embed query
        query_embedding = await self.embeddings.embed(query)

        # 2. Extract concepts from query
        concept_result = await self.concept_extractor.extract(query)
        query_concepts = concept_result.concepts

        # 3. Find matching concepts in graph
        seed_concept_ids: list[str] = []
        matched_concepts: list[Concept] = []

        for concept in query_concepts:
            existing = await self.db.get_concept_by_name(concept.name)
            if existing:
                seed_concept_ids.append(existing.id)
                matched_concepts.append(existing)

        # If no concepts matched, use vector search
        if not seed_concept_ids:
            vector_concepts = await self.db.vector_search_concepts(
                embedding=query_embedding, k=5
            )
            for concept, score in vector_concepts:
                if score > 0.5:
                    seed_concept_ids.append(concept.id)
                    matched_concepts.append(concept)

        # 4. Spread activation through concept network
        activation_result: ActivationResult | None = None
        activated_concepts: dict[str, float] = {}

        if seed_concept_ids:
            activation_result = await self.spreading.activate(
                seed_concept_ids, query_embedding
            )
            activated_concepts = activation_result.activations

        # 5. Get memories connected to activated concepts
        graph_memories: list[SemanticMemory] = []
        graph_memory_scores: dict[str, float] = {}

        if activated_concepts:
            active_concept_ids = [
                cid for cid, act in activated_concepts.items()
                if act >= settings.activation_threshold
            ]

            if active_concept_ids:
                graph_memories = await self.db.get_memories_for_concepts(
                    concept_ids=active_concept_ids,
                    limit=top_k_memories * 2,
                )

                for memory in graph_memories:
                    score = sum(
                        activated_concepts.get(cid, 0)
                        for cid in memory.concept_ids
                    )
                    graph_memory_scores[memory.id] = score

        # 6. Get candidates from vector + BM25 + graph (WITHOUT reranker/MMR)
        # Use larger k for initial retrieval
        vector_results = await self.db.vector_search_memories(
            embedding=query_embedding, k=top_k_memories
        )
        bm25_results = await self.db.fulltext_search_memories(
            query_text=query, k=top_k_memories
        )

        # Build memory lookup and track sources
        all_memories: dict[str, SemanticMemory] = {}
        memory_sources: dict[str, list[str]] = {}

        for m, _ in vector_results:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("V")  # V = Vector

        for m, _ in bm25_results:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("B")  # B = BM25

        for m in graph_memories:
            all_memories[m.id] = m
            memory_sources.setdefault(m.id, []).append("G")  # G = Graph

        # Simple scoring: combine signals
        from engram.retrieval.fusion import rrf_scores_only
        vector_ranked = [(m.id, score) for m, score in vector_results]
        bm25_ranked = [(m.id, score) for m, score in bm25_results]
        graph_ranked = [
            (m.id, graph_memory_scores.get(m.id, 0))
            for m in graph_memories
        ]
        graph_ranked.sort(key=lambda x: x[1], reverse=True)

        ranked_lists = [r for r in [vector_ranked, bm25_ranked, graph_ranked] if r]
        fused_scores = rrf_scores_only(ranked_lists)

        # Build scored memories
        scored_memories: list[ScoredMemory] = []
        for memory_id, memory in all_memories.items():
            score = fused_scores.get(memory_id, 0)
            scored_memories.append(ScoredMemory(
                memory=memory,
                score=score,
                sources=memory_sources.get(memory_id, []),
            ))

        # Sort by score
        scored_memories.sort(key=lambda x: x.score, reverse=True)
        scored_memories = scored_memories[:top_k_memories]

        # Count retrieval sources
        retrieval_sources: dict[str, int] = {}
        for sm in scored_memories:
            for source in sm.sources:
                retrieval_sources[source] = retrieval_sources.get(source, 0) + 1

        logger.info(
            f"Retrieved {len(scored_memories)} candidates for LLM selection "
            f"from {len(activated_concepts)} activated concepts"
        )

        return RetrievalResult(
            query=query,
            query_embedding=query_embedding,
            query_concepts=matched_concepts,
            activated_concepts=activated_concepts,
            activation_result=activation_result,
            memories=scored_memories,
            episodes=[],  # Skip episodes for candidate retrieval
            retrieval_sources=retrieval_sources,
            query_complexity="candidate_retrieval",
        )

    async def retrieve_for_concepts(
        self,
        concept_ids: list[str],
        query_embedding: list[float] | None = None,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """
        Retrieve memories for specific concepts (used by reasoning).

        Args:
            concept_ids: Concept IDs to retrieve for
            query_embedding: Optional query embedding for relevance scoring
            top_k: Number of memories to return

        Returns:
            List of scored memories
        """
        memories = await self.db.get_memories_for_concepts(
            concept_ids=concept_ids,
            limit=top_k * 2,
        )

        scored: list[ScoredMemory] = []
        for memory in memories:
            score = memory.importance / 10.0

            if query_embedding and memory.embedding:
                from engram.retrieval.embeddings import cosine_similarity
                relevance = cosine_similarity(query_embedding, memory.embedding)
                score = (score + relevance) / 2

            scored.append(ScoredMemory(memory=memory, score=score, sources=["graph"]))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


async def retrieve(
    db: Neo4jClient,
    query: str,
    top_k: int = 10,
) -> RetrievalResult:
    """Convenience function for retrieval."""
    pipeline = RetrievalPipeline(db=db)
    return await pipeline.retrieve(query=query, top_k_memories=top_k)


async def retrieve_candidates(
    db: Neo4jClient,
    query: str,
    top_k: int = 100,
) -> RetrievalResult:
    """
    Retrieve many candidates for LLM selection (two-phase retrieval).

    Returns raw candidates without reranking or MMR filtering.
    Used when LLM will select which memories are relevant.
    """
    pipeline = RetrievalPipeline(db=db)
    return await pipeline.retrieve_candidates(query=query, top_k_memories=top_k)
