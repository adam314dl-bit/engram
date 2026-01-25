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

    # v4.3 additions (multi-query enrichment)
    query_variants: dict[str, str] = field(default_factory=dict)  # label -> variant
    used_enrichment: bool = False

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

    async def retrieve_multi_query(
        self,
        query: str,
        bm25_expanded: str | None = None,
        semantic_rewrite: str | None = None,
        hyde_document: str | None = None,
        top_k_memories: int | None = None,
        top_k_episodes: int | None = None,
        include_episodes: bool = True,
    ) -> RetrievalResult:
        """
        Execute retrieval with multiple query variants (v4.3).

        Uses EnrichedQuery variants for multi-signal retrieval:
        - Original → full hybrid (BM25 + vector + graph)
        - BM25 expanded → BM25 only
        - Semantic rewrite → Vector only
        - HyDE document → Vector only

        All results fused with RRF, reranked against original query.

        Args:
            query: Original user query
            bm25_expanded: BM25-expanded query (synonyms, lemmas, domain terms)
            semantic_rewrite: Semantically rewritten query
            hyde_document: Hypothetical document for HyDE
            top_k_memories: Number of memories to return
            top_k_episodes: Number of episodes to return
            include_episodes: Whether to include similar episodes

        Returns:
            RetrievalResult with multi-query retrieved information
        """
        top_k_memories = top_k_memories or settings.retrieval_top_k
        top_k_episodes = top_k_episodes or 3

        # 1. Classify query (person/role, complexity)
        person_query_type, person_entity = classify_person_query(query)
        query_complexity, recommended_k = classify_query_complexity(query)

        # Use dynamic top_k if enabled
        if settings.dynamic_topk_enabled and top_k_memories == settings.retrieval_top_k:
            top_k_memories = recommended_k

        # 2. Embed all variants
        logger.debug("Embedding query variants...")
        query_embedding = await self.embeddings.embed(query)

        # Embed semantic rewrite if different from original
        semantic_rewrite_embedding = None
        if semantic_rewrite and semantic_rewrite != query:
            semantic_rewrite_embedding = await self.embeddings.embed(semantic_rewrite)

        # Embed HyDE document if present
        hyde_embedding = None
        if hyde_document:
            hyde_embedding = await self.embeddings.embed(hyde_document)

        # 3. Extract concepts from query
        concept_result = await self.concept_extractor.extract(query)
        query_concepts = concept_result.concepts

        # 4. Find matching concepts in graph
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

        # 5. Spread activation through concept network
        activation_result: ActivationResult | None = None
        activated_concepts: dict[str, float] = {}

        if seed_concept_ids:
            activation_result = await self.spreading.activate(
                seed_concept_ids, query_embedding
            )
            activated_concepts = activation_result.activations

        # 6. Get memories from activated concepts (for graph signal)
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

        # 7. Multi-query hybrid search
        logger.debug("Performing multi-query hybrid search")
        scored_memories = await self.hybrid.search_memories_multi_query(
            original_query=query,
            original_embedding=query_embedding,
            bm25_expanded=bm25_expanded,
            semantic_rewrite=semantic_rewrite,
            semantic_rewrite_embedding=semantic_rewrite_embedding,
            hyde_document=hyde_document,
            hyde_embedding=hyde_embedding,
            graph_memories=graph_memories,
            graph_memory_scores=graph_memory_scores,
        )

        # 8. Apply source weights
        for sm in scored_memories:
            source_type = None
            if sm.memory.metadata:
                source_type = sm.memory.metadata.get("source_type")
            sm.score = apply_source_weight(sm.score, source_type)

        scored_memories.sort(key=lambda x: x.score, reverse=True)
        scored_memories = scored_memories[:top_k_memories]

        # 9. Fetch raw tables for summary memories
        raw_tables: dict[str, SemanticMemory] = {}
        summary_ids = [
            sm.memory.id for sm in scored_memories
            if sm.memory.memory_type == "table_summary"
        ]
        if summary_ids:
            from engram.preprocessing.table_enricher import fetch_raw_tables_for_summaries
            raw_tables = await fetch_raw_tables_for_summaries(self.db, summary_ids)

        # 10. Count retrieval sources
        retrieval_sources: dict[str, int] = {}
        for sm in scored_memories:
            for source in sm.sources:
                retrieval_sources[source] = retrieval_sources.get(source, 0) + 1

        # 11. Find similar episodes
        episodes: list[ScoredEpisode] = []
        if include_episodes:
            episodes = await self.hybrid.search_similar_episodes(
                query_embedding=query_embedding,
                k=top_k_episodes,
            )

        # 12. Update access counts
        for sm in scored_memories:
            await self.db.update_memory_access(sm.memory.id)

        # Build query variants dict for debugging
        query_variants = {"original": query}
        if bm25_expanded:
            query_variants["bm25_expanded"] = bm25_expanded
        if semantic_rewrite:
            query_variants["semantic_rewrite"] = semantic_rewrite
        if hyde_document:
            query_variants["hyde"] = hyde_document

        logger.info(
            f"Multi-query retrieved {len(scored_memories)} memories, "
            f"{len(episodes)} episodes from {len(activated_concepts)} concepts "
            f"(variants: {len(query_variants)})"
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
            transliteration_variants=[],
            raw_tables=raw_tables,
            query_variants=query_variants,
            used_enrichment=True,
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
    top_k: int = 200,
) -> RetrievalResult:
    """Convenience function for retrieval."""
    pipeline = RetrievalPipeline(db=db)
    return await pipeline.retrieve(query=query, top_k_memories=top_k)
