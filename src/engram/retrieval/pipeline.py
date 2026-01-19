"""Full retrieval pipeline combining all retrieval components.

Orchestrates:
1. Concept extraction from query
2. Spreading activation through concept network
3. Hybrid search (vector + BM25 + graph)
4. Similar episode retrieval
5. Final ranking and result assembly
"""

import logging
from dataclasses import dataclass, field

from engram.config import settings
from engram.ingestion.concept_extractor import ConceptExtractor
from engram.models import Concept, EpisodicMemory, SemanticMemory
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.hybrid_search import HybridSearch, ScoredEpisode, ScoredMemory
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

    @property
    def top_memories(self) -> list[SemanticMemory]:
        """Get just the memory objects from scored results."""
        return [sm.memory for sm in self.memories]

    @property
    def top_episodes(self) -> list[EpisodicMemory]:
        """Get just the episode objects from scored results."""
        return [se.episode for se in self.episodes]


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
        force_include_nodes: list[str] | None = None,
        force_exclude_nodes: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.

        Args:
            query: User query text
            top_k_memories: Number of memories to return (default from settings)
            top_k_episodes: Number of episodes to return (default 3)
            include_episodes: Whether to include similar episodes
            force_include_nodes: Node IDs to force include in results
            force_exclude_nodes: Node IDs to force exclude from results

        Returns:
            RetrievalResult with all retrieved information
        """
        top_k_memories = top_k_memories or settings.retrieval_top_k
        top_k_episodes = top_k_episodes or 3
        force_include_ids = force_include_nodes or []
        force_exclude_ids = set(force_exclude_nodes or [])

        # 1. Embed query
        logger.debug(f"Embedding query: {query[:50]}...")
        query_embedding = await self.embeddings.embed(query)

        # 2. Extract concepts from query
        logger.debug("Extracting concepts from query")
        concept_result = await self.concept_extractor.extract(query)
        query_concepts = concept_result.concepts

        # 3. Find matching concepts in graph
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

        # 4. Spread activation through concept network
        logger.debug(f"Spreading activation from {len(seed_concept_ids)} seed concepts")
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

        # 6. Hybrid search (combining all signals)
        logger.debug("Performing hybrid search")
        scored_memories = await self.hybrid.search_memories(
            query=query,
            query_embedding=query_embedding,
            graph_memories=graph_memories,
            graph_memory_scores=graph_memory_scores,
            force_include_ids=force_include_ids,
            force_exclude_ids=force_exclude_ids,
        )

        # Apply exclusions and limit to top_k
        if force_exclude_ids:
            scored_memories = [
                sm for sm in scored_memories
                if sm.memory.id not in force_exclude_ids
            ]
        scored_memories = scored_memories[:top_k_memories]

        # Count retrieval sources
        retrieval_sources: dict[str, int] = {}
        for sm in scored_memories:
            for source in sm.sources:
                retrieval_sources[source] = retrieval_sources.get(source, 0) + 1

        # 7. Find similar episodes (reasoning templates)
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
            f"from {len(activated_concepts)} activated concepts"
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
