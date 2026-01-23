"""Complete reasoning pipeline orchestrating retrieval, synthesis, and episode storage.

This is the main entry point for the reasoning system.

Flow: Query -> Concepts -> Spreading Activation ->
      Hybrid Search (vector + BM25 + graph + RRF) ->
      Reranker + MMR -> Top-k Memories -> Synthesis
"""

import logging
from dataclasses import dataclass

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import EpisodicMemory
from engram.reasoning.episode_manager import EpisodeManager
from engram.reasoning.re_reasoning import ReReasoner, ReReasoningResult
from engram.reasoning.synthesizer import ResponseSynthesizer, SynthesisResult
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Complete result from the reasoning pipeline."""

    # The answer
    answer: str

    # Episode created for this interaction
    episode: EpisodicMemory

    # Retrieval results
    retrieval: RetrievalResult

    # Synthesis details
    synthesis: SynthesisResult

    # Metadata
    confidence: float

    @property
    def episode_id(self) -> str:
        """Get episode ID for feedback."""
        return self.episode.id


class ReasoningPipeline:
    """
    Complete reasoning pipeline for Engram.

    Pipeline steps:
    1. Retrieve relevant context (memories, episodes, concepts)
    2. Synthesize response using LLM
    3. Extract behavior pattern
    4. Create and store episode
    5. Return complete result

    Also supports re-reasoning on failure.
    """

    def __init__(
        self,
        db: Neo4jClient,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.db = db
        self.llm = llm_client or get_llm_client()
        self.embeddings = embedding_service or get_embedding_service()

        # Initialize components
        self.retrieval = RetrievalPipeline(
            db=db,
            embedding_service=self.embeddings,
        )
        self.synthesizer = ResponseSynthesizer(llm_client=self.llm)
        self.episode_manager = EpisodeManager(
            db=db,
            embedding_service=self.embeddings,
        )
        self.re_reasoner = ReReasoner(db=db, llm_client=self.llm)

    async def reason(
        self,
        query: str,
        top_k_memories: int = 100,
        top_k_episodes: int = 3,
        temperature: float = 0.4,
        force_include_nodes: list[str] | None = None,
        force_exclude_nodes: list[str] | None = None,
    ) -> ReasoningResult:
        """
        Execute the full reasoning pipeline.

        Args:
            query: User query
            top_k_memories: Number of memories to retrieve
            top_k_episodes: Number of similar episodes to retrieve
            temperature: LLM temperature

        Returns:
            ReasoningResult with answer, episode, and metadata
        """
        logger.info(f"Reasoning for query: {query[:50]}...")

        # 1. Retrieve relevant context
        retrieval_result = await self.retrieval.retrieve(
            query=query,
            top_k_memories=top_k_memories,
            top_k_episodes=top_k_episodes,
        )

        # Apply force exclude - remove specified nodes
        if force_exclude_nodes:
            exclude_set = set(force_exclude_nodes)
            retrieval_result.memories = [
                m for m in retrieval_result.memories
                if m.memory.id not in exclude_set
            ]
            # Also exclude from activated concepts
            retrieval_result.activated_concepts = {
                k: v for k, v in retrieval_result.activated_concepts.items()
                if k not in exclude_set
            }

        # Apply force include - fetch and add specified nodes
        if force_include_nodes:
            from engram.retrieval.hybrid_search import ScoredMemory
            for node_id in force_include_nodes:
                # Check if already in results
                existing_ids = {m.memory.id for m in retrieval_result.memories}
                if node_id not in existing_ids and node_id.startswith("mem_"):
                    # Try to fetch the memory
                    memory = await self.db.get_semantic_memory(node_id)
                    if memory:
                        retrieval_result.memories.insert(0, ScoredMemory(
                            memory=memory,
                            score=1.0,  # Forced nodes get max score
                            sources=["F"],  # F = Forced
                        ))
                elif node_id not in existing_ids and node_id.startswith("c_"):
                    # Force include a concept - add to activated concepts
                    if node_id not in retrieval_result.activated_concepts:
                        retrieval_result.activated_concepts[node_id] = 1.0

        logger.debug(
            f"Retrieved {len(retrieval_result.memories)} memories, "
            f"{len(retrieval_result.episodes)} episodes"
        )

        # 2. Synthesize response
        synthesis = await self.synthesizer.synthesize(
            query=query,
            retrieval=retrieval_result,
            temperature=temperature,
        )

        logger.debug(f"Synthesized response with behavior: {synthesis.behavior.name}")

        # 3. Create episode
        episode = await self.episode_manager.create_episode(
            synthesis=synthesis,
        )

        logger.info(
            f"Reasoning complete: {episode.behavior_name} "
            f"(confidence: {synthesis.confidence:.0%})"
        )

        return ReasoningResult(
            answer=synthesis.answer,
            episode=episode,
            retrieval=retrieval_result,
            synthesis=synthesis,
            confidence=synthesis.confidence,
        )

    async def re_reason(
        self,
        episode_id: str,
        user_feedback: str | None = None,
    ) -> ReReasoningResult:
        """
        Re-reason after a failed episode.

        Args:
            episode_id: ID of the failed episode
            user_feedback: Optional feedback about what went wrong

        Returns:
            ReReasoningResult with alternative answer
        """
        # Get the failed episode
        episode = await self.episode_manager.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")

        # Record failure
        await self.episode_manager.record_failure(episode_id)

        # Re-reason
        result = await self.re_reasoner.re_reason(
            failed_episode=episode,
            user_feedback=user_feedback,
        )

        logger.info(
            f"Re-reasoning complete for {episode_id}: "
            f"approach_changed={result.approach_changed}"
        )

        return result

    async def record_feedback(
        self,
        episode_id: str,
        positive: bool,
    ) -> None:
        """
        Record feedback for an episode.

        Args:
            episode_id: Episode to update
            positive: True for positive feedback, False for negative
        """
        if positive:
            await self.episode_manager.record_success(episode_id)
        else:
            await self.episode_manager.record_failure(episode_id)


async def reason(
    db: Neo4jClient,
    query: str,
    top_k: int = 100,
) -> ReasoningResult:
    """Convenience function for reasoning."""
    pipeline = ReasoningPipeline(db=db)
    return await pipeline.reason(query=query, top_k_memories=top_k)
