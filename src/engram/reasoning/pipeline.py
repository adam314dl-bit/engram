"""Complete reasoning pipeline orchestrating retrieval, synthesis, and episode storage.

This is the main entry point for the reasoning system.

Supports two modes:
1. Standard mode: Retrieve memories -> Synthesize response
2. Two-phase mode: Retrieve candidates -> LLM select -> Fetch documents -> Synthesize
"""

import logging
from dataclasses import dataclass, field

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import Document, EpisodicMemory
from engram.reasoning.episode_manager import EpisodeManager
from engram.reasoning.re_reasoning import ReReasoner, ReReasoningResult
from engram.reasoning.selector import MemorySelector, SelectionResult
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

    # Two-phase retrieval info (optional)
    selection_result: SelectionResult | None = None
    source_documents: list[Document] = field(default_factory=list)

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
        self.selector = MemorySelector(llm_client=self.llm)
        self.episode_manager = EpisodeManager(
            db=db,
            embedding_service=self.embeddings,
        )
        self.re_reasoner = ReReasoner(db=db, llm_client=self.llm)

    async def reason(
        self,
        query: str,
        top_k_memories: int = 10,
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

    async def reason_with_documents(
        self,
        query: str,
        top_k_candidates: int = 100,
        temperature: float = 0.4,
    ) -> ReasoningResult:
        """
        Execute two-phase retrieval with full document context.

        Flow:
        1. Retrieve 100+ memory summaries (candidates)
        2. LLM selects which memories have needed info
        3. Fetch full source documents for selected memories
        4. Synthesize response using full documents

        Args:
            query: User query
            top_k_candidates: Number of candidates to retrieve (default 100)
            temperature: LLM temperature for synthesis

        Returns:
            ReasoningResult with answer, documents, and selection info
        """
        logger.info(f"Two-phase reasoning for query: {query[:50]}...")

        # 1. Retrieve many candidates (summaries only)
        retrieval_result = await self.retrieval.retrieve_candidates(
            query=query,
            top_k_memories=top_k_candidates,
        )

        logger.debug(f"Retrieved {len(retrieval_result.memories)} candidates")

        # 2. LLM selects relevant memories
        selection_result = await self.selector.select(
            query=query,
            candidates=retrieval_result.memories,
            max_candidates=top_k_candidates,
        )

        logger.debug(
            f"LLM selected {len(selection_result.selected_ids)} memories "
            f"({selection_result.selection_ratio:.1%})"
        )

        # 3. Fetch full source documents for selected memories
        source_documents: list[Document] = []
        if selection_result.selected_ids:
            source_documents = await self.db.get_source_documents_for_memories(
                memory_ids=selection_result.selected_ids
            )
            logger.debug(f"Fetched {len(source_documents)} source documents")

        # 4. Synthesize response using full documents
        synthesis = await self.synthesizer.synthesize_from_documents(
            query=query,
            documents=source_documents,
            retrieval=retrieval_result,
            temperature=temperature,
        )

        logger.debug(f"Synthesized response with behavior: {synthesis.behavior.name}")

        # 5. Create episode
        episode = await self.episode_manager.create_episode(
            synthesis=synthesis,
        )

        logger.info(
            f"Two-phase reasoning complete: {episode.behavior_name} "
            f"(confidence: {synthesis.confidence:.0%}, "
            f"docs: {len(source_documents)})"
        )

        return ReasoningResult(
            answer=synthesis.answer,
            episode=episode,
            retrieval=retrieval_result,
            synthesis=synthesis,
            confidence=synthesis.confidence,
            selection_result=selection_result,
            source_documents=source_documents,
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
    top_k: int = 10,
) -> ReasoningResult:
    """Convenience function for reasoning."""
    pipeline = ReasoningPipeline(db=db)
    return await pipeline.reason(query=query, top_k_memories=top_k)


async def reason_with_documents(
    db: Neo4jClient,
    query: str,
    top_k_candidates: int = 100,
) -> ReasoningResult:
    """
    Convenience function for two-phase reasoning with documents.

    Uses LLM selection to pick relevant memories from a large candidate pool,
    then fetches full source documents for context.
    """
    pipeline = ReasoningPipeline(db=db)
    return await pipeline.reason_with_documents(
        query=query,
        top_k_candidates=top_k_candidates,
    )
