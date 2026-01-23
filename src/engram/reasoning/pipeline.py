"""Complete reasoning pipeline orchestrating retrieval, synthesis, and episode storage.

This is the main entry point for the reasoning system.

Supports two modes:
1. Standard mode: Retrieve memories -> Synthesize response
2. Two-phase mode: Always runs both phases in parallel:
   - Phase 1: Memory retrieval (graph + vector + BM25) -> LLM selection
   - Phase 2: BM25 chunk search -> LLM selection
   - Merge & rank documents (docs in both phases get priority)
   - Select top N documents for synthesis
"""

import asyncio
import logging
from dataclasses import dataclass, field

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import Document, EpisodicMemory
from engram.reasoning.episode_manager import EpisodeManager
from engram.reasoning.re_reasoning import ReReasoner, ReReasoningResult
from engram.reasoning.selector import ChunkSelectionResult, MemorySelector, SelectionResult
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

    # Phase 2 fallback info (optional)
    phase2_triggered: bool = False
    chunk_selection_result: ChunkSelectionResult | None = None

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
        top_k_candidates: int | None = None,
        temperature: float = 0.4,
    ) -> ReasoningResult:
        """
        Execute two-phase retrieval - always runs both phases in parallel.

        Flow:
        Phase 1 (parallel): Retrieve memories -> LLM select
        Phase 2 (parallel): BM25 search on raw chunks -> LLM select
        Merge & deduplicate documents from both phases
        Rank by score (docs in both phases get priority)
        Select top N documents (default 6)
        Send to LLM for synthesis

        Args:
            query: User query
            top_k_candidates: Number of candidates to retrieve (default from settings)
            temperature: LLM temperature for synthesis

        Returns:
            ReasoningResult with answer, documents, and selection info
        """
        top_k_candidates = top_k_candidates or settings.phase1_candidates
        max_docs = settings.max_synthesis_documents

        logger.info(f"Two-phase reasoning for query: {query[:50]}...")

        # =================================================================
        # Run Phase 1 and Phase 2 in parallel
        # =================================================================

        async def run_phase1() -> tuple[RetrievalResult, SelectionResult]:
            """Phase 1: Memory-based retrieval + LLM selection."""
            retrieval = await self.retrieval.retrieve_candidates(
                query=query,
                top_k_memories=top_k_candidates,
            )
            logger.debug(f"Phase 1: Retrieved {len(retrieval.memories)} memory candidates")

            selection = await self.selector.select(
                query=query,
                candidates=retrieval.memories,
                max_candidates=top_k_candidates,
            )
            logger.info(
                f"Phase 1: Selected {len(selection.selected_ids)} memories, "
                f"confidence: {selection.confidence:.1f}"
            )
            return retrieval, selection

        async def run_phase2() -> ChunkSelectionResult | None:
            """Phase 2: BM25 chunk search + LLM selection."""
            chunk_results = await self.db.fulltext_search_chunks(query, k=100)
            logger.debug(f"Phase 2: Found {len(chunk_results)} chunks via BM25")

            if not chunk_results:
                return None

            chunk_selection = await self.selector.select_from_chunks(
                query=query,
                chunks=chunk_results,
                max_candidates=100,
            )
            logger.info(
                f"Phase 2: Selected {len(chunk_selection.selected_ids)} chunks, "
                f"confidence: {chunk_selection.confidence:.1f}"
            )
            return chunk_selection

        # Run both phases in parallel
        (retrieval_result, selection_result), chunk_selection_result = await asyncio.gather(
            run_phase1(),
            run_phase2(),
        )

        # =================================================================
        # Merge & Score Documents
        # =================================================================
        doc_scores: dict[str, float] = {}  # doc_id -> score

        # Phase 1 documents (from memories)
        phase1_doc_ids: set[str] = set()
        if selection_result.selected_ids:
            phase1_docs = await self.db.get_source_documents_for_memories(
                memory_ids=selection_result.selected_ids
            )
            for doc in phase1_docs:
                doc_scores[doc.id] = doc_scores.get(doc.id, 0) + 1.0
                phase1_doc_ids.add(doc.id)

        # Phase 2 documents (from chunks)
        phase2_doc_ids: set[str] = set()
        if chunk_selection_result and chunk_selection_result.chunks:
            chunk_doc_ids = {
                doc_id for _, (_, doc_id) in chunk_selection_result.chunks.items()
            }
            for doc_id in chunk_doc_ids:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0
                phase2_doc_ids.add(doc_id)

        # Log overlap between phases
        overlap = phase1_doc_ids & phase2_doc_ids
        logger.debug(
            f"Phase 1 docs: {len(phase1_doc_ids)}, Phase 2 docs: {len(phase2_doc_ids)}, "
            f"overlap: {len(overlap)}"
        )

        # =================================================================
        # Select Top N Documents
        # =================================================================
        # Sort by score (docs in both phases score 2.0, single phase score 1.0)
        sorted_doc_ids = sorted(
            doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True
        )
        top_doc_ids = sorted_doc_ids[:max_docs]

        # Fetch full documents
        source_documents: list[Document] = []
        for doc_id in top_doc_ids:
            doc = await self.db.get_document(doc_id)
            if doc:
                source_documents.append(doc)

        logger.info(
            f"Selected {len(source_documents)} documents for synthesis "
            f"(from {len(doc_scores)} unique docs)"
        )

        # =================================================================
        # Synthesize response
        # =================================================================
        synthesis = await self.synthesizer.synthesize_from_documents(
            query=query,
            documents=source_documents,
            retrieval=retrieval_result,
            temperature=temperature,
        )

        logger.debug(f"Synthesized response with behavior: {synthesis.behavior.name}")

        # Create episode
        episode = await self.episode_manager.create_episode(
            synthesis=synthesis,
        )

        logger.info(
            f"Two-phase reasoning complete: {episode.behavior_name} "
            f"(confidence: {synthesis.confidence:.0%}, docs: {len(source_documents)})"
        )

        return ReasoningResult(
            answer=synthesis.answer,
            episode=episode,
            retrieval=retrieval_result,
            synthesis=synthesis,
            confidence=synthesis.confidence,
            selection_result=selection_result,
            source_documents=source_documents,
            phase2_triggered=True,  # Always True now since we always run both phases
            chunk_selection_result=chunk_selection_result,
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
    top_k_candidates: int | None = None,
) -> ReasoningResult:
    """
    Convenience function for two-phase reasoning with documents.

    Always runs both phases:
    - Phase 1: Memory-based retrieval with LLM selection
    - Phase 2: BM25 chunk search with LLM selection
    Merges and deduplicates documents, then selects top N for synthesis.
    """
    pipeline = ReasoningPipeline(db=db)
    return await pipeline.reason_with_documents(
        query=query,
        top_k_candidates=top_k_candidates,
    )
