"""Complete reasoning pipeline orchestrating retrieval, synthesis, and episode storage.

This is the main entry point for the reasoning system.

v4.3 Flow (when query_enrichment_enabled=true):
  Query -> Intent Classification (can skip retrieval) ->
  Query Enrichment (multi-query) -> Multi-Query Retrieval ->
  Light CRAG (grade top N) -> Synthesis ->
  Light Confidence (no LLM) -> Episode Creation

Standard Flow:
  Query -> Concepts -> Spreading Activation ->
  Hybrid Search (vector + BM25 + graph + RRF) ->
  Reranker + MMR -> Top-k Memories -> Synthesis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from engram.config import settings
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
    episode: EpisodicMemory | None

    # Retrieval results
    retrieval: RetrievalResult | None

    # Synthesis details
    synthesis: SynthesisResult | None

    # Metadata
    confidence: float

    # v4.3 additions
    intent: "IntentResult | None" = None
    crag: "CRAGResult | None" = None
    confidence_result: "ConfidenceResult | None" = None
    enriched_query: "EnrichedQuery | None" = None
    used_v43_pipeline: bool = False
    processing_time_ms: float = 0.0

    @property
    def episode_id(self) -> str | None:
        """Get episode ID for feedback."""
        return self.episode.id if self.episode else None


class ReasoningPipeline:
    """
    Complete reasoning pipeline for Engram.

    v4.3 pipeline steps (when query_enrichment_enabled=true):
    1. Intent Classification (can skip retrieval for greetings)
    2. Query Enrichment (multi-query generation)
    3. Multi-Query Retrieval
    4. Light CRAG (grade top N docs)
    5. Synthesis
    6. Light Confidence (no extra LLM call)
    7. Episode Creation

    Standard pipeline steps:
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

        # Initialize standard components
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

        # v4.3 components (initialized lazily)
        self._intent_classifier: "IntentClassifier | None" = None
        self._crag_evaluator: "CRAGEvaluator | None" = None
        self._confidence_calibrator: "ConfidenceCalibrator | None" = None
        self._query_enrichment: "QueryEnrichmentPipeline | None" = None
        self._v43_initialized = False

    async def _init_v43_components(self) -> None:
        """Lazily initialize v4.3 components."""
        if self._v43_initialized:
            return

        if settings.query_enrichment_enabled:
            # Import here to avoid circular imports
            from engram.query.enrichment import QueryEnrichmentPipeline
            from engram.reasoning.confidence import ConfidenceCalibrator
            from engram.reasoning.intent_classifier import IntentClassifier
            from engram.retrieval.crag import CRAGEvaluator

            if settings.standard_intent_enabled:
                self._intent_classifier = IntentClassifier(llm_client=self.llm)

            if settings.standard_crag_enabled:
                self._crag_evaluator = CRAGEvaluator(llm_client=self.llm)

            if settings.standard_confidence_enabled:
                self._confidence_calibrator = ConfidenceCalibrator(llm_client=self.llm)

            self._query_enrichment = QueryEnrichmentPipeline(
                db=self.db,
                llm_client=self.llm,
            )
            await self._query_enrichment.initialize()

            self._v43_initialized = True
            logger.info("v4.3 pipeline components initialized")

    async def reason(
        self,
        query: str,
        top_k_memories: int = 200,
        top_k_episodes: int = 3,
        temperature: float = 0.4,
        force_include_nodes: list[str] | None = None,
        force_exclude_nodes: list[str] | None = None,
    ) -> ReasoningResult:
        """
        Execute the full reasoning pipeline.

        When query_enrichment_enabled=true (v4.3), uses the enhanced pipeline:
        1. Intent Classification (can skip retrieval)
        2. Query Enrichment (multi-query generation)
        3. Multi-Query Retrieval
        4. Light CRAG (grade top N docs)
        5. Synthesis
        6. Light Confidence
        7. Episode Creation

        Args:
            query: User query
            top_k_memories: Number of memories to retrieve
            top_k_episodes: Number of similar episodes to retrieve
            temperature: LLM temperature
            force_include_nodes: Force include specific node IDs
            force_exclude_nodes: Force exclude specific node IDs

        Returns:
            ReasoningResult with answer, episode, and metadata
        """
        start_time = datetime.utcnow()
        logger.info(f"Reasoning for query: {query[:50]}...")

        # Check if v4.3 pipeline should be used
        if settings.query_enrichment_enabled:
            return await self._reason_v43(
                query=query,
                top_k_memories=top_k_memories,
                top_k_episodes=top_k_episodes,
                temperature=temperature,
                force_include_nodes=force_include_nodes,
                force_exclude_nodes=force_exclude_nodes,
                start_time=start_time,
            )

        # Standard v3 pipeline
        return await self._reason_standard(
            query=query,
            top_k_memories=top_k_memories,
            top_k_episodes=top_k_episodes,
            temperature=temperature,
            force_include_nodes=force_include_nodes,
            force_exclude_nodes=force_exclude_nodes,
            start_time=start_time,
        )

    async def _reason_standard(
        self,
        query: str,
        top_k_memories: int,
        top_k_episodes: int,
        temperature: float,
        force_include_nodes: list[str] | None,
        force_exclude_nodes: list[str] | None,
        start_time: datetime,
    ) -> ReasoningResult:
        """Standard v3 reasoning pipeline."""
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

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Reasoning complete: {episode.behavior_name} "
            f"(confidence: {synthesis.confidence:.0%}, time: {elapsed_ms:.0f}ms)"
        )

        return ReasoningResult(
            answer=synthesis.answer,
            episode=episode,
            retrieval=retrieval_result,
            synthesis=synthesis,
            confidence=synthesis.confidence,
            processing_time_ms=elapsed_ms,
        )

    async def _reason_v43(
        self,
        query: str,
        top_k_memories: int,
        top_k_episodes: int,
        temperature: float,
        force_include_nodes: list[str] | None,
        force_exclude_nodes: list[str] | None,
        start_time: datetime,
    ) -> ReasoningResult:
        """v4.3 enhanced reasoning pipeline with query enrichment."""
        # Import types for annotations
        from engram.query.enrichment import EnrichedQuery
        from engram.reasoning.confidence import ConfidenceResult, ResponseAction
        from engram.reasoning.intent_classifier import IntentResult, RetrievalDecision
        from engram.retrieval.crag import CRAGResult

        # Initialize v4.3 components
        await self._init_v43_components()

        intent: IntentResult | None = None
        crag_result: CRAGResult | None = None
        confidence_result: ConfidenceResult | None = None
        enriched: EnrichedQuery | None = None

        # 1. Intent Classification (can skip retrieval)
        if self._intent_classifier and settings.standard_intent_enabled:
            intent = await self._intent_classifier.classify(query)
            logger.debug(f"Intent: {intent.decision.value}, complexity: {intent.complexity.value}")

            # Handle no-retrieval case (greetings, meta questions)
            if intent.decision == RetrievalDecision.NO_RETRIEVE:
                answer = await self._direct_response(query)
                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return ReasoningResult(
                    answer=answer,
                    episode=None,
                    retrieval=None,
                    synthesis=None,
                    confidence=0.9,  # High confidence for direct responses
                    intent=intent,
                    used_v43_pipeline=True,
                    processing_time_ms=elapsed_ms,
                )

            # Handle clarification case
            if intent.decision == RetrievalDecision.CLARIFY:
                answer = await self._request_clarification(query)
                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return ReasoningResult(
                    answer=answer,
                    episode=None,
                    retrieval=None,
                    synthesis=None,
                    confidence=0.5,
                    intent=intent,
                    used_v43_pipeline=True,
                    processing_time_ms=elapsed_ms,
                )

        # 2. Query Enrichment
        if self._query_enrichment:
            enriched = await self._query_enrichment.enrich(query)
            logger.debug(
                f"Query enriched: type={enriched.query_type.value}, "
                f"complexity={enriched.complexity.value}"
            )

            # Handle out-of-scope
            if enriched.understanding and enriched.understanding.is_out_of_scope:
                answer = "К сожалению, этот вопрос выходит за рамки доступной базы знаний."
                if enriched.understanding.clarification_question:
                    answer += f"\n\nВозможно, вы имели в виду: {enriched.understanding.clarification_question}"
                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return ReasoningResult(
                    answer=answer,
                    episode=None,
                    retrieval=None,
                    synthesis=None,
                    confidence=0.3,
                    intent=intent,
                    enriched_query=enriched,
                    used_v43_pipeline=True,
                    processing_time_ms=elapsed_ms,
                )

        # 3. Multi-Query Retrieval
        if enriched:
            retrieval_result = await self.retrieval.retrieve_multi_query(
                query=query,
                bm25_expanded=enriched.bm25_expanded,
                semantic_rewrite=enriched.semantic_rewrite,
                hyde_document=enriched.hyde_document,
                top_k_memories=top_k_memories,
                top_k_episodes=top_k_episodes,
            )
        else:
            # Fallback to standard retrieval
            retrieval_result = await self.retrieval.retrieve(
                query=query,
                top_k_memories=top_k_memories,
                top_k_episodes=top_k_episodes,
            )

        # Apply force exclude
        if force_exclude_nodes:
            exclude_set = set(force_exclude_nodes)
            retrieval_result.memories = [
                m for m in retrieval_result.memories
                if m.memory.id not in exclude_set
            ]
            retrieval_result.activated_concepts = {
                k: v for k, v in retrieval_result.activated_concepts.items()
                if k not in exclude_set
            }

        # Apply force include
        if force_include_nodes:
            from engram.retrieval.hybrid_search import ScoredMemory
            existing_ids = {m.memory.id for m in retrieval_result.memories}
            for node_id in force_include_nodes:
                if node_id not in existing_ids and node_id.startswith("mem_"):
                    memory = await self.db.get_semantic_memory(node_id)
                    if memory:
                        retrieval_result.memories.insert(0, ScoredMemory(
                            memory=memory,
                            score=1.0,
                            sources=["F"],
                        ))
                elif node_id not in existing_ids and node_id.startswith("c_"):
                    if node_id not in retrieval_result.activated_concepts:
                        retrieval_result.activated_concepts[node_id] = 1.0

        # 4. Light CRAG (grade top N docs only)
        if self._crag_evaluator and settings.standard_crag_enabled:
            # Only grade top N to limit LLM calls
            top_memories = retrieval_result.memories[:settings.standard_crag_top_k]
            crag_result = await self._crag_evaluator.evaluate(
                query=query,
                memories=top_memories,
            )
            logger.debug(
                f"Light CRAG: {crag_result.quality.value}, "
                f"{len(crag_result.relevant_ids)}/{len(top_memories)} relevant"
            )

            # Handle query rewrite if all irrelevant (one retry only)
            if crag_result.needs_rewrite and crag_result.rewritten_query:
                logger.info(f"Query rewritten: {crag_result.rewritten_query[:50]}...")
                # Re-retrieve with rewritten query (standard retrieval for simplicity)
                retrieval_result = await self.retrieval.retrieve(
                    query=crag_result.rewritten_query,
                    top_k_memories=top_k_memories,
                    top_k_episodes=top_k_episodes,
                )
            elif crag_result.relevant_ids:
                # Filter to relevant only for synthesis
                relevant_set = set(crag_result.relevant_ids)
                # Keep relevant from top N, plus all below top N
                filtered_memories = []
                for i, m in enumerate(retrieval_result.memories):
                    if i < settings.standard_crag_top_k:
                        if m.memory.id in relevant_set:
                            filtered_memories.append(m)
                    else:
                        # Keep ungraded memories
                        filtered_memories.append(m)
                retrieval_result.memories = filtered_memories

        # 5. Synthesis
        synthesis = await self.synthesizer.synthesize(
            query=query,
            retrieval=retrieval_result,
            temperature=temperature,
        )
        answer = synthesis.answer

        # 6. Light Confidence (no extra LLM call)
        if self._confidence_calibrator and settings.standard_confidence_enabled:
            confidence_result = self._confidence_calibrator.calibrate_light(
                crag_result=crag_result,
                memories=retrieval_result.memories,
            )
            logger.debug(
                f"Light confidence: {confidence_result.level.value} "
                f"({confidence_result.combined_score:.2f})"
            )

            # Apply confidence action (add caveat or abstain)
            answer = self._confidence_calibrator.apply_action(answer, confidence_result)

            # Don't create episode if abstained
            if confidence_result.action == ResponseAction.ABSTAIN:
                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return ReasoningResult(
                    answer=answer,
                    episode=None,
                    retrieval=retrieval_result,
                    synthesis=synthesis,
                    confidence=confidence_result.combined_score,
                    intent=intent,
                    crag=crag_result,
                    confidence_result=confidence_result,
                    enriched_query=enriched,
                    used_v43_pipeline=True,
                    processing_time_ms=elapsed_ms,
                )

        # 7. Create episode
        episode = await self.episode_manager.create_episode(synthesis=synthesis)

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        final_confidence = (
            confidence_result.combined_score if confidence_result
            else synthesis.confidence
        )

        logger.info(
            f"v4.3 reasoning complete: {episode.behavior_name} "
            f"(confidence: {final_confidence:.0%}, time: {elapsed_ms:.0f}ms)"
        )

        return ReasoningResult(
            answer=answer,
            episode=episode,
            retrieval=retrieval_result,
            synthesis=synthesis,
            confidence=final_confidence,
            intent=intent,
            crag=crag_result,
            confidence_result=confidence_result,
            enriched_query=enriched,
            used_v43_pipeline=True,
            processing_time_ms=elapsed_ms,
        )

    async def _direct_response(self, query: str) -> str:
        """Generate direct response without retrieval."""
        prompt = f"""Ответь на вопрос напрямую, без использования внешних источников.

Вопрос: {query}

Ответ:"""
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.strip()

    async def _request_clarification(self, query: str) -> str:
        """Generate clarification request."""
        prompt = f"""Вопрос требует уточнения.

Исходный вопрос: {query}

Сформулируй уточняющий вопрос для пользователя:"""
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=256,
        )
        return response.strip()

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
    top_k: int = 200,
) -> ReasoningResult:
    """Convenience function for reasoning."""
    pipeline = ReasoningPipeline(db=db)
    return await pipeline.reason(query=query, top_k_memories=top_k)
