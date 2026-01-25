"""Agentic RAG pipeline integrating all v4 components.

Complete flow (v4.3 with query enrichment):
1. Intent Classification -> Decide whether to retrieve
2. Query Enrichment -> Generate multi-query variants (v4.3)
3. Multi-Query Retrieval -> Get relevant documents
4. CRAG Evaluation -> Grade documents, rewrite query if needed
5. IRCoT (for complex queries) -> Multi-hop reasoning
6. Generation with Citations -> Create response with sources
7. Self-RAG Validation -> Verify response is supported
8. NLI Hallucination Check -> Additional verification
9. Confidence Calibration -> Decide on caveats/abstention
10. Return AgenticResult

Each component can be enabled/disabled via configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.query.enrichment import QueryEnrichmentPipeline

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import EpisodicMemory
from engram.reasoning.citations import CitationManager, CitedResponse
from engram.reasoning.confidence import (
    ConfidenceCalibrator,
    ConfidenceResult,
    ResponseAction,
)
from engram.reasoning.episode_manager import EpisodeManager
from engram.reasoning.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
)
from engram.reasoning.intent_classifier import (
    IntentClassifier,
    IntentResult,
    RetrievalDecision,
)
from engram.reasoning.ircot import IRCoTReasoner, IRCoTResult
from engram.reasoning.self_rag import SelfRAGResult, SelfRAGValidator
from engram.reasoning.synthesizer import ResponseSynthesizer, SynthesisResult
from engram.retrieval.crag import CRAGEvaluator, CRAGResult
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.hybrid_search import ScoredMemory
from engram.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class AgenticMetadata:
    """Metadata about the agentic processing."""

    used_retrieval: bool = False
    used_ircot: bool = False
    used_crag: bool = False
    used_self_rag: bool = False
    used_nli: bool = False
    used_citations: bool = False
    iterations: int = 1
    abstained: bool = False
    query_rewritten: bool = False
    rewritten_query: str | None = None
    processing_time_ms: float = 0.0

    # v4.3 additions
    used_query_enrichment: bool = False
    enriched_query: str | None = None
    query_variants: list[str] | None = None
    enrichment_ms: float = 0.0


@dataclass
class AgenticResult:
    """Complete result from the agentic pipeline."""

    # Final answer
    answer: str

    # Episode created for this interaction
    episode: EpisodicMemory | None = None

    # Intermediate results (all optional, depend on which components ran)
    intent: IntentResult | None = None
    retrieval: RetrievalResult | None = None
    crag: CRAGResult | None = None
    ircot: IRCoTResult | None = None
    synthesis: SynthesisResult | None = None
    cited_response: CitedResponse | None = None
    self_rag: SelfRAGResult | None = None
    hallucination: HallucinationResult | None = None
    confidence: ConfidenceResult | None = None

    # Metadata
    metadata: AgenticMetadata = field(default_factory=AgenticMetadata)

    @property
    def episode_id(self) -> str | None:
        """Get episode ID for feedback."""
        return self.episode.id if self.episode else None

    @property
    def memories_used(self) -> list[ScoredMemory]:
        """Get all memories used."""
        if self.retrieval:
            return self.retrieval.memories
        return []


# Direct response for no-retrieval queries
DIRECT_RESPONSE_PROMPT = """Ответь на вопрос напрямую, без использования внешних источников.

Вопрос: {query}

Ответ:"""

# Clarification request
CLARIFICATION_PROMPT = """Вопрос требует уточнения.

Исходный вопрос: {query}

Сформулируй уточняющий вопрос для пользователя:"""


class AgenticPipeline:
    """
    Complete agentic RAG pipeline for Engram v4.

    Orchestrates all v4 components with configurable feature flags.

    v4.3 additions:
    - Query enrichment stage after intent classification
    - Multi-query retrieval using enriched variants
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
        self.intent_classifier = IntentClassifier(llm_client=self.llm)
        self.retrieval = RetrievalPipeline(
            db=db,
            embedding_service=self.embeddings,
        )
        self.crag_evaluator = CRAGEvaluator(llm_client=self.llm)
        self.ircot_reasoner = IRCoTReasoner(
            db=db,
            llm_client=self.llm,
            embedding_service=self.embeddings,
        )
        self.synthesizer = ResponseSynthesizer(llm_client=self.llm)
        self.citation_manager = CitationManager(llm_client=self.llm)
        self.self_rag_validator = SelfRAGValidator(llm_client=self.llm)
        self.hallucination_detector = HallucinationDetector(llm_client=self.llm)
        self.confidence_calibrator = ConfidenceCalibrator(llm_client=self.llm)
        self.episode_manager = EpisodeManager(
            db=db,
            embedding_service=self.embeddings,
        )

        # v4.3: Query enrichment (initialized lazily)
        self._query_enrichment: "QueryEnrichmentPipeline | None" = None
        self._enrichment_initialized = False

    async def _init_query_enrichment(self) -> None:
        """Lazily initialize query enrichment pipeline."""
        if self._enrichment_initialized:
            return

        if settings.query_enrichment_enabled:
            from engram.query.enrichment import QueryEnrichmentPipeline
            self._query_enrichment = QueryEnrichmentPipeline(
                db=self.db,
                llm_client=self.llm,
            )
            await self._query_enrichment.initialize()
            logger.info("Query enrichment pipeline initialized for agentic mode")

        self._enrichment_initialized = True

    async def reason(
        self,
        query: str,
        top_k_memories: int = 200,
        temperature: float = 0.4,
        force_include_nodes: list[str] | None = None,
        force_exclude_nodes: list[str] | None = None,
    ) -> AgenticResult:
        """
        Execute the full agentic pipeline.

        Args:
            query: User query
            top_k_memories: Number of memories to retrieve
            temperature: LLM temperature
            force_include_nodes: Force include specific node IDs
            force_exclude_nodes: Force exclude specific node IDs

        Returns:
            AgenticResult with answer and all intermediate results
        """
        start_time = datetime.utcnow()
        metadata = AgenticMetadata()

        logger.info(f"Agentic reasoning for: {query[:50]}...")

        # 1. Intent Classification
        intent: IntentResult | None = None
        if settings.intent_classification_enabled:
            intent = await self.intent_classifier.classify(query)
            logger.info(f"Intent: {intent.decision.value}, complexity: {intent.complexity.value}")

            # Handle no-retrieval case
            if intent.decision == RetrievalDecision.NO_RETRIEVE:
                answer = await self._direct_response(query)
                metadata.processing_time_ms = self._elapsed_ms(start_time)
                return AgenticResult(
                    answer=answer,
                    intent=intent,
                    metadata=metadata,
                )

            # Handle clarification case
            if intent.decision == RetrievalDecision.CLARIFY:
                answer = await self._request_clarification(query)
                metadata.processing_time_ms = self._elapsed_ms(start_time)
                return AgenticResult(
                    answer=answer,
                    intent=intent,
                    metadata=metadata,
                )

        # 2. Query Enrichment (v4.3)
        enriched = None
        if settings.query_enrichment_enabled:
            await self._init_query_enrichment()
            if self._query_enrichment:
                enrichment_start = datetime.utcnow()
                from engram.query.enrichment import EnrichedQuery
                enriched: EnrichedQuery | None = await self._query_enrichment.enrich(query)
                metadata.enrichment_ms = self._elapsed_ms(enrichment_start)
                metadata.used_query_enrichment = True

                if enriched:
                    metadata.enriched_query = enriched.semantic_rewrite or query
                    metadata.query_variants = enriched.get_all_variants()

                    # Handle out-of-scope from enrichment
                    if enriched.understanding and enriched.understanding.is_out_of_scope:
                        answer = "К сожалению, этот вопрос выходит за рамки доступной базы знаний."
                        if enriched.understanding.clarification_question:
                            answer += f"\n\n{enriched.understanding.clarification_question}"
                        metadata.processing_time_ms = self._elapsed_ms(start_time)
                        return AgenticResult(
                            answer=answer,
                            intent=intent,
                            metadata=metadata,
                        )

                logger.info(
                    f"Query enriched: {len(enriched.get_all_variants())} variants, "
                    f"{metadata.enrichment_ms:.0f}ms"
                )

        # 3. Retrieval (with multi-query if enriched)
        metadata.used_retrieval = True
        if enriched:
            retrieval_result = await self.retrieval.retrieve_multi_query(
                query=query,
                bm25_expanded=enriched.bm25_expanded,
                semantic_rewrite=enriched.semantic_rewrite,
                hyde_document=enriched.hyde_document,
                top_k_memories=top_k_memories,
            )
        else:
            retrieval_result = await self.retrieval.retrieve(
                query=query,
                top_k_memories=top_k_memories,
            )

        # Apply force exclude
        if force_exclude_nodes:
            exclude_set = set(force_exclude_nodes)
            retrieval_result.memories = [
                m for m in retrieval_result.memories
                if m.memory.id not in exclude_set
            ]

        # Apply force include
        if force_include_nodes:
            await self._apply_force_include(
                force_include_nodes, retrieval_result
            )

        # 4. CRAG Evaluation
        crag_result: CRAGResult | None = None
        if settings.crag_enabled:
            metadata.used_crag = True
            crag_result = await self.crag_evaluator.evaluate(
                query=query,
                memories=retrieval_result.memories,
            )
            logger.info(f"CRAG: {crag_result.quality.value}, {len(crag_result.relevant_ids)}/{len(retrieval_result.memories)} relevant")

            # Handle query rewrite if needed
            if crag_result.needs_rewrite and crag_result.rewritten_query:
                metadata.query_rewritten = True
                metadata.rewritten_query = crag_result.rewritten_query
                logger.info(f"Query rewritten to: {crag_result.rewritten_query[:50]}...")

                # Re-retrieve with rewritten query
                retrieval_result = await self.retrieval.retrieve(
                    query=crag_result.rewritten_query,
                    top_k_memories=top_k_memories,
                )

            # Filter to relevant documents only
            if crag_result.relevant_ids:
                retrieval_result.memories = self.crag_evaluator.filter_relevant(
                    retrieval_result.memories, crag_result
                )

        # 5. IRCoT for complex queries
        ircot_result: IRCoTResult | None = None
        synthesis: SynthesisResult | None = None
        answer: str = ""

        use_ircot = (
            settings.ircot_enabled and
            intent is not None and
            intent.should_use_ircot
        )

        if use_ircot:
            metadata.used_ircot = True
            ircot_result = await self.ircot_reasoner.reason(
                query=query,
                initial_memories=retrieval_result.memories,
            )
            answer = ircot_result.final_answer
            logger.info(f"IRCoT: {ircot_result.step_count} steps, {ircot_result.paragraph_count} paragraphs")
        else:
            # 6. Standard synthesis
            synthesis = await self.synthesizer.synthesize(
                query=query,
                retrieval=retrieval_result,
                temperature=temperature,
            )
            answer = synthesis.answer

        # 7. Add citations
        cited_response: CitedResponse | None = None
        if settings.citations_enabled:
            metadata.used_citations = True
            cited_response = await self.citation_manager.add_citations(
                response=answer,
                query=query,
                memories=retrieval_result.memories,
            )
            answer = cited_response.text
            if cited_response.references:
                answer += cited_response.references

        # 8. Self-RAG validation
        self_rag_result: SelfRAGResult | None = None
        if settings.self_rag_enabled:
            metadata.used_self_rag = True
            self_rag_result = await self.self_rag_validator.validate_and_refine(
                query=query,
                initial_response=answer,
                memories=retrieval_result.memories,
            )
            metadata.iterations = self_rag_result.iteration_count
            answer = self_rag_result.final_response

            if self_rag_result.abstained:
                metadata.abstained = True
                logger.info("Self-RAG: abstained")

        # 9. NLI hallucination check
        hallucination_result: HallucinationResult | None = None
        if settings.nli_enabled and not metadata.abstained:
            metadata.used_nli = True
            hallucination_result = await self.hallucination_detector.detect(
                response=answer,
                memories=retrieval_result.memories,
            )
            logger.info(f"NLI: faithfulness={hallucination_result.faithfulness_score:.2f}")

        # 10. Confidence calibration
        confidence_result: ConfidenceResult | None = None
        if not metadata.abstained:
            confidence_result = await self.confidence_calibrator.calibrate(
                query=query,
                response=answer,
                memories=retrieval_result.memories,
                crag_result=crag_result,
                self_rag_result=self_rag_result,
                hallucination_result=hallucination_result,
            )
            logger.info(f"Confidence: {confidence_result.level.value} ({confidence_result.combined_score:.2f})")

            # Apply confidence action
            answer = self.confidence_calibrator.apply_action(answer, confidence_result)
            if confidence_result.action == ResponseAction.ABSTAIN:
                metadata.abstained = True

        # 11. Create episode
        episode: EpisodicMemory | None = None
        if synthesis and not metadata.abstained:
            episode = await self.episode_manager.create_episode(synthesis)

        metadata.processing_time_ms = self._elapsed_ms(start_time)

        logger.info(
            f"Agentic reasoning complete: "
            f"abstained={metadata.abstained}, "
            f"iterations={metadata.iterations}, "
            f"time={metadata.processing_time_ms:.0f}ms"
        )

        return AgenticResult(
            answer=answer,
            episode=episode,
            intent=intent,
            retrieval=retrieval_result,
            crag=crag_result,
            ircot=ircot_result,
            synthesis=synthesis,
            cited_response=cited_response,
            self_rag=self_rag_result,
            hallucination=hallucination_result,
            confidence=confidence_result,
            metadata=metadata,
        )

    async def _direct_response(self, query: str) -> str:
        """Generate direct response without retrieval."""
        prompt = DIRECT_RESPONSE_PROMPT.format(query=query)
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.strip()

    async def _request_clarification(self, query: str) -> str:
        """Generate clarification request."""
        prompt = CLARIFICATION_PROMPT.format(query=query)
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=256,
        )
        return response.strip()

    async def _apply_force_include(
        self,
        force_include_nodes: list[str],
        retrieval_result: RetrievalResult,
    ) -> None:
        """Apply force include for specific nodes."""
        existing_ids = {m.memory.id for m in retrieval_result.memories}

        for node_id in force_include_nodes:
            if node_id in existing_ids:
                continue

            if node_id.startswith("mem_"):
                memory = await self.db.get_semantic_memory(node_id)
                if memory:
                    retrieval_result.memories.insert(0, ScoredMemory(
                        memory=memory,
                        score=1.0,
                        sources=["F"],  # F = Forced
                    ))
            elif node_id.startswith("c_"):
                if node_id not in retrieval_result.activated_concepts:
                    retrieval_result.activated_concepts[node_id] = 1.0

    def _elapsed_ms(self, start: datetime) -> float:
        """Calculate elapsed milliseconds."""
        return (datetime.utcnow() - start).total_seconds() * 1000


async def agentic_reason(
    db: Neo4jClient,
    query: str,
    top_k: int = 200,
) -> AgenticResult:
    """Convenience function for agentic reasoning."""
    pipeline = AgenticPipeline(db=db)
    return await pipeline.reason(query=query, top_k_memories=top_k)
