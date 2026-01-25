"""API routes for Engram.

Provides:
- OpenAI-compatible /v1/chat/completions
- /v1/feedback for learning
- Admin endpoints for health, calibration, review
"""

import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from engram.config import settings
from engram.learning import FeedbackHandler, FeedbackType
from engram.reasoning.agentic_pipeline import AgenticPipeline
from engram.reasoning.pipeline import ReasoningPipeline
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# OpenAI-Compatible Models
# ============================================================================


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "engram"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False

    # Engram-specific options
    top_k_memories: int = Field(default=100, ge=1, le=200)
    top_k_episodes: int = Field(default=3, ge=0, le=10)

    # v4 Agentic mode
    agentic: bool = Field(
        default=False,
        description="Use v4 agentic pipeline with intent classification, CRAG, Self-RAG, etc."
    )

    # Debug options
    debug: bool = False
    force_include_nodes: list[str] = []
    force_exclude_nodes: list[str] = []


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class DebugMemoryInfo(BaseModel):
    """Debug info for a retrieved memory."""
    id: str
    content: str
    score: float
    sources: list[str] = []  # V=Vector, B=BM25, G=Graph, F=Forced
    included: bool = True


class DebugConceptInfo(BaseModel):
    """Debug info for an activated concept."""
    id: str
    name: str
    activation: float
    hop: int = 0
    included: bool = True


class DebugInfo(BaseModel):
    """Debug information for chat response."""
    retrieved_memories: list[DebugMemoryInfo] = []
    activated_concepts: list[DebugConceptInfo] = []
    query_concepts: list[str] = []
    thresholds: dict = {}


class QueryEnrichmentDebugInfo(BaseModel):
    """Debug information for v4.3 query enrichment."""

    enabled: bool = False
    query_type: str | None = None  # factual, procedural, person, comparison, etc.
    complexity: str | None = None  # simple, multi_hop, ambiguous, out_of_scope
    variants: list[str] = []  # All query variants generated
    bm25_expanded: str | None = None
    semantic_rewrite: str | None = None
    hyde_document: str | None = None
    enrichment_ms: float | None = None


class AgenticDebugInfo(BaseModel):
    """Debug information for v4 agentic pipeline."""

    # Intent classification
    intent_decision: str | None = None  # retrieve, no_retrieve, clarify
    intent_complexity: str | None = None  # simple, moderate, complex
    intent_confidence: float | None = None

    # v4.3 Query Enrichment
    query_enrichment: QueryEnrichmentDebugInfo | None = None

    # CRAG
    crag_quality: str | None = None  # correct, incorrect, ambiguous
    crag_relevant_ratio: float | None = None
    query_rewritten: bool = False
    rewritten_query: str | None = None

    # IRCoT
    used_ircot: bool = False
    ircot_steps: int | None = None
    ircot_paragraphs: int | None = None

    # Self-RAG
    self_rag_iterations: int | None = None
    self_rag_support_level: str | None = None  # fully_supported, partially_supported, not_supported

    # NLI
    nli_faithfulness: float | None = None
    nli_supported_claims: int | None = None
    nli_contradicted_claims: int | None = None

    # Confidence
    confidence_level: str | None = None  # high, medium, low, very_low
    confidence_action: str | None = None  # respond_normally, respond_with_caveat, abstain
    confidence_score: float | None = None

    # Citations
    citations_count: int | None = None
    citations_verified: int | None = None

    # Metadata
    abstained: bool = False
    processing_time_ms: float | None = None


class SourceDocument(BaseModel):
    """A source document used in the response."""
    title: str
    url: str | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage

    # Engram-specific metadata
    episode_id: str | None = None
    confidence: float | None = None
    concepts_activated: list[str] = []  # Concept IDs
    memories_used: list[str] = []  # Memory IDs
    memories_count: int = 0
    sources_used: list[SourceDocument] = Field(
        default_factory=list,
        description="Deduplicated list of source documents (title + URL) used in the answer"
    )

    # Debug info (only when debug=true)
    debug_info: DebugInfo | None = None

    # v4 Agentic debug info (only when agentic=true and debug=true)
    agentic_debug: AgenticDebugInfo | None = None


# ============================================================================
# Feedback Models
# ============================================================================


class FeedbackRequest(BaseModel):
    """Feedback request for an episode."""

    episode_id: str
    feedback: FeedbackType
    correction_text: str | None = None


class FeedbackResponse(BaseModel):
    """Response after processing feedback."""

    success: bool
    feedback_type: FeedbackType
    episode_id: str

    # Positive feedback results
    memories_strengthened: int = 0
    concepts_strengthened: int = 0
    consolidation_triggered: bool = False
    reflection_triggered: bool = False

    # Negative feedback results
    memories_weakened: int = 0
    alternative_answer: str | None = None

    # Correction results
    correction_memory_id: str | None = None


# ============================================================================
# Admin Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    neo4j_connected: bool
    version: str = "0.1.0"


class StatsResponse(BaseModel):
    """System statistics."""

    concepts_count: int
    semantic_memories_count: int
    episodic_memories_count: int
    documents_count: int


class ConceptInfo(BaseModel):
    """Brief concept information."""

    id: str
    name: str
    type: str
    activation_count: int


class MemoryInfo(BaseModel):
    """Brief memory information."""

    id: str
    content: str
    memory_type: str
    importance: float
    strength: float
    access_count: int


class EpisodeInfo(BaseModel):
    """Brief episode information."""

    id: str
    query: str
    behavior_name: str
    feedback: str | None
    success_count: int
    failure_count: int


# ============================================================================
# Helper Functions
# ============================================================================


def get_db(request: Request) -> Neo4jClient:
    """Get database from app state."""
    return request.app.state.db


# ============================================================================
# Chat Completion Endpoint
# ============================================================================


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Takes conversation messages and generates a response using
    the Engram reasoning pipeline.
    """
    # Note: streaming not implemented, will return non-streaming response
    # Open WebUI handles this gracefully

    db = get_db(request)

    # Extract user query from messages (last user message)
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in request",
        )

    query = user_messages[-1].content

    # Run reasoning pipeline
    try:
        # Use agentic pipeline if requested
        if body.agentic:
            return await _handle_agentic_request(db, query, body)

        # Standard v3 pipeline with hybrid search
        pipeline = ReasoningPipeline(db=db)

        result = await pipeline.reason(
            query=query,
            top_k_memories=body.top_k_memories,
            top_k_episodes=body.top_k_episodes,
            temperature=body.temperature,
            force_include_nodes=body.force_include_nodes if body.force_include_nodes else None,
            force_exclude_nodes=body.force_exclude_nodes if body.force_exclude_nodes else None,
        )

        # Fetch source documents for memories used
        source_documents = await db.get_source_documents_for_memories(
            memory_ids=result.synthesis.memories_used
        )

        # Build deduplicated sources list with title + URL
        sources_used: list[SourceDocument] = []
        seen_urls: set[str] = set()

        for doc in source_documents:
            # Use source_path as URL (should be Confluence URL)
            url = doc.source_path

            # Skip duplicates (by URL or title)
            dedup_key = url or doc.title
            if dedup_key in seen_urls:
                continue
            seen_urls.add(dedup_key)

            # Get title (fallback to filename from path if no title)
            title = doc.title
            if not title and url:
                # Extract page name from URL as fallback
                title = url.split("/")[-1].replace("-", " ").replace("+", " ")

            if title:  # Only add if we have at least a title
                sources_used.append(SourceDocument(
                    title=title,
                    url=url,
                ))

        # Append confidence and sources to response content for visibility in Open WebUI
        memories_count = len(result.synthesis.memories_used)
        confidence_pct = int(result.confidence * 100)

        # Format sources for display (markdown links) - limit to top 5
        if sources_used:
            sources_used = sources_used[:5]
            sources_lines = []
            for src in sources_used:
                if src.url:
                    sources_lines.append(f"• [{src.title}]({src.url})")
                else:
                    sources_lines.append(f"• {src.title}")
            sources_text = "\n".join(sources_lines)
            stats_footer = f"\n\n---\n**Confidence: {confidence_pct}%**\n\n**Sources:**\n{sources_text}"
        else:
            stats_footer = f"\n\n---\n**Confidence: {confidence_pct}%**"

        content_with_stats = result.answer + stats_footer

        # Build debug info if requested
        debug_info = None
        if body.debug:
            debug_memories = []
            for sm in result.retrieval.memories[:30]:
                debug_memories.append(DebugMemoryInfo(
                    id=sm.memory.id,
                    content=sm.memory.content[:100] + "..." if len(sm.memory.content) > 100 else sm.memory.content,
                    score=sm.score,
                    sources=sm.sources,
                    included=sm.memory.id in result.synthesis.memories_used,
                ))

            debug_concepts = []
            for concept_id, activation in list(result.retrieval.activated_concepts.items())[:30]:
                # Extract concept name from id (e.g., c_docker_1a8e -> docker)
                parts = concept_id.split("_")
                name = "_".join(parts[1:-1]) if len(parts) > 2 else concept_id
                debug_concepts.append(DebugConceptInfo(
                    id=concept_id,
                    name=name,
                    activation=activation,
                    hop=0,  # Would need more info from activation result
                    included=concept_id in result.synthesis.concepts_activated,
                ))

            # Build thresholds dict
            thresholds = {"top_k_memories": body.top_k_memories}

            # Add v4.3 info if used
            if result.used_v43_pipeline:
                thresholds["v43_pipeline"] = True
                if result.intent:
                    thresholds["intent_decision"] = result.intent.decision.value
                if result.enriched_query:
                    thresholds["query_type"] = result.enriched_query.query_type.value
                    thresholds["query_complexity"] = result.enriched_query.complexity.value
                    thresholds["query_variants"] = len(result.enriched_query.get_all_variants())
                if result.crag:
                    thresholds["crag_quality"] = result.crag.quality.value
                    thresholds["crag_relevant_ratio"] = result.crag.relevant_ratio
                if result.confidence_result:
                    thresholds["confidence_level"] = result.confidence_result.level.value
                    thresholds["confidence_score"] = result.confidence_result.combined_score

            debug_info = DebugInfo(
                retrieved_memories=debug_memories,
                activated_concepts=debug_concepts,
                query_concepts=[c.name for c in result.retrieval.query_concepts],
                thresholds=thresholds,
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content_with_stats,
                    ),
                )
            ],
            usage=ChatCompletionUsage(),
            episode_id=result.episode_id,
            confidence=result.confidence,
            concepts_activated=result.synthesis.concepts_activated[:50],  # Concept IDs
            memories_used=result.synthesis.memories_used[:50],  # Memory IDs
            memories_count=memories_count,
            sources_used=sources_used,
            debug_info=debug_info,
        )

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {str(e)}",
        )


# ============================================================================
# Agentic Pipeline Helper
# ============================================================================


async def _handle_agentic_request(
    db: Neo4jClient,
    query: str,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Handle request using v4 agentic pipeline."""
    pipeline = AgenticPipeline(db=db)
    result = await pipeline.reason(
        query=query,
        top_k_memories=body.top_k_memories,
        temperature=body.temperature,
        force_include_nodes=body.force_include_nodes if body.force_include_nodes else None,
        force_exclude_nodes=body.force_exclude_nodes if body.force_exclude_nodes else None,
    )

    # Get memories used for source documents
    memories_used = []
    if result.synthesis:
        memories_used = result.synthesis.memories_used
    elif result.retrieval:
        memories_used = [m.memory.id for m in result.retrieval.memories[:10]]

    # Fetch source documents
    source_documents = await db.get_source_documents_for_memories(memory_ids=memories_used)

    # Build deduplicated sources list
    sources_used: list[SourceDocument] = []
    seen_urls: set[str] = set()

    for doc in source_documents:
        url = doc.source_path
        dedup_key = url or doc.title
        if dedup_key in seen_urls:
            continue
        seen_urls.add(dedup_key)

        title = doc.title
        if not title and url:
            title = url.split("/")[-1].replace("-", " ").replace("+", " ")

        if title:
            sources_used.append(SourceDocument(title=title, url=url))

    # Build response content
    content = result.answer

    # Add sources if available - limit to top 5
    if sources_used:
        sources_used = sources_used[:5]
        sources_lines = []
        for src in sources_used:
            if src.url:
                sources_lines.append(f"• [{src.title}]({src.url})")
            else:
                sources_lines.append(f"• {src.title}")
        sources_text = "\n".join(sources_lines)

        # Get confidence info
        confidence_str = ""
        if result.confidence:
            conf_pct = int(result.confidence.combined_score * 100)
            confidence_str = f"**Confidence: {conf_pct}% ({result.confidence.level.value})**\n\n"

        content += f"\n\n---\n{confidence_str}**Sources:**\n{sources_text}"
    elif result.confidence:
        conf_pct = int(result.confidence.combined_score * 100)
        content += f"\n\n---\n**Confidence: {conf_pct}% ({result.confidence.level.value})**"

    # Build agentic debug info
    agentic_debug = None
    if body.debug:
        agentic_debug = AgenticDebugInfo(
            processing_time_ms=result.metadata.processing_time_ms,
            abstained=result.metadata.abstained,
            used_ircot=result.metadata.used_ircot,
            query_rewritten=result.metadata.query_rewritten,
            rewritten_query=result.metadata.rewritten_query,
        )

        if result.intent:
            agentic_debug.intent_decision = result.intent.decision.value
            agentic_debug.intent_complexity = result.intent.complexity.value
            agentic_debug.intent_confidence = result.intent.confidence

        # v4.3 Query Enrichment
        if result.metadata.used_query_enrichment:
            agentic_debug.query_enrichment = QueryEnrichmentDebugInfo(
                enabled=True,
                variants=result.metadata.query_variants or [],
                enrichment_ms=result.metadata.enrichment_ms,
            )
            # Try to get more details from retrieval result
            if result.retrieval and result.retrieval.query_variants:
                agentic_debug.query_enrichment.bm25_expanded = result.retrieval.query_variants.get("bm25_expanded")
                agentic_debug.query_enrichment.semantic_rewrite = result.retrieval.query_variants.get("semantic_rewrite")
                agentic_debug.query_enrichment.hyde_document = result.retrieval.query_variants.get("hyde")

        if result.crag:
            agentic_debug.crag_quality = result.crag.quality.value
            agentic_debug.crag_relevant_ratio = result.crag.relevant_ratio

        if result.ircot:
            agentic_debug.ircot_steps = result.ircot.step_count
            agentic_debug.ircot_paragraphs = result.ircot.paragraph_count

        if result.self_rag:
            agentic_debug.self_rag_iterations = result.self_rag.iteration_count
            agentic_debug.self_rag_support_level = result.self_rag.final_validation.support_level.value

        if result.hallucination:
            agentic_debug.nli_faithfulness = result.hallucination.faithfulness_score
            agentic_debug.nli_supported_claims = result.hallucination.supported_count
            agentic_debug.nli_contradicted_claims = result.hallucination.contradicted_count

        if result.confidence:
            agentic_debug.confidence_level = result.confidence.level.value
            agentic_debug.confidence_action = result.confidence.action.value
            agentic_debug.confidence_score = result.confidence.combined_score

        if result.cited_response:
            agentic_debug.citations_count = len(result.cited_response.citations)
            agentic_debug.citations_verified = result.cited_response.verified_count

    # Build standard debug info
    debug_info = None
    if body.debug and result.retrieval:
        debug_memories = []
        for sm in result.retrieval.memories[:30]:
            debug_memories.append(DebugMemoryInfo(
                id=sm.memory.id,
                content=sm.memory.content[:100] + "..." if len(sm.memory.content) > 100 else sm.memory.content,
                score=sm.score,
                sources=sm.sources,
                included=sm.memory.id in memories_used,
            ))

        debug_concepts = []
        for concept_id, activation in list(result.retrieval.activated_concepts.items())[:30]:
            parts = concept_id.split("_")
            name = "_".join(parts[1:-1]) if len(parts) > 2 else concept_id
            concepts_activated = result.synthesis.concepts_activated if result.synthesis else []
            debug_concepts.append(DebugConceptInfo(
                id=concept_id,
                name=name,
                activation=activation,
                hop=0,
                included=concept_id in concepts_activated,
            ))

        debug_info = DebugInfo(
            retrieved_memories=debug_memories,
            activated_concepts=debug_concepts,
            query_concepts=[c.name for c in result.retrieval.query_concepts],
            thresholds={"top_k_memories": body.top_k_memories, "agentic": True},
        )

    # Calculate confidence
    confidence = None
    if result.confidence:
        confidence = result.confidence.combined_score
    elif result.synthesis:
        confidence = result.synthesis.confidence

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
            )
        ],
        usage=ChatCompletionUsage(),
        episode_id=result.episode_id,
        confidence=confidence,
        concepts_activated=result.synthesis.concepts_activated[:50] if result.synthesis else [],
        memories_used=memories_used[:50],
        memories_count=len(memories_used),
        sources_used=sources_used,
        debug_info=debug_info,
        agentic_debug=agentic_debug,
    )


# ============================================================================
# Feedback Endpoint
# ============================================================================


@router.post("/v1/feedback", response_model=FeedbackResponse, include_in_schema=False)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
) -> FeedbackResponse:
    """
    Submit feedback for an episode.

    Feedback types:
    - positive: Strengthens memories and concepts, may trigger consolidation
    - negative: Weakens memories, triggers re-reasoning
    - correction: Creates new memory from correction text
    """
    db = get_db(request)

    try:
        handler = FeedbackHandler(db=db)
        result = await handler.handle_feedback(
            episode_id=body.episode_id,
            feedback=body.feedback,
            correction_text=body.correction_text,
        )

        response = FeedbackResponse(
            success=result.success,
            feedback_type=result.feedback_type,
            episode_id=result.episode_id,
        )

        if result.feedback_type == "positive":
            response.memories_strengthened = result.memories_strengthened
            response.concepts_strengthened = result.concepts_strengthened
            response.consolidation_triggered = (
                result.consolidation.should_consolidate
                if result.consolidation
                else False
            )
            response.reflection_triggered = (
                result.reflection.triggered if result.reflection else False
            )

        elif result.feedback_type == "negative":
            response.memories_weakened = result.memories_weakened
            if result.re_reasoning:
                response.alternative_answer = result.re_reasoning.answer

        elif result.feedback_type == "correction":
            if result.correction_memory:
                response.correction_memory_id = result.correction_memory.id

        return response

    except Exception as e:
        logger.exception(f"Error processing feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Feedback processing failed: {str(e)}",
        )


# ============================================================================
# Admin Endpoints
# ============================================================================


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint."""
    try:
        db = get_db(request)
        # Try a simple query to verify connection
        await db.execute_query("RETURN 1 as n")
        neo4j_connected = True
    except Exception:
        neo4j_connected = False

    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        neo4j_connected=neo4j_connected,
    )


@router.get("/models", include_in_schema=False)
@router.get("/v1/models", include_in_schema=False)
async def list_models() -> dict:
    """
    List available models (OpenAI compatibility).

    Returns Engram as the only available model.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "engram",
                "object": "model",
                "created": 1700000000,
                "owned_by": "engram",
            }
        ],
    }


@router.get("/admin/stats", response_model=StatsResponse, include_in_schema=False)
async def get_stats(request: Request) -> StatsResponse:
    """Get system statistics."""
    db = get_db(request)

    try:
        result = await db.execute_query(
            """
            MATCH (c:Concept) WITH count(c) as concepts
            MATCH (s:SemanticMemory) WITH concepts, count(s) as semantic
            MATCH (e:EpisodicMemory) WITH concepts, semantic, count(e) as episodic
            MATCH (d:Document) WITH concepts, semantic, episodic, count(d) as docs
            RETURN concepts, semantic, episodic, docs
            """
        )

        if result:
            row = result[0]
            return StatsResponse(
                concepts_count=row["concepts"],
                semantic_memories_count=row["semantic"],
                episodic_memories_count=row["episodic"],
                documents_count=row["docs"],
            )

        return StatsResponse(
            concepts_count=0,
            semantic_memories_count=0,
            episodic_memories_count=0,
            documents_count=0,
        )

    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}",
        )


@router.get("/admin/concepts", response_model=list[ConceptInfo], include_in_schema=False)
async def list_concepts(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[ConceptInfo]:
    """List concepts with pagination."""
    db = get_db(request)

    try:
        result = await db.execute_query(
            """
            MATCH (c:Concept)
            RETURN c.id as id, c.name as name, c.type as type,
                   coalesce(c.activation_count, 0) as activation_count
            ORDER BY c.activation_count DESC
            SKIP $offset LIMIT $limit
            """,
            offset=offset,
            limit=limit,
        )

        return [
            ConceptInfo(
                id=row["id"],
                name=row["name"],
                type=row["type"] or "unknown",
                activation_count=row["activation_count"],
            )
            for row in result
        ]

    except Exception as e:
        logger.exception(f"Error listing concepts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list concepts: {str(e)}",
        )


@router.get("/admin/memories", response_model=list[MemoryInfo], include_in_schema=False)
async def list_memories(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    memory_type: str | None = None,
) -> list[MemoryInfo]:
    """List semantic memories with pagination and optional type filter."""
    db = get_db(request)

    try:
        if memory_type:
            query = """
                MATCH (s:SemanticMemory)
                WHERE s.memory_type = $memory_type
                RETURN s.id as id, s.content as content, s.memory_type as memory_type,
                       s.importance as importance, s.strength as strength,
                       coalesce(s.access_count, 0) as access_count
                ORDER BY s.importance DESC
                SKIP $offset LIMIT $limit
            """
            result = await db.execute_query(
                query, memory_type=memory_type, offset=offset, limit=limit
            )
        else:
            query = """
                MATCH (s:SemanticMemory)
                RETURN s.id as id, s.content as content, s.memory_type as memory_type,
                       s.importance as importance, s.strength as strength,
                       coalesce(s.access_count, 0) as access_count
                ORDER BY s.importance DESC
                SKIP $offset LIMIT $limit
            """
            result = await db.execute_query(query, offset=offset, limit=limit)

        return [
            MemoryInfo(
                id=row["id"],
                content=row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                memory_type=row["memory_type"],
                importance=row["importance"],
                strength=row["strength"],
                access_count=row["access_count"],
            )
            for row in result
        ]

    except Exception as e:
        logger.exception(f"Error listing memories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list memories: {str(e)}",
        )


@router.get("/admin/episodes", response_model=list[EpisodeInfo], include_in_schema=False)
async def list_episodes(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    feedback: str | None = None,
) -> list[EpisodeInfo]:
    """List episodic memories with pagination and optional feedback filter."""
    db = get_db(request)

    try:
        if feedback:
            query = """
                MATCH (e:EpisodicMemory)
                WHERE e.feedback = $feedback
                RETURN e.id as id, e.query as query, e.behavior_name as behavior_name,
                       e.feedback as feedback, e.success_count as success_count,
                       e.failure_count as failure_count
                ORDER BY e.created_at DESC
                SKIP $offset LIMIT $limit
            """
            result = await db.execute_query(
                query, feedback=feedback, offset=offset, limit=limit
            )
        else:
            query = """
                MATCH (e:EpisodicMemory)
                RETURN e.id as id, e.query as query, e.behavior_name as behavior_name,
                       e.feedback as feedback, e.success_count as success_count,
                       e.failure_count as failure_count
                ORDER BY e.created_at DESC
                SKIP $offset LIMIT $limit
            """
            result = await db.execute_query(query, offset=offset, limit=limit)

        return [
            EpisodeInfo(
                id=row["id"],
                query=row["query"][:100] + "..." if len(row["query"]) > 100 else row["query"],
                behavior_name=row["behavior_name"] or "unknown",
                feedback=row["feedback"],
                success_count=row["success_count"] or 0,
                failure_count=row["failure_count"] or 0,
            )
            for row in result
        ]

    except Exception as e:
        logger.exception(f"Error listing episodes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list episodes: {str(e)}",
        )


@router.delete("/admin/episodes/{episode_id}", include_in_schema=False)
async def delete_episode(
    request: Request,
    episode_id: str,
) -> dict:
    """Delete an episode by ID."""
    db = get_db(request)

    try:
        result = await db.execute_query(
            """
            MATCH (e:EpisodicMemory {id: $id})
            DETACH DELETE e
            RETURN count(*) as deleted
            """,
            id=episode_id,
        )

        deleted = result[0]["deleted"] if result else 0
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Episode not found")

        return {"deleted": True, "episode_id": episode_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting episode: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete episode: {str(e)}",
        )


@router.post("/admin/calibrate", include_in_schema=False)
async def calibrate_memories(
    request: Request,
    decay_factor: float = 0.99,
) -> dict:
    """
    Run calibration on memories.

    Applies time-based decay to memory strength.
    """
    db = get_db(request)

    try:
        result = await db.execute_query(
            """
            MATCH (s:SemanticMemory)
            WHERE s.strength > 1.3
            SET s.strength = CASE
                WHEN s.strength * $decay > 1.3 THEN s.strength * $decay
                ELSE 1.3
            END
            RETURN count(s) as updated
            """,
            decay=decay_factor,
        )

        updated = result[0]["updated"] if result else 0
        return {"calibrated": True, "memories_updated": updated}

    except Exception as e:
        logger.exception(f"Error calibrating: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Calibration failed: {str(e)}",
        )


@router.post("/admin/reset", include_in_schema=False)
async def reset_database(
    request: Request,
    confirm: bool = False,
) -> dict:
    """
    Reset the database by deleting all nodes, relationships, and indexes.

    WARNING: This will delete ALL data!
    Requires confirm=true query parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm database reset. This will delete ALL data!",
        )

    db = get_db(request)

    try:
        # Drop vector indexes (they have dimension locked in)
        vector_indexes = [
            "concept_embeddings",
            "semantic_embeddings",
            "episodic_embeddings",
        ]
        for idx in vector_indexes:
            try:
                await db.execute_query(f"DROP INDEX {idx} IF EXISTS")
                logger.info(f"Dropped index: {idx}")
            except Exception as e:
                logger.warning(f"Could not drop index {idx}: {e}")

        # Drop fulltext index
        try:
            await db.execute_query("DROP INDEX semantic_content IF EXISTS")
        except Exception:
            pass

        # Delete all nodes and relationships
        await db.execute_query("MATCH (n) DETACH DELETE n")

        # Recreate schema (including vector indexes with correct dimensions)
        await db.setup_schema()

        logger.warning("Database reset completed")
        return {
            "reset": True,
            "message": "All data and indexes deleted. Schema recreated. Ready for re-ingestion.",
        }

    except Exception as e:
        logger.exception(f"Error resetting database: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}",
        )
