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

from engram.learning import FeedbackHandler, FeedbackType
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
    top_k_memories: int = Field(default=10, ge=1, le=50)
    top_k_episodes: int = Field(default=3, ge=0, le=10)


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
    concepts_activated: list[str] = []
    memories_used: int = 0


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
        pipeline = ReasoningPipeline(db=db)
        result = await pipeline.reason(
            query=query,
            top_k_memories=body.top_k_memories,
            top_k_episodes=body.top_k_episodes,
            temperature=body.temperature,
        )

        # Append stats to response content for visibility in Open WebUI
        memories_count = len(result.synthesis.memories_used)
        concepts = result.synthesis.concepts_activated[:5]
        confidence_pct = int(result.confidence * 100)

        stats_footer = f"\n\n---\n*Confidence: {confidence_pct}% | Memories: {memories_count} | Concepts: {', '.join(concepts[:3])}{'...' if len(concepts) > 3 else ''}*"

        content_with_stats = result.answer + stats_footer

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
            concepts_activated=result.synthesis.concepts_activated[:10],
            memories_used=memories_count,
        )

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {str(e)}",
        )


# ============================================================================
# Feedback Endpoint
# ============================================================================


@router.post("/v1/feedback", response_model=FeedbackResponse)
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


@router.get("/health", response_model=HealthResponse)
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


@router.get("/v1/models")
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


@router.get("/admin/stats", response_model=StatsResponse)
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


@router.get("/admin/concepts", response_model=list[ConceptInfo])
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


@router.get("/admin/memories", response_model=list[MemoryInfo])
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


@router.get("/admin/episodes", response_model=list[EpisodeInfo])
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


@router.delete("/admin/episodes/{episode_id}")
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


@router.post("/admin/calibrate")
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


@router.post("/admin/reset")
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
