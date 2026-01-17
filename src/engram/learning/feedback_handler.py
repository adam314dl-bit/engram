"""Feedback handler for processing user feedback on responses.

Handles three types of feedback:
1. Positive - strengthens memories and concepts, checks consolidation
2. Negative - weakens memories, triggers re-reasoning
3. Correction - creates new memory from correction text
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.learning.consolidation import Consolidator, ConsolidationResult
from engram.learning.hebbian import strengthen_concept_links, weaken_concept_links
from engram.learning.memory_strength import (
    batch_strengthen_memories,
    batch_weaken_memories,
)
from engram.learning.reflection import Reflector, ReflectionResult
from engram.models import EpisodicMemory, SemanticMemory
from engram.reasoning.episode_manager import EpisodeManager
from engram.reasoning.re_reasoning import ReReasoner, ReReasoningResult
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

FeedbackType = Literal["positive", "negative", "correction"]


@dataclass
class FeedbackResult:
    """Result of feedback processing."""

    feedback_type: FeedbackType
    episode_id: str
    success: bool

    # For positive feedback
    memories_strengthened: int = 0
    concepts_strengthened: int = 0
    consolidation: ConsolidationResult | None = None
    reflection: ReflectionResult | None = None

    # For negative feedback
    memories_weakened: int = 0
    re_reasoning: ReReasoningResult | None = None

    # For correction feedback
    correction_memory: SemanticMemory | None = None


class FeedbackHandler:
    """
    Handles user feedback on system responses.

    Feedback triggers learning:
    - Positive: Hebbian strengthening, consolidation check, reflection check
    - Negative: Weakening, re-reasoning
    - Correction: Create new memory from correction
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
        self.episode_manager = EpisodeManager(db=db, embedding_service=self.embeddings)
        self.consolidator = Consolidator(db=db, llm_client=self.llm, embedding_service=self.embeddings)
        self.reflector = Reflector(db=db, llm_client=self.llm, embedding_service=self.embeddings)
        self.re_reasoner = ReReasoner(db=db, llm_client=self.llm)

    async def handle_feedback(
        self,
        episode_id: str,
        feedback: FeedbackType,
        correction_text: str | None = None,
    ) -> FeedbackResult:
        """
        Process user feedback on an episode.

        Args:
            episode_id: ID of the episode receiving feedback
            feedback: Type of feedback (positive/negative/correction)
            correction_text: Correction text if feedback is "correction"

        Returns:
            FeedbackResult with details of actions taken
        """
        episode = await self.episode_manager.get_episode(episode_id)
        if not episode:
            logger.error(f"Episode not found: {episode_id}")
            return FeedbackResult(
                feedback_type=feedback,
                episode_id=episode_id,
                success=False,
            )

        logger.info(f"Processing {feedback} feedback for episode {episode_id}")

        if feedback == "positive":
            return await self._handle_positive(episode)
        elif feedback == "negative":
            return await self._handle_negative(episode)
        elif feedback == "correction":
            return await self._handle_correction(episode, correction_text)
        else:
            raise ValueError(f"Unknown feedback type: {feedback}")

    async def _handle_positive(self, episode: EpisodicMemory) -> FeedbackResult:
        """Handle positive feedback."""
        # Update episode
        episode.success_count += 1
        episode.feedback = "positive"
        episode.last_used = datetime.utcnow()
        await self.db.save_episodic_memory(episode)

        # 1. Strengthen used memories (Hebbian)
        memories_strengthened = await batch_strengthen_memories(
            self.db,
            episode.memories_used,
            boost=0.1,
        )

        # 2. Strengthen concept connections
        concepts_strengthened = await strengthen_concept_links(
            self.db,
            episode.concepts_activated,
            boost=0.05,
        )

        # 3. Check consolidation criteria
        consolidation = await self.consolidator.maybe_consolidate(episode)

        # 4. Check reflection trigger
        reflection = await self.reflector.maybe_reflect()

        logger.info(
            f"Positive feedback processed: "
            f"{memories_strengthened} memories strengthened, "
            f"{concepts_strengthened} links strengthened, "
            f"consolidation={consolidation.should_consolidate}, "
            f"reflection={reflection.triggered}"
        )

        return FeedbackResult(
            feedback_type="positive",
            episode_id=episode.id,
            success=True,
            memories_strengthened=memories_strengthened,
            concepts_strengthened=concepts_strengthened,
            consolidation=consolidation,
            reflection=reflection,
        )

    async def _handle_negative(self, episode: EpisodicMemory) -> FeedbackResult:
        """Handle negative feedback."""
        # Update episode
        episode.failure_count += 1
        episode.feedback = "negative"
        episode.last_used = datetime.utcnow()
        await self.db.save_episodic_memory(episode)

        # 1. Weaken used memories slightly
        memories_weakened = await batch_weaken_memories(
            self.db,
            episode.memories_used,
            factor=0.95,
        )

        # 2. Optionally weaken concept links
        await weaken_concept_links(
            self.db,
            episode.concepts_activated,
            decay=0.02,
        )

        # 3. Trigger re-reasoning
        re_reasoning = await self.re_reasoner.re_reason(episode)

        logger.info(
            f"Negative feedback processed: "
            f"{memories_weakened} memories weakened, "
            f"alternative approach found={re_reasoning.approach_changed}"
        )

        return FeedbackResult(
            feedback_type="negative",
            episode_id=episode.id,
            success=True,
            memories_weakened=memories_weakened,
            re_reasoning=re_reasoning,
        )

    async def _handle_correction(
        self,
        episode: EpisodicMemory,
        correction_text: str | None,
    ) -> FeedbackResult:
        """Handle correction feedback."""
        if not correction_text:
            logger.warning("Correction feedback without correction text")
            return FeedbackResult(
                feedback_type="correction",
                episode_id=episode.id,
                success=False,
            )

        # Update episode
        episode.feedback = "correction"
        episode.correction_text = correction_text
        episode.failure_count += 1
        episode.last_used = datetime.utcnow()
        await self.db.save_episodic_memory(episode)

        # Create new semantic memory from correction
        memory = await self._create_memory_from_correction(correction_text, episode)

        logger.info(
            f"Correction feedback processed: created memory {memory.id}"
        )

        return FeedbackResult(
            feedback_type="correction",
            episode_id=episode.id,
            success=True,
            correction_memory=memory,
        )

    async def _create_memory_from_correction(
        self,
        correction_text: str,
        episode: EpisodicMemory,
    ) -> SemanticMemory:
        """Create a new semantic memory from user correction."""
        # Generate embedding
        embedding = await self.embeddings.embed(correction_text)

        # Determine memory type based on content
        memory_type = "fact"
        if any(word in correction_text.lower() for word in ["чтобы", "нужно", "следует", "используй"]):
            memory_type = "procedure"

        # Create memory with high confidence (user-provided)
        memory = SemanticMemory(
            id=f"memory-correction-{uuid.uuid4()}",
            content=correction_text,
            concept_ids=episode.concepts_activated[:5],  # Inherit concepts
            source_episode_ids=[episode.id],
            memory_type=memory_type,
            importance=8.0,  # High importance for corrections
            confidence=0.95,  # High confidence (user-provided)
            strength=2.5,  # High initial strength
            embedding=embedding,
        )

        await self.db.save_semantic_memory(memory)

        # Link to concepts
        for concept_id in memory.concept_ids:
            try:
                await self.db.link_memory_to_concept(memory.id, concept_id)
            except Exception as e:
                logger.warning(f"Failed to link correction memory to concept: {e}")

        return memory


async def handle_feedback(
    db: Neo4jClient,
    episode_id: str,
    feedback: FeedbackType,
    correction_text: str | None = None,
) -> FeedbackResult:
    """Convenience function for handling feedback."""
    handler = FeedbackHandler(db=db)
    return await handler.handle_feedback(episode_id, feedback, correction_text)
