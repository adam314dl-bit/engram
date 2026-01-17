"""Consolidation: Converting successful episodic patterns into semantic memories.

Episodic memories that meet consolidation criteria (3 of 4):
1. Repetition: 3+ successful uses
2. Success rate: 85%+
3. Importance: 7+
4. Cross-domain: used in 2+ domains

...are "crystallized" into procedural semantic memories.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import EpisodicMemory, SemanticMemory
from engram.reasoning.episode_manager import EpisodeManager
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Consolidation thresholds
MIN_SUCCESSFUL_EPISODES = 3
MIN_SUCCESS_RATE = 0.85
MIN_IMPORTANCE = 7.0
MIN_DOMAINS = 2
CRITERIA_REQUIRED = 3  # Need 3 of 4 criteria


@dataclass
class ConsolidationResult:
    """Result of consolidation check."""

    should_consolidate: bool
    criteria_met: int
    similar_episodes: list[EpisodicMemory]
    created_memory: SemanticMemory | None = None

    # Detailed criteria results
    repetition_met: bool = False
    success_rate_met: bool = False
    importance_met: bool = False
    cross_domain_met: bool = False


class Consolidator:
    """
    Handles consolidation of episodic memories into semantic memories.

    Consolidation transforms successful reasoning patterns into
    permanent procedural knowledge.
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
        self.episode_manager = EpisodeManager(db=db, embedding_service=self.embeddings)

    async def maybe_consolidate(
        self,
        episode: EpisodicMemory,
    ) -> ConsolidationResult:
        """
        Check if episode should be consolidated and do it if criteria met.

        Args:
            episode: Episode to check for consolidation

        Returns:
            ConsolidationResult with details
        """
        if episode.consolidated:
            logger.debug(f"Episode {episode.id} already consolidated")
            return ConsolidationResult(
                should_consolidate=False,
                criteria_met=0,
                similar_episodes=[],
            )

        # Find similar episodes
        similar = await self._find_similar_episodes(episode)

        # Check all criteria
        result = self._check_criteria(episode, similar)

        if result.should_consolidate:
            logger.info(
                f"Episode {episode.id} meets consolidation criteria "
                f"({result.criteria_met}/4)"
            )
            memory = await self._crystallize(episode, result.similar_episodes)
            result.created_memory = memory

        return result

    async def _find_similar_episodes(
        self,
        episode: EpisodicMemory,
        k: int = 10,
    ) -> list[EpisodicMemory]:
        """Find episodes with similar behavior patterns."""
        if not episode.embedding:
            return []

        results = await self.db.vector_search_episodes(
            embedding=episode.embedding,
            k=k,
        )

        # Filter to successful episodes
        similar = []
        for ep, similarity in results:
            if similarity >= 0.7 and ep.success_count > ep.failure_count:
                similar.append(ep)

        return similar

    def _check_criteria(
        self,
        episode: EpisodicMemory,
        similar: list[EpisodicMemory],
    ) -> ConsolidationResult:
        """Check consolidation criteria."""
        criteria_met = 0
        all_episodes = [episode] + similar

        # 1. Repetition: 3+ successful uses
        successful_count = sum(
            1 for e in all_episodes if e.success_count > e.failure_count
        )
        repetition_met = successful_count >= MIN_SUCCESSFUL_EPISODES
        if repetition_met:
            criteria_met += 1

        # 2. Success rate: 85%+
        total_success = sum(e.success_count for e in all_episodes)
        total_failure = sum(e.failure_count for e in all_episodes)
        total = total_success + total_failure
        success_rate = total_success / max(total, 1)
        success_rate_met = success_rate >= MIN_SUCCESS_RATE
        if success_rate_met:
            criteria_met += 1

        # 3. Importance: average 7+
        avg_importance = sum(e.importance for e in all_episodes) / len(all_episodes)
        importance_met = avg_importance >= MIN_IMPORTANCE
        if importance_met:
            criteria_met += 1

        # 4. Cross-domain: used in 2+ domains
        domains = set(e.domain for e in all_episodes)
        cross_domain_met = len(domains) >= MIN_DOMAINS
        if cross_domain_met:
            criteria_met += 1

        return ConsolidationResult(
            should_consolidate=criteria_met >= CRITERIA_REQUIRED,
            criteria_met=criteria_met,
            similar_episodes=similar,
            repetition_met=repetition_met,
            success_rate_met=success_rate_met,
            importance_met=importance_met,
            cross_domain_met=cross_domain_met,
        )

    async def _crystallize(
        self,
        episode: EpisodicMemory,
        similar: list[EpisodicMemory],
    ) -> SemanticMemory:
        """
        Transform successful episode pattern into semantic memory.

        Uses LLM to generalize the pattern across similar episodes.
        """
        all_episodes = [episode] + similar

        # Format episodes for LLM
        episodes_text = self._format_episodes_for_crystallization(all_episodes)

        # Generate generalized knowledge
        prompt = f"""Эти похожие вопросы успешно решались одинаковым подходом:

{episodes_text}

Извлеки общий паттерн знания в формате:
"Когда [ситуация], [действие] потому что [причина]"

Требования:
1. Будь кратким и общим, не привязывайся к конкретным деталям.
2. Сформулируй как универсальную процедуру, применимую к похожим ситуациям.
3. Включи причину/обоснование если возможно.

Ответ должен быть одним предложением."""

        generalized = await self.llm.generate(prompt, temperature=0.3)
        generalized = generalized.strip()

        # Generate embedding for the generalized knowledge
        embedding = await self.embeddings.embed(generalized)

        # Calculate importance and confidence
        avg_importance = sum(e.importance for e in all_episodes) / len(all_episodes)
        total_success = sum(e.success_count for e in all_episodes)
        total_failure = sum(e.failure_count for e in all_episodes)
        confidence = total_success / max(total_success + total_failure, 1)

        # Create semantic memory
        memory = SemanticMemory(
            id=f"memory-crystallized-{uuid.uuid4()}",
            content=generalized,
            concept_ids=list(set(
                cid for e in all_episodes for cid in e.concepts_activated
            ))[:10],  # Limit concepts
            source_episode_ids=[e.id for e in all_episodes],
            memory_type="procedure",
            importance=avg_importance,
            confidence=confidence,
            strength=2.5,  # High initial strength for crystallized knowledge
            embedding=embedding,
        )

        # Save memory
        await self.db.save_semantic_memory(memory)

        # Link memory to concepts
        for concept_id in memory.concept_ids:
            try:
                await self.db.link_memory_to_concept(memory.id, concept_id)
            except Exception as e:
                logger.warning(f"Failed to link memory to concept {concept_id}: {e}")

        # Mark episodes as consolidated
        for ep in all_episodes:
            await self.episode_manager.mark_consolidated(ep.id, memory.id)

        logger.info(
            f"Crystallized {len(all_episodes)} episodes into memory {memory.id}: "
            f"{generalized[:80]}..."
        )

        return memory

    def _format_episodes_for_crystallization(
        self,
        episodes: list[EpisodicMemory],
        max_count: int = 5,
    ) -> str:
        """Format episodes for LLM crystallization prompt."""
        lines = []
        for i, ep in enumerate(episodes[:max_count], 1):
            lines.append(f"{i}. Вопрос: {ep.query}")
            lines.append(f"   Подход: {ep.behavior_instruction}")
            if ep.answer_summary:
                lines.append(f"   Результат: {ep.answer_summary}")
            lines.append("")

        return "\n".join(lines)


async def maybe_consolidate(
    db: Neo4jClient,
    episode: EpisodicMemory,
) -> ConsolidationResult:
    """Convenience function for consolidation check."""
    consolidator = Consolidator(db=db)
    return await consolidator.maybe_consolidate(episode)
