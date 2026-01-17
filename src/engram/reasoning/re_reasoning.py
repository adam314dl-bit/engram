"""Re-reasoning module for handling failed responses.

When an answer doesn't help, this module finds alternative approaches
by excluding failed paths and using successful alternative episodes.
"""

import logging
from dataclasses import dataclass

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import EpisodicMemory, SemanticMemory
from engram.retrieval.hybrid_search import HybridSearch, ScoredEpisode, ScoredMemory
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class ReReasoningResult:
    """Result of re-reasoning attempt."""

    answer: str
    alternative_memories: list[SemanticMemory]
    alternative_episodes: list[EpisodicMemory]
    approach_changed: bool  # Whether we found a different approach


class ReReasoner:
    """
    Handles re-reasoning when initial response fails.

    Strategy:
    1. Get the failed reasoning path (memories and concepts used)
    2. Find alternative memories (exclude failed ones)
    3. Find successful episodes with different approaches
    4. Generate new response using alternatives
    """

    def __init__(
        self,
        db: Neo4jClient,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.db = db
        self.llm = llm_client or get_llm_client()
        self.hybrid_search = HybridSearch(db=db)

    async def re_reason(
        self,
        failed_episode: EpisodicMemory,
        user_feedback: str | None = None,
    ) -> ReReasoningResult:
        """
        Find alternative approach after failure.

        Args:
            failed_episode: The episode that didn't help
            user_feedback: Optional feedback from user about what went wrong

        Returns:
            ReReasoningResult with alternative answer
        """
        logger.info(
            f"Re-reasoning for failed episode {failed_episode.id}: "
            f"{failed_episode.behavior_name}"
        )

        # 1. Get the failed reasoning path
        failed_memories = failed_episode.memories_used
        failed_concepts = failed_episode.concepts_activated

        # 2. Find alternative memories (exclude failed ones)
        alternative_memories = await self._get_alternative_memories(
            concept_ids=failed_concepts,
            exclude_ids=failed_memories,
        )

        # 3. Find successful episodes with different approaches
        alternative_episodes = await self._find_alternative_episodes(
            concept_ids=failed_concepts,
            exclude_behavior=failed_episode.behavior_name,
        )

        # 4. Generate new response with alternatives
        answer = await self._generate_alternative_response(
            failed_episode=failed_episode,
            alternative_memories=alternative_memories,
            alternative_episodes=alternative_episodes,
            user_feedback=user_feedback,
        )

        # Determine if we actually changed the approach
        approach_changed = bool(alternative_episodes) or len(alternative_memories) > 0

        return ReReasoningResult(
            answer=answer,
            alternative_memories=[sm.memory for sm in alternative_memories],
            alternative_episodes=alternative_episodes,
            approach_changed=approach_changed,
        )

    async def _get_alternative_memories(
        self,
        concept_ids: list[str],
        exclude_ids: list[str],
        limit: int = 10,
    ) -> list[ScoredMemory]:
        """Get memories for concepts, excluding already-tried ones."""
        memories = await self.db.get_memories_for_concepts(
            concept_ids=concept_ids,
            exclude_ids=exclude_ids,
            limit=limit * 2,  # Get more to filter
        )

        # Score by importance since we don't have query embedding here
        scored = []
        for memory in memories:
            score = memory.importance / 10.0
            scored.append(ScoredMemory(memory=memory, score=score, sources=["graph"]))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    async def _find_alternative_episodes(
        self,
        concept_ids: list[str],
        exclude_behavior: str,
        limit: int = 5,
    ) -> list[EpisodicMemory]:
        """Find successful episodes with different behavior approaches."""
        episodes = await self.hybrid_search.find_alternative_episodes(
            concepts=concept_ids,
            exclude_behavior=exclude_behavior,
            k=limit,
        )
        return [se.episode for se in episodes]

    async def _generate_alternative_response(
        self,
        failed_episode: EpisodicMemory,
        alternative_memories: list[ScoredMemory],
        alternative_episodes: list[EpisodicMemory],
        user_feedback: str | None = None,
    ) -> str:
        """Generate new response using alternative context."""
        # Format alternative memories
        memories_context = self._format_alternative_memories(alternative_memories)

        # Format alternative episodes
        episodes_context = self._format_alternative_episodes(alternative_episodes)

        # Build prompt
        feedback_section = ""
        if user_feedback:
            feedback_section = f"\nОбратная связь пользователя: {user_feedback}\n"

        prompt = f"""Предыдущий подход не помог пользователю.

Исходный вопрос: {failed_episode.query}

Предыдущий подход: {failed_episode.behavior_instruction}
{feedback_section}
Попробуй альтернативный подход используя эти знания:
{memories_context}

Успешные альтернативные подходы в похожих ситуациях:
{episodes_context}

Инструкции:
1. НЕ повторяй предыдущий подход — попробуй что-то другое.
2. Если альтернативных знаний мало, задай уточняющий вопрос пользователю.
3. Если есть успешные альтернативные подходы, адаптируй их.
4. Объясни, чем новый подход отличается от предыдущего.

Альтернативный ответ:"""

        response = await self.llm.generate(prompt, temperature=0.8)
        return response

    def _format_alternative_memories(
        self,
        memories: list[ScoredMemory],
        max_count: int = 8,
    ) -> str:
        """Format alternative memories for prompt."""
        if not memories:
            return "Нет альтернативных знаний по этой теме."

        lines = []
        for i, sm in enumerate(memories[:max_count], 1):
            memory = sm.memory
            lines.append(f"{i}. {memory.content}")

        return "\n".join(lines)

    def _format_alternative_episodes(
        self,
        episodes: list[EpisodicMemory],
        max_count: int = 3,
    ) -> str:
        """Format alternative episodes for prompt."""
        if not episodes:
            return "Нет успешных альтернативных подходов."

        lines = []
        for i, episode in enumerate(episodes[:max_count], 1):
            success_rate = f"{episode.success_rate:.0%}"
            lines.append(
                f"{i}. [{success_rate} успеха] {episode.behavior_name}: "
                f"{episode.behavior_instruction}"
            )

        return "\n".join(lines)


async def re_reason(
    db: Neo4jClient,
    failed_episode: EpisodicMemory,
    user_feedback: str | None = None,
) -> ReReasoningResult:
    """Convenience function for re-reasoning."""
    reasoner = ReReasoner(db=db)
    return await reasoner.re_reason(failed_episode, user_feedback)
