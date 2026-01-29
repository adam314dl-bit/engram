"""Episode manager for creating and storing episodic memories.

Handles:
1. Episode creation from synthesis results
2. Episode storage with concept/memory links
3. Episode retrieval and updates
"""

import logging
import uuid
from datetime import datetime

from engram.config import settings
from engram.embeddings.bge_service import BGEEmbeddingService, get_bge_embedding_service
from engram.models import EpisodicMemory
from engram.reasoning.synthesizer import SynthesisResult
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class EpisodeManager:
    """
    Manages episodic memory lifecycle.

    Responsibilities:
    1. Create episodes from reasoning results
    2. Link episodes to concepts and memories
    3. Update episode statistics
    4. Find similar episodes for templates
    """

    def __init__(
        self,
        db: Neo4jClient,
        embedding_service: BGEEmbeddingService | None = None,
    ) -> None:
        self.db = db
        self.embeddings = embedding_service or get_bge_embedding_service()

    async def create_episode(
        self,
        synthesis: SynthesisResult,
        answer_summary: str | None = None,
    ) -> EpisodicMemory:
        """
        Create an episodic memory from a synthesis result.

        Args:
            synthesis: Result from response synthesis
            answer_summary: Optional brief summary of the answer

        Returns:
            Created EpisodicMemory
        """
        # Generate embedding for behavior instruction (skip in bm25_graph mode)
        if settings.retrieval_mode != "bm25_graph":
            behavior_embedding = await self.embeddings.embed(
                synthesis.behavior.instruction
            )
        else:
            behavior_embedding = None

        # Generate answer summary if not provided
        if answer_summary is None:
            answer_summary = self._generate_summary(synthesis.answer)

        episode = EpisodicMemory(
            id=f"episode-{uuid.uuid4()}",
            query=synthesis.query,
            concepts_activated=synthesis.concepts_activated,
            memories_used=synthesis.memories_used,
            behavior_name=synthesis.behavior.name,
            behavior_instruction=synthesis.behavior.instruction,
            domain=synthesis.behavior.domain,
            answer_summary=answer_summary,
            importance=synthesis.importance,
            embedding=behavior_embedding,
        )

        # Save to database
        await self.db.save_episodic_memory(episode)

        # Link to concepts
        for concept_id in synthesis.concepts_activated[:10]:  # Limit links
            try:
                await self.db.link_episode_to_concept(episode.id, concept_id)
            except Exception as e:
                logger.warning(f"Failed to link episode to concept {concept_id}: {e}")

        # Link to memories
        for memory_id in synthesis.memories_used[:10]:  # Limit links
            try:
                await self.db.link_episode_to_memory(episode.id, memory_id)
            except Exception as e:
                logger.warning(f"Failed to link episode to memory {memory_id}: {e}")

        logger.info(
            f"Created episode {episode.id}: {episode.behavior_name} "
            f"({len(synthesis.concepts_activated)} concepts, "
            f"{len(synthesis.memories_used)} memories)"
        )

        return episode

    def _generate_summary(self, answer: str, max_length: int = 150) -> str:
        """Generate a brief summary of the answer."""
        # Take first sentence or truncate
        sentences = answer.split(".")
        if sentences:
            summary = sentences[0].strip()
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
            return summary
        return answer[:max_length]

    async def record_success(self, episode_id: str) -> None:
        """Record successful use of an episode."""
        episode = await self.db.get_episodic_memory(episode_id)
        if episode:
            episode.success_count += 1
            episode.repetition_count += 1
            episode.last_used = datetime.utcnow()
            await self.db.save_episodic_memory(episode)
            logger.debug(f"Recorded success for episode {episode_id}")

    async def record_failure(self, episode_id: str) -> None:
        """Record failed use of an episode."""
        episode = await self.db.get_episodic_memory(episode_id)
        if episode:
            episode.failure_count += 1
            episode.last_used = datetime.utcnow()
            await self.db.save_episodic_memory(episode)
            logger.debug(f"Recorded failure for episode {episode_id}")

    async def get_episode(self, episode_id: str) -> EpisodicMemory | None:
        """Get an episode by ID."""
        return await self.db.get_episodic_memory(episode_id)

    async def find_similar_episodes(
        self,
        behavior_instruction: str,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[tuple[EpisodicMemory, float]]:
        """
        Find episodes with similar behavior patterns.

        Args:
            behavior_instruction: Behavior to match
            k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (episode, similarity) tuples
        """
        embedding = await self.embeddings.embed(behavior_instruction)
        results = await self.db.vector_search_episodes(embedding=embedding, k=k)
        return [(ep, sim) for ep, sim in results if sim >= min_similarity]

    async def find_successful_episodes(
        self,
        concept_ids: list[str],
        min_success_rate: float = 0.5,
        limit: int = 10,
    ) -> list[EpisodicMemory]:
        """
        Find successful episodes that activated given concepts.

        Args:
            concept_ids: Concepts to search for
            min_success_rate: Minimum success rate threshold
            limit: Maximum results

        Returns:
            List of successful episodes
        """
        query = """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        WHERE c.id IN $concept_ids
          AND e.success_count > e.failure_count
        WITH e, count(DISTINCT c) as concept_overlap
        ORDER BY concept_overlap DESC, e.success_count DESC
        LIMIT $limit
        RETURN e
        """

        results = await self.db.execute_query(
            query,
            concept_ids=concept_ids,
            limit=limit,
        )

        episodes = []
        for record in results:
            episode = EpisodicMemory.from_dict(dict(record["e"]))
            if episode.success_rate >= min_success_rate:
                episodes.append(episode)

        return episodes

    async def get_recent_episodes(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[EpisodicMemory]:
        """Get recent episodes for reflection."""
        return await self.db.get_recent_episodes(hours=hours, limit=limit)

    async def mark_consolidated(
        self,
        episode_id: str,
        memory_id: str,
    ) -> None:
        """Mark an episode as consolidated into a semantic memory."""
        episode = await self.db.get_episodic_memory(episode_id)
        if episode:
            episode.consolidated = True
            episode.consolidated_memory_id = memory_id
            await self.db.save_episodic_memory(episode)

            # Create CRYSTALLIZED_TO relationship
            query = """
            MATCH (e:EpisodicMemory {id: $episode_id})
            MATCH (s:SemanticMemory {id: $memory_id})
            MERGE (e)-[:CRYSTALLIZED_TO]->(s)
            """
            await self.db.execute_query(
                query,
                episode_id=episode_id,
                memory_id=memory_id,
            )

            logger.info(f"Episode {episode_id} consolidated to memory {memory_id}")


async def create_episode(
    db: Neo4jClient,
    synthesis: SynthesisResult,
) -> EpisodicMemory:
    """Convenience function for episode creation."""
    manager = EpisodeManager(db=db)
    return await manager.create_episode(synthesis)
