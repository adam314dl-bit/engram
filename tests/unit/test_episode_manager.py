"""Unit tests for episode manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from engram.models import EpisodicMemory
from engram.reasoning.episode_manager import EpisodeManager
from engram.reasoning.synthesizer import Behavior, SynthesisResult


class TestEpisodeManager:
    """Tests for EpisodeManager class."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock Neo4j client."""
        db = MagicMock()
        db.save_episodic_memory = AsyncMock()
        db.get_episodic_memory = AsyncMock(return_value=None)
        db.link_episode_to_concept = AsyncMock()
        db.link_episode_to_memory = AsyncMock()
        db.get_recent_episodes = AsyncMock(return_value=[])
        db.execute_query = AsyncMock(return_value=[])
        db.vector_search_episodes = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def mock_synthesis(self) -> SynthesisResult:
        """Create mock synthesis result."""
        return SynthesisResult(
            answer="Docker is a container platform.",
            behavior=Behavior(
                name="explain_concept",
                instruction="Explain the concept with examples",
                domain="docker",
            ),
            memories_used=["mem-1", "mem-2"],
            concepts_activated=["concept-1", "concept-2"],
            confidence=0.85,
            query="What is Docker?",
            importance=6.0,
        )

    @pytest.mark.asyncio
    async def test_create_episode(
        self,
        mock_db,
        mock_embedding_service,
        mock_synthesis,
    ) -> None:
        """Test creating an episode from synthesis result."""
        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        episode = await manager.create_episode(mock_synthesis)

        assert episode.query == "What is Docker?"
        assert episode.behavior_name == "explain_concept"
        assert episode.behavior_instruction == "Explain the concept with examples"
        assert episode.domain == "docker"
        assert episode.importance == 6.0
        assert len(episode.concepts_activated) == 2
        assert len(episode.memories_used) == 2
        assert episode.embedding is not None

        # Verify DB calls
        mock_db.save_episodic_memory.assert_called_once()
        assert mock_db.link_episode_to_concept.call_count == 2
        assert mock_db.link_episode_to_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_create_episode_with_summary(
        self,
        mock_db,
        mock_embedding_service,
        mock_synthesis,
    ) -> None:
        """Test creating episode with custom summary."""
        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        episode = await manager.create_episode(
            mock_synthesis,
            answer_summary="Custom summary of the answer",
        )

        assert episode.answer_summary == "Custom summary of the answer"

    @pytest.mark.asyncio
    async def test_create_episode_auto_summary(
        self,
        mock_db,
        mock_embedding_service,
        mock_synthesis,
    ) -> None:
        """Test automatic summary generation."""
        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        episode = await manager.create_episode(mock_synthesis)

        # Should use first sentence of answer
        assert "Docker is a container platform" in episode.answer_summary

    @pytest.mark.asyncio
    async def test_record_success(
        self,
        mock_db,
        mock_embedding_service,
        sample_episodic_memory,
    ) -> None:
        """Test recording successful episode."""
        mock_db.get_episodic_memory = AsyncMock(return_value=sample_episodic_memory)
        original_success = sample_episodic_memory.success_count

        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        await manager.record_success(sample_episodic_memory.id)

        assert sample_episodic_memory.success_count == original_success + 1
        mock_db.save_episodic_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_failure(
        self,
        mock_db,
        mock_embedding_service,
        sample_episodic_memory,
    ) -> None:
        """Test recording failed episode."""
        mock_db.get_episodic_memory = AsyncMock(return_value=sample_episodic_memory)
        original_failure = sample_episodic_memory.failure_count

        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        await manager.record_failure(sample_episodic_memory.id)

        assert sample_episodic_memory.failure_count == original_failure + 1
        mock_db.save_episodic_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_similar_episodes(
        self,
        mock_db,
        mock_embedding_service,
        sample_episodic_memory,
    ) -> None:
        """Test finding similar episodes."""
        mock_db.vector_search_episodes = AsyncMock(
            return_value=[(sample_episodic_memory, 0.85)]
        )

        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        results = await manager.find_similar_episodes(
            behavior_instruction="Check disk usage",
            k=5,
        )

        assert len(results) == 1
        assert results[0][0] == sample_episodic_memory
        assert results[0][1] == 0.85

    @pytest.mark.asyncio
    async def test_find_similar_episodes_filters_low_similarity(
        self,
        mock_db,
        mock_embedding_service,
        sample_episodic_memory,
    ) -> None:
        """Test that low similarity episodes are filtered."""
        mock_db.vector_search_episodes = AsyncMock(
            return_value=[(sample_episodic_memory, 0.3)]  # Below threshold
        )

        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        results = await manager.find_similar_episodes(
            behavior_instruction="Unrelated instruction",
            k=5,
            min_similarity=0.5,
        )

        assert len(results) == 0

    def test_generate_summary_short(
        self,
        mock_db,
        mock_embedding_service,
    ) -> None:
        """Test summary generation for short answers."""
        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        summary = manager._generate_summary("Docker is great.")
        assert summary == "Docker is great"

    def test_generate_summary_long(
        self,
        mock_db,
        mock_embedding_service,
    ) -> None:
        """Test summary generation for long answers."""
        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        long_answer = "This is a very long answer that exceeds the maximum length for a summary and should be truncated to fit within the specified limit." * 3

        summary = manager._generate_summary(long_answer, max_length=50)

        assert len(summary) <= 50
        assert summary.endswith("...")


class TestEpisodeManagerConsolidation:
    """Tests for episode consolidation."""

    @pytest.fixture
    def mock_db(self, sample_episodic_memory) -> MagicMock:
        """Create mock DB with consolidation support."""
        db = MagicMock()
        db.get_episodic_memory = AsyncMock(return_value=sample_episodic_memory)
        db.save_episodic_memory = AsyncMock()
        db.execute_query = AsyncMock(return_value=[])
        return db

    @pytest.mark.asyncio
    async def test_mark_consolidated(
        self,
        mock_db,
        mock_embedding_service,
        sample_episodic_memory,
    ) -> None:
        """Test marking episode as consolidated."""
        manager = EpisodeManager(db=mock_db, embedding_service=mock_embedding_service)

        await manager.mark_consolidated(
            episode_id=sample_episodic_memory.id,
            memory_id="new-semantic-memory-id",
        )

        assert sample_episodic_memory.consolidated is True
        assert sample_episodic_memory.consolidated_memory_id == "new-semantic-memory-id"
        mock_db.save_episodic_memory.assert_called_once()
        mock_db.execute_query.assert_called_once()  # CRYSTALLIZED_TO relationship
