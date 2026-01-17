"""Unit tests for learning system."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from engram.models import EpisodicMemory, SemanticMemory
from engram.learning.memory_strength import (
    update_memory_strength,
    strengthen_memory,
    weaken_memory,
    MIN_EASINESS,
    MAX_EASINESS,
)
from engram.learning.hebbian import (
    strengthen_concept_links,
    weaken_concept_links,
)
from engram.learning.consolidation import (
    Consolidator,
    ConsolidationResult,
    MIN_SUCCESSFUL_EPISODES,
    MIN_SUCCESS_RATE,
    MIN_IMPORTANCE,
)
from engram.learning.reflection import (
    Reflector,
    Reflection,
    IMPORTANCE_THRESHOLD,
)
from engram.learning.feedback_handler import (
    FeedbackHandler,
    FeedbackResult,
)


class TestMemoryStrength:
    """Tests for SM-2 memory strength algorithm."""

    @pytest.fixture
    def mock_db(self, sample_semantic_memory) -> MagicMock:
        """Create mock DB."""
        db = MagicMock()
        db.get_semantic_memory = AsyncMock(return_value=sample_semantic_memory)
        db.save_semantic_memory = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_strengthen_memory_success(self, mock_db, sample_semantic_memory) -> None:
        """Test strengthening memory increases strength."""
        sample_semantic_memory.strength = 2.0  # Set below max
        old_strength = sample_semantic_memory.strength

        result = await strengthen_memory(mock_db, sample_semantic_memory.id, boost=0.1)

        assert result is not None
        assert result.strength > old_strength
        assert result.access_count > 0
        mock_db.save_semantic_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_strengthen_memory_max_cap(self, mock_db, sample_semantic_memory) -> None:
        """Test strength is capped at MAX_EASINESS."""
        sample_semantic_memory.strength = MAX_EASINESS - 0.05

        result = await strengthen_memory(mock_db, sample_semantic_memory.id, boost=0.2)

        assert result.strength == MAX_EASINESS

    @pytest.mark.asyncio
    async def test_weaken_memory(self, mock_db, sample_semantic_memory) -> None:
        """Test weakening memory decreases strength."""
        sample_semantic_memory.strength = 2.0
        old_strength = sample_semantic_memory.strength

        result = await weaken_memory(mock_db, sample_semantic_memory.id, factor=0.9)

        assert result is not None
        assert result.strength < old_strength
        assert result.strength >= MIN_EASINESS

    @pytest.mark.asyncio
    async def test_weaken_memory_min_cap(self, mock_db, sample_semantic_memory) -> None:
        """Test strength doesn't go below MIN_EASINESS."""
        sample_semantic_memory.strength = MIN_EASINESS + 0.1

        result = await weaken_memory(mock_db, sample_semantic_memory.id, factor=0.5)

        assert result.strength >= MIN_EASINESS

    @pytest.mark.asyncio
    async def test_update_memory_strength_quality_5(self, mock_db, sample_semantic_memory) -> None:
        """Test SM-2 with perfect quality (5)."""
        sample_semantic_memory.strength = 2.0
        old_strength = sample_semantic_memory.strength

        result = await update_memory_strength(mock_db, sample_semantic_memory.id, quality=5)

        # Perfect quality should increase strength
        assert result.strength > old_strength
        assert result.access_count > 0

    @pytest.mark.asyncio
    async def test_update_memory_strength_quality_0(self, mock_db, sample_semantic_memory) -> None:
        """Test SM-2 with complete failure (0)."""
        sample_semantic_memory.strength = 2.0
        old_strength = sample_semantic_memory.strength

        result = await update_memory_strength(mock_db, sample_semantic_memory.id, quality=0)

        # Failure should decrease strength
        assert result.strength < old_strength

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, mock_db) -> None:
        """Test handling of non-existent memory."""
        mock_db.get_semantic_memory = AsyncMock(return_value=None)

        result = await update_memory_strength(mock_db, "nonexistent", quality=5)

        assert result is None


class TestHebbianLearning:
    """Tests for Hebbian concept link strengthening."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock DB."""
        db = MagicMock()
        db.execute_query = AsyncMock(return_value=[{"new_weight": 0.6}])
        return db

    @pytest.mark.asyncio
    async def test_strengthen_concept_links(self, mock_db) -> None:
        """Test strengthening links between concepts."""
        concept_ids = ["c1", "c2", "c3"]

        count = await strengthen_concept_links(mock_db, concept_ids)

        # Should try to strengthen edges between all pairs
        # 3 concepts = 3 pairs × 2 directions = 6 edges
        assert count > 0

    @pytest.mark.asyncio
    async def test_strengthen_single_concept(self, mock_db) -> None:
        """Test with single concept (nothing to link)."""
        count = await strengthen_concept_links(mock_db, ["c1"])

        assert count == 0

    @pytest.mark.asyncio
    async def test_weaken_concept_links(self, mock_db) -> None:
        """Test weakening links between concepts."""
        concept_ids = ["c1", "c2"]

        count = await weaken_concept_links(mock_db, concept_ids)

        assert count >= 0  # May be 0 if no edges exist


class TestConsolidation:
    """Tests for episode consolidation."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock DB."""
        db = MagicMock()
        db.vector_search_episodes = AsyncMock(return_value=[])
        db.save_semantic_memory = AsyncMock()
        db.link_memory_to_concept = AsyncMock()
        db.get_episodic_memory = AsyncMock(return_value=None)
        db.save_episodic_memory = AsyncMock()
        db.execute_query = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def successful_episode(self) -> EpisodicMemory:
        """Create a successful episode for testing."""
        return EpisodicMemory(
            id="ep-1",
            query="test query",
            concepts_activated=["c1", "c2"],
            memories_used=["m1"],
            behavior_name="test_behavior",
            behavior_instruction="Do the test",
            domain="docker",
            importance=8.0,
            success_count=5,
            failure_count=0,
            embedding=[0.1] * 384,
        )

    def test_check_criteria_all_met(self) -> None:
        """Test criteria checking when all criteria are met."""
        consolidator = Consolidator(db=MagicMock())

        episode = EpisodicMemory(
            id="ep-1", query="test", importance=8.0,
            success_count=3, failure_count=0, domain="docker",
        )
        similar = [
            EpisodicMemory(id=f"ep-{i}", query="test", importance=8.0,
                          success_count=3, failure_count=0, domain=d)
            for i, d in enumerate(["kubernetes", "linux", "general"], 2)
        ]

        result = consolidator._check_criteria(episode, similar)

        assert result.should_consolidate is True
        assert result.criteria_met >= 3
        assert result.repetition_met is True
        assert result.success_rate_met is True
        assert result.importance_met is True
        assert result.cross_domain_met is True

    def test_check_criteria_not_met(self) -> None:
        """Test criteria checking when criteria are not met."""
        consolidator = Consolidator(db=MagicMock())

        episode = EpisodicMemory(
            id="ep-1", query="test", importance=3.0,
            success_count=1, failure_count=2, domain="docker",
        )

        result = consolidator._check_criteria(episode, [])

        assert result.should_consolidate is False
        assert result.criteria_met < 3

    @pytest.mark.asyncio
    async def test_maybe_consolidate_already_consolidated(
        self, mock_db, successful_episode
    ) -> None:
        """Test that already consolidated episodes are skipped."""
        successful_episode.consolidated = True

        consolidator = Consolidator(db=mock_db)
        result = await consolidator.maybe_consolidate(successful_episode)

        assert result.should_consolidate is False


class TestReflection:
    """Tests for threshold-triggered reflection."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock DB."""
        db = MagicMock()
        db.get_recent_episodes = AsyncMock(return_value=[])
        db.get_concept_by_name = AsyncMock(return_value=None)
        db.save_semantic_memory = AsyncMock()
        db.link_memory_to_concept = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_reflection_not_triggered_low_importance(self, mock_db) -> None:
        """Test reflection not triggered when importance is low."""
        episodes = [
            EpisodicMemory(id=f"ep-{i}", query="test", importance=5.0)
            for i in range(5)
        ]
        mock_db.get_recent_episodes = AsyncMock(return_value=episodes)

        reflector = Reflector(db=mock_db, importance_threshold=IMPORTANCE_THRESHOLD)
        result = await reflector.maybe_reflect()

        # 5 episodes × 5 importance = 25 < 150 threshold
        assert result.triggered is False
        assert result.importance_sum == 25

    @pytest.mark.asyncio
    async def test_reflection_triggered_high_importance(
        self, mock_db, mock_llm_client, mock_embedding_service
    ) -> None:
        """Test reflection triggered when importance exceeds threshold."""
        episodes = [
            EpisodicMemory(id=f"ep-{i}", query="test", importance=10.0, domain="docker")
            for i in range(20)
        ]
        mock_db.get_recent_episodes = AsyncMock(return_value=episodes)

        reflector = Reflector(
            db=mock_db,
            llm_client=mock_llm_client,
            embedding_service=mock_embedding_service,
            importance_threshold=150,
        )

        # Mock LLM response
        mock_llm_client.generate = AsyncMock(
            return_value="ПАТТЕРН: [docker, container] Пользователи часто спрашивают о Docker"
        )

        result = await reflector.maybe_reflect()

        # 20 episodes × 10 importance = 200 > 150 threshold
        assert result.triggered is True
        assert result.importance_sum == 200

    def test_parse_reflections(self) -> None:
        """Test parsing LLM reflection response."""
        reflector = Reflector(db=MagicMock())

        response = """ПАТТЕРН: [docker, container] Частые вопросы о контейнерах
ПОДХОД: [kubernetes] kubectl apply работает лучше всего
ПРОБЕЛ: [networking, dns] Много вопросов о сети без ответов"""

        reflections = reflector._parse_reflections(response)

        assert len(reflections) == 3
        assert reflections[0].reflection_type == "pattern"
        assert "docker" in reflections[0].concepts
        assert reflections[1].reflection_type == "approach"
        assert reflections[2].reflection_type == "gap"


class TestFeedbackHandler:
    """Tests for feedback handler."""

    @pytest.fixture
    def mock_db(self, sample_semantic_memory, sample_episodic_memory) -> MagicMock:
        """Create mock DB with full setup."""
        db = MagicMock()
        db.get_episodic_memory = AsyncMock(return_value=sample_episodic_memory)
        db.save_episodic_memory = AsyncMock()
        db.get_semantic_memory = AsyncMock(return_value=sample_semantic_memory)
        db.save_semantic_memory = AsyncMock()
        db.link_memory_to_concept = AsyncMock()
        db.execute_query = AsyncMock(return_value=[{"new_weight": 0.6}])
        db.vector_search_episodes = AsyncMock(return_value=[])
        db.get_recent_episodes = AsyncMock(return_value=[])
        db.get_memories_for_concepts = AsyncMock(return_value=[])
        return db

    @pytest.mark.asyncio
    async def test_handle_positive_feedback(
        self, mock_db, mock_embedding_service, sample_episodic_memory
    ) -> None:
        """Test positive feedback handling."""
        handler = FeedbackHandler(db=mock_db, embedding_service=mock_embedding_service)

        result = await handler.handle_feedback(
            episode_id=sample_episodic_memory.id,
            feedback="positive",
        )

        assert result.success is True
        assert result.feedback_type == "positive"
        assert sample_episodic_memory.success_count > 0
        assert sample_episodic_memory.feedback == "positive"

    @pytest.mark.asyncio
    async def test_handle_negative_feedback(
        self, mock_db, mock_embedding_service, mock_llm_client, sample_episodic_memory
    ) -> None:
        """Test negative feedback handling."""
        # Ensure find_alternative_episodes returns empty (no records with "e" key)
        mock_db.execute_query = AsyncMock(return_value=[])

        handler = FeedbackHandler(
            db=mock_db,
            llm_client=mock_llm_client,
            embedding_service=mock_embedding_service,
        )

        result = await handler.handle_feedback(
            episode_id=sample_episodic_memory.id,
            feedback="negative",
        )

        assert result.success is True
        assert result.feedback_type == "negative"
        assert sample_episodic_memory.failure_count > 0
        assert sample_episodic_memory.feedback == "negative"
        assert result.re_reasoning is not None

    @pytest.mark.asyncio
    async def test_handle_correction_feedback(
        self, mock_db, mock_embedding_service, sample_episodic_memory
    ) -> None:
        """Test correction feedback handling."""
        handler = FeedbackHandler(db=mock_db, embedding_service=mock_embedding_service)

        result = await handler.handle_feedback(
            episode_id=sample_episodic_memory.id,
            feedback="correction",
            correction_text="Правильный ответ: используйте docker system prune -a",
        )

        assert result.success is True
        assert result.feedback_type == "correction"
        assert result.correction_memory is not None
        assert sample_episodic_memory.correction_text is not None

    @pytest.mark.asyncio
    async def test_handle_correction_without_text(
        self, mock_db, mock_embedding_service, sample_episodic_memory
    ) -> None:
        """Test correction feedback without text fails gracefully."""
        handler = FeedbackHandler(db=mock_db, embedding_service=mock_embedding_service)

        result = await handler.handle_feedback(
            episode_id=sample_episodic_memory.id,
            feedback="correction",
            correction_text=None,
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_handle_feedback_episode_not_found(
        self, mock_db, mock_embedding_service
    ) -> None:
        """Test feedback for non-existent episode."""
        mock_db.get_episodic_memory = AsyncMock(return_value=None)

        handler = FeedbackHandler(db=mock_db, embedding_service=mock_embedding_service)

        result = await handler.handle_feedback(
            episode_id="nonexistent",
            feedback="positive",
        )

        assert result.success is False
