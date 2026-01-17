"""Unit tests for re-reasoning module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engram.models import EpisodicMemory, SemanticMemory
from engram.reasoning.re_reasoning import ReReasoner, ReReasoningResult
from engram.retrieval.hybrid_search import ScoredEpisode, ScoredMemory


class TestReReasoner:
    """Tests for ReReasoner class."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock Neo4j client."""
        db = MagicMock()
        db.get_memories_for_concepts = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM client."""
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value="Вот альтернативный подход: попробуйте использовать docker-compose."
        )
        return llm

    @pytest.fixture
    def failed_episode(self) -> EpisodicMemory:
        """Create a failed episode for testing."""
        return EpisodicMemory(
            id="episode-failed-123",
            query="Как освободить место на диске Docker?",
            concepts_activated=["concept-docker", "concept-disk"],
            memories_used=["mem-1", "mem-2"],
            behavior_name="check_disk_usage",
            behavior_instruction="Проверить использование диска командой df -h",
            domain="docker",
            answer_summary="Использовать df -h для проверки",
            success_count=0,
            failure_count=1,
        )

    @pytest.mark.asyncio
    async def test_re_reason_basic(
        self,
        mock_db,
        mock_llm,
        failed_episode,
    ) -> None:
        """Test basic re-reasoning."""
        # Setup mock hybrid search
        mock_hybrid = MagicMock()
        mock_hybrid.find_alternative_episodes = AsyncMock(return_value=[])

        reasoner = ReReasoner(db=mock_db, llm_client=mock_llm)
        reasoner.hybrid_search = mock_hybrid

        result = await reasoner.re_reason(failed_episode)

        assert isinstance(result, ReReasoningResult)
        assert "альтернативный" in result.answer.lower()
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_re_reason_with_alternative_memories(
        self,
        mock_db,
        mock_llm,
        failed_episode,
    ) -> None:
        """Test re-reasoning finds alternative memories."""
        # Setup alternative memory
        alt_memory = SemanticMemory(
            id="mem-alt-1",
            content="Используйте docker system prune для очистки",
            memory_type="procedure",
            importance=8.0,
        )
        mock_db.get_memories_for_concepts = AsyncMock(return_value=[alt_memory])

        mock_hybrid = MagicMock()
        mock_hybrid.find_alternative_episodes = AsyncMock(return_value=[])

        reasoner = ReReasoner(db=mock_db, llm_client=mock_llm)
        reasoner.hybrid_search = mock_hybrid

        result = await reasoner.re_reason(failed_episode)

        assert len(result.alternative_memories) == 1
        assert result.alternative_memories[0].id == "mem-alt-1"

    @pytest.mark.asyncio
    async def test_re_reason_with_alternative_episodes(
        self,
        mock_db,
        mock_llm,
        failed_episode,
    ) -> None:
        """Test re-reasoning finds alternative successful episodes."""
        # Setup alternative episode
        alt_episode = EpisodicMemory(
            id="episode-alt-1",
            query="Как очистить Docker?",
            behavior_name="prune_docker",
            behavior_instruction="Использовать docker system prune -a",
            success_count=5,
            failure_count=0,
        )

        mock_hybrid = MagicMock()
        mock_hybrid.find_alternative_episodes = AsyncMock(
            return_value=[ScoredEpisode(episode=alt_episode, score=0.9)]
        )

        reasoner = ReReasoner(db=mock_db, llm_client=mock_llm)
        reasoner.hybrid_search = mock_hybrid

        result = await reasoner.re_reason(failed_episode)

        assert len(result.alternative_episodes) == 1
        assert result.alternative_episodes[0].behavior_name == "prune_docker"
        assert result.approach_changed is True

    @pytest.mark.asyncio
    async def test_re_reason_with_user_feedback(
        self,
        mock_db,
        mock_llm,
        failed_episode,
    ) -> None:
        """Test re-reasoning uses user feedback."""
        mock_hybrid = MagicMock()
        mock_hybrid.find_alternative_episodes = AsyncMock(return_value=[])

        reasoner = ReReasoner(db=mock_db, llm_client=mock_llm)
        reasoner.hybrid_search = mock_hybrid

        await reasoner.re_reason(
            failed_episode,
            user_feedback="Команда df не помогла, нужен способ удаления старых образов",
        )

        # Check that prompt included user feedback
        call_args = mock_llm.generate.call_args
        prompt = call_args[0][0]
        assert "Обратная связь пользователя" in prompt
        assert "старых образов" in prompt

    @pytest.mark.asyncio
    async def test_re_reason_approach_not_changed_when_no_alternatives(
        self,
        mock_db,
        mock_llm,
        failed_episode,
    ) -> None:
        """Test approach_changed is False when no alternatives found."""
        mock_db.get_memories_for_concepts = AsyncMock(return_value=[])

        mock_hybrid = MagicMock()
        mock_hybrid.find_alternative_episodes = AsyncMock(return_value=[])

        reasoner = ReReasoner(db=mock_db, llm_client=mock_llm)
        reasoner.hybrid_search = mock_hybrid

        result = await reasoner.re_reason(failed_episode)

        assert result.approach_changed is False


class TestReReasoningFormatting:
    """Tests for formatting functions in re-reasoning."""

    @pytest.fixture
    def reasoner(self) -> ReReasoner:
        """Create reasoner with mocks."""
        mock_db = MagicMock()
        return ReReasoner(db=mock_db)

    def test_format_alternative_memories_empty(self, reasoner) -> None:
        """Test formatting empty memories."""
        result = reasoner._format_alternative_memories([])
        assert "Нет альтернативных знаний" in result

    def test_format_alternative_memories_with_content(self, reasoner) -> None:
        """Test formatting memories with content."""
        memory = SemanticMemory(
            id="mem-1",
            content="Используйте docker system prune",
            memory_type="procedure",
        )
        scored = ScoredMemory(memory=memory, score=0.8, sources=["graph"])

        result = reasoner._format_alternative_memories([scored])

        assert "docker system prune" in result
        assert "1." in result

    def test_format_alternative_episodes_empty(self, reasoner) -> None:
        """Test formatting empty episodes."""
        result = reasoner._format_alternative_episodes([])
        assert "Нет успешных альтернативных" in result

    def test_format_alternative_episodes_with_success_rate(self, reasoner) -> None:
        """Test formatting episodes shows success rate."""
        episode = EpisodicMemory(
            id="ep-1",
            query="test",
            behavior_name="test_behavior",
            behavior_instruction="Do the test",
            success_count=4,
            failure_count=1,
        )

        result = reasoner._format_alternative_episodes([episode])

        assert "test_behavior" in result
        assert "80%" in result  # 4/(4+1) = 80%
