"""Unit tests for response synthesizer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engram.models import SemanticMemory, EpisodicMemory
from engram.reasoning.synthesizer import (
    Behavior,
    SynthesisResult,
    extract_behavior,
    format_memories_context,
    format_episodes_context,
    infer_behavior,
    infer_domain,
    ResponseSynthesizer,
)
from engram.retrieval.hybrid_search import ScoredMemory, ScoredEpisode
from engram.retrieval.pipeline import RetrievalResult


class TestBehaviorExtraction:
    """Tests for behavior extraction functions."""

    def test_extract_behavior_explicit_strategy(self) -> None:
        """Test extracting explicit СТРАТЕГИЯ line."""
        response = """Docker — это платформа контейнеризации.

СТРАТЕГИЯ: [explain_concept] — Объяснить концепцию Docker и её основные компоненты"""

        behavior = extract_behavior(response, "Что такое Docker?")

        assert behavior.name == "explain_concept"
        assert "Объяснить концепцию" in behavior.instruction
        assert behavior.domain == "docker"

    def test_extract_behavior_with_dash(self) -> None:
        """Test extracting strategy with dash separator."""
        response = """Ответ на вопрос.

СТРАТЕГИЯ: troubleshoot_issue - Диагностировать проблему и найти решение"""

        behavior = extract_behavior(response, "Ошибка при запуске контейнера")

        assert behavior.name == "troubleshoot_issue"
        assert "Диагностировать" in behavior.instruction

    def test_extract_behavior_fallback(self) -> None:
        """Test fallback behavior inference when no strategy line."""
        response = "Вот ответ на ваш вопрос без явной стратегии."

        behavior = extract_behavior(response, "Как настроить Docker?")

        assert behavior.name == "provide_instructions"
        assert behavior.domain == "docker"

    def test_infer_behavior_explain(self) -> None:
        """Test behavior inference for explain queries."""
        name, instruction = infer_behavior("Что такое Kubernetes?", "response")
        assert name == "explain_concept"

        name, instruction = infer_behavior("Explain Docker containers", "response")
        assert name == "explain_concept"

    def test_infer_behavior_how_to(self) -> None:
        """Test behavior inference for how-to queries."""
        name, instruction = infer_behavior("Как создать контейнер?", "response")
        assert name == "provide_instructions"

        name, instruction = infer_behavior("How to deploy an app?", "response")
        assert name == "provide_instructions"

    def test_infer_behavior_troubleshoot(self) -> None:
        """Test behavior inference for troubleshooting queries."""
        name, instruction = infer_behavior("Ошибка при запуске", "response")
        assert name == "troubleshoot"

        name, instruction = infer_behavior("Container failed to start", "response")
        assert name == "troubleshoot"

    def test_infer_domain_docker(self) -> None:
        """Test domain inference for Docker content."""
        domain = infer_domain("Docker контейнер", "")
        assert domain == "docker"

        domain = infer_domain("Dockerfile", "")
        assert domain == "docker"

    def test_infer_domain_kubernetes(self) -> None:
        """Test domain inference for Kubernetes content."""
        domain = infer_domain("kubectl get pods", "")
        assert domain == "kubernetes"

        domain = infer_domain("Kubernetes deployment", "")
        assert domain == "kubernetes"

    def test_infer_domain_general(self) -> None:
        """Test domain inference for general content."""
        domain = infer_domain("общий вопрос", "ответ")
        assert domain == "general"


class TestContextFormatting:
    """Tests for context formatting functions."""

    def test_format_memories_empty(self) -> None:
        """Test formatting empty memories list."""
        result = format_memories_context([])
        assert "Нет релевантных знаний" in result

    def test_format_memories_facts(self) -> None:
        """Test formatting fact memories."""
        memory = SemanticMemory(
            id="mem-1",
            content="Docker — платформа контейнеризации",
            memory_type="fact",
            confidence=0.9,
        )
        scored = ScoredMemory(memory=memory, score=0.8, sources=["vector"])

        result = format_memories_context([scored])

        assert "[Факт]" in result
        assert "Docker" in result

    def test_format_memories_procedures(self) -> None:
        """Test formatting procedure memories."""
        memory = SemanticMemory(
            id="mem-1",
            content="Выполните docker system prune",
            memory_type="procedure",
            confidence=0.95,
        )
        scored = ScoredMemory(memory=memory, score=0.9, sources=["bm25"])

        result = format_memories_context([scored])

        assert "[Процедура]" in result
        assert "prune" in result

    def test_format_memories_low_confidence(self) -> None:
        """Test formatting memories with low confidence."""
        memory = SemanticMemory(
            id="mem-1",
            content="Возможно это работает так",
            memory_type="fact",
            confidence=0.5,
        )
        scored = ScoredMemory(memory=memory, score=0.7, sources=["vector"])

        result = format_memories_context([scored])

        assert "уверенность" in result

    def test_format_episodes_empty(self) -> None:
        """Test formatting empty episodes list."""
        result = format_episodes_context([])
        assert "Нет похожих" in result

    def test_format_episodes_successful(self, sample_episodic_memory) -> None:
        """Test formatting successful episodes."""
        sample_episodic_memory.success_count = 3
        sample_episodic_memory.failure_count = 0
        scored = ScoredEpisode(episode=sample_episodic_memory, score=0.85)

        result = format_episodes_context([scored])

        assert "✓" in result
        assert sample_episodic_memory.behavior_name in result


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_synthesis_result_creation(self) -> None:
        """Test creating a synthesis result."""
        behavior = Behavior(
            name="explain_concept",
            instruction="Explain the concept with examples",
            domain="docker",
        )

        result = SynthesisResult(
            answer="Docker is a container platform.",
            behavior=behavior,
            memories_used=["mem-1", "mem-2"],
            concepts_activated=["concept-1"],
            confidence=0.85,
            query="What is Docker?",
            importance=6.0,
        )

        assert result.answer == "Docker is a container platform."
        assert result.behavior.name == "explain_concept"
        assert len(result.memories_used) == 2
        assert result.confidence == 0.85


class TestResponseSynthesizer:
    """Tests for ResponseSynthesizer class."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM client."""
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="""Docker — это платформа для контейнеризации.

СТРАТЕГИЯ: [explain_concept] — Объяснить концепцию Docker""")
        return llm

    @pytest.fixture
    def mock_retrieval_result(self, sample_semantic_memory, sample_episodic_memory) -> RetrievalResult:
        """Create mock retrieval result."""
        return RetrievalResult(
            query="What is Docker?",
            query_embedding=[0.1] * 384,
            query_concepts=[],
            activated_concepts={"concept-1": 0.8},
            memories=[
                ScoredMemory(memory=sample_semantic_memory, score=0.9, sources=["vector"])
            ],
            episodes=[
                ScoredEpisode(episode=sample_episodic_memory, score=0.7)
            ],
            retrieval_sources={"vector": 1, "graph": 1},
        )

    @pytest.mark.asyncio
    async def test_synthesize_basic(self, mock_llm, mock_retrieval_result) -> None:
        """Test basic response synthesis."""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm)

        result = await synthesizer.synthesize(
            query="What is Docker?",
            retrieval=mock_retrieval_result,
        )

        assert "Docker" in result.answer
        assert result.behavior.name == "explain_concept"
        assert result.query == "What is Docker?"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_cleans_strategy_line(self, mock_llm, mock_retrieval_result) -> None:
        """Test that strategy line is removed from answer."""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm)

        result = await synthesizer.synthesize(
            query="What is Docker?",
            retrieval=mock_retrieval_result,
        )

        # Answer should not contain СТРАТЕГИЯ line
        assert "СТРАТЕГИЯ:" not in result.answer

    @pytest.mark.asyncio
    async def test_synthesize_estimates_importance(self, mock_llm, mock_retrieval_result) -> None:
        """Test importance estimation."""
        # Add more concepts to increase complexity
        mock_retrieval_result.query_concepts = [MagicMock() for _ in range(5)]
        mock_retrieval_result.activated_concepts = {f"c{i}": 0.5 for i in range(8)}

        synthesizer = ResponseSynthesizer(llm_client=mock_llm)

        result = await synthesizer.synthesize(
            query="A complex question about Docker containers and Kubernetes deployments",
            retrieval=mock_retrieval_result,
        )

        # Should have higher importance due to complexity
        assert result.importance > 5.0

    @pytest.mark.asyncio
    async def test_synthesize_estimates_confidence(self, mock_llm, mock_retrieval_result) -> None:
        """Test confidence estimation."""
        # Set high confidence on memory
        mock_retrieval_result.memories[0].memory.confidence = 0.95

        synthesizer = ResponseSynthesizer(llm_client=mock_llm)

        result = await synthesizer.synthesize(
            query="What is Docker?",
            retrieval=mock_retrieval_result,
        )

        assert result.confidence > 0.5
