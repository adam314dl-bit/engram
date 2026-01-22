"""Unit tests for two-phase retrieval with LLM selection."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engram.models import Document, SemanticMemory
from engram.retrieval.hybrid_search import ScoredMemory
from engram.reasoning.selector import MemorySelector, SelectionResult


class TestMemorySelector:
    """Tests for LLM-based memory selection."""

    @pytest.fixture
    def sample_memories(self) -> list[ScoredMemory]:
        """Create sample scored memories."""
        return [
            ScoredMemory(
                memory=SemanticMemory(
                    id="mem_abc123",
                    content="Summary: Описание работы Docker | Keywords: docker, контейнер",
                    memory_type="fact",
                    importance=8.0,
                ),
                score=0.9,
                sources=["V", "B"],
            ),
            ScoredMemory(
                memory=SemanticMemory(
                    id="mem_def456",
                    content="Summary: Инструкция по установке | Keywords: установка, linux",
                    memory_type="procedure",
                    importance=7.0,
                ),
                score=0.8,
                sources=["V"],
            ),
            ScoredMemory(
                memory=SemanticMemory(
                    id="mem_ghi789",
                    content="Summary: Команды для очистки | Keywords: prune, очистка",
                    memory_type="procedure",
                    importance=6.0,
                ),
                score=0.7,
                sources=["B"],
            ),
        ]

    def test_parse_selected_ids_with_selected_line(self) -> None:
        """Test parsing SELECTED: line from LLM response."""
        selector = MemorySelector()
        valid_ids = ["mem_abc123", "mem_def456", "mem_ghi789"]

        response = "SELECTED: mem_abc123, mem_ghi789"
        result = selector._parse_selected_ids(response, valid_ids)

        assert result == ["mem_abc123", "mem_ghi789"]

    def test_parse_selected_ids_without_prefix(self) -> None:
        """Test parsing when LLM doesn't use SELECTED: prefix."""
        selector = MemorySelector()
        valid_ids = ["mem_abc123", "mem_def456"]

        response = "mem_abc123, mem_def456"
        result = selector._parse_selected_ids(response, valid_ids)

        assert result == ["mem_abc123", "mem_def456"]

    def test_parse_selected_ids_filters_invalid(self) -> None:
        """Test that invalid IDs are filtered out."""
        selector = MemorySelector()
        valid_ids = ["mem_abc123", "mem_def456"]

        response = "SELECTED: mem_abc123, mem_invalid, mem_def456"
        result = selector._parse_selected_ids(response, valid_ids)

        assert result == ["mem_abc123", "mem_def456"]

    def test_parse_selected_ids_deduplicates(self) -> None:
        """Test that duplicate IDs are removed."""
        selector = MemorySelector()
        valid_ids = ["mem_abc123"]

        response = "SELECTED: mem_abc123, mem_abc123, mem_abc123"
        result = selector._parse_selected_ids(response, valid_ids)

        assert result == ["mem_abc123"]

    def test_format_memories(self, sample_memories) -> None:
        """Test formatting memories for LLM prompt."""
        selector = MemorySelector()
        formatted = selector._format_memories(sample_memories)

        assert "[mem_abc123]" in formatted
        assert "[mem_def456]" in formatted
        assert "Docker" in formatted
        assert "установка" in formatted

    def test_select_with_mock_llm(self, sample_memories) -> None:
        """Test selection with mocked LLM."""
        import asyncio

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "SELECTED: mem_abc123, mem_ghi789"

        selector = MemorySelector(llm_client=mock_llm)

        async def run_test():
            return await selector.select(
                query="Что такое Docker?",
                candidates=sample_memories,
            )

        result = asyncio.get_event_loop().run_until_complete(run_test())

        assert isinstance(result, SelectionResult)
        assert result.selected_ids == ["mem_abc123", "mem_ghi789"]
        assert len(result.all_candidate_ids) == 3
        assert result.selection_ratio == 2 / 3

    def test_select_empty_candidates(self) -> None:
        """Test selection with no candidates."""
        import asyncio

        selector = MemorySelector()

        async def run_test():
            return await selector.select(
                query="Test",
                candidates=[],
            )

        result = asyncio.get_event_loop().run_until_complete(run_test())

        assert result.selected_ids == []
        assert result.all_candidate_ids == []
        assert result.selection_ratio == 0.0

    def test_select_llm_failure_fallback(self, sample_memories) -> None:
        """Test fallback when LLM fails."""
        import asyncio

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        selector = MemorySelector(llm_client=mock_llm)

        async def run_test():
            return await selector.select(
                query="Test",
                candidates=sample_memories,
            )

        result = asyncio.get_event_loop().run_until_complete(run_test())

        # Should fallback to top 10 by score
        assert len(result.selected_ids) == 3  # All 3 since we have only 3


class TestMemoryExtractionFormat:
    """Tests for the new memory extraction format."""

    def test_new_format_parsing(self) -> None:
        """Test parsing the new summary+keywords format."""
        from engram.ingestion.memory_extractor import MemoryExtractor

        extractor = MemoryExtractor()

        # New format: MEMORY|summary|keywords|type|concepts|importance
        text = """MEMORY|Описание работы Docker с контейнерами|docker, контейнер, изоляция|fact|docker,контейнер|8
MEMORY|Инструкция по очистке дискового пространства|docker system prune, очистка|procedure|docker,диск|7"""

        result = extractor._parse_memory_lines(text, doc_id="doc_123")

        assert len(result.memories) == 2

        # Check first memory
        mem1 = result.memories[0]
        assert "Summary:" in mem1.content
        assert "Keywords:" in mem1.content
        assert "Docker с контейнерами" in mem1.content
        assert "docker, контейнер, изоляция" in mem1.content
        assert mem1.memory_type == "fact"
        assert mem1.importance == 8.0

        # Check second memory
        mem2 = result.memories[1]
        assert mem2.memory_type == "procedure"
        assert mem2.importance == 7.0

    def test_old_format_still_works(self) -> None:
        """Test that old format is still parsed correctly."""
        from engram.ingestion.memory_extractor import MemoryExtractor

        extractor = MemoryExtractor()

        # Old format: MEMORY|content|type|concepts|importance
        text = "MEMORY|Docker использует контейнеры для изоляции|fact|docker,контейнер|8"

        result = extractor._parse_memory_lines(text, doc_id="doc_123")

        assert len(result.memories) == 1
        mem = result.memories[0]
        assert mem.content == "Docker использует контейнеры для изоляции"
        assert mem.memory_type == "fact"
        assert mem.importance == 8.0


class TestDocumentFormatting:
    """Tests for document context formatting."""

    def test_format_documents_context(self) -> None:
        """Test formatting documents for LLM context."""
        from engram.reasoning.synthesizer import format_documents_context

        docs = [
            Document(
                id="doc_1",
                title="Docker Guide",
                content="Docker is a containerization platform...",
            ),
            Document(
                id="doc_2",
                title="Kubernetes Intro",
                content="Kubernetes orchestrates containers...",
            ),
        ]

        formatted = format_documents_context(docs)

        assert "Документ 1: Docker Guide" in formatted
        assert "Документ 2: Kubernetes Intro" in formatted
        assert "Docker is a containerization platform" in formatted
        assert "Kubernetes orchestrates containers" in formatted

    def test_format_documents_truncation(self) -> None:
        """Test that long documents are truncated."""
        from engram.reasoning.synthesizer import format_documents_context

        # Create a very long document
        long_content = "x" * 20000
        docs = [
            Document(id="doc_1", title="Long Doc", content=long_content),
        ]

        formatted = format_documents_context(docs, max_chars=1000)

        assert len(formatted) < 2000
        assert "сокращён" in formatted

    def test_format_empty_documents(self) -> None:
        """Test formatting with no documents."""
        from engram.reasoning.synthesizer import format_documents_context

        formatted = format_documents_context([])

        assert "Нет релевантных документов" in formatted


class TestRetrievalPipelineCandidates:
    """Tests for candidate retrieval pipeline."""

    def test_retrieve_candidates_method_exists(self) -> None:
        """Test that retrieve_candidates method is available."""
        from engram.retrieval.pipeline import RetrievalPipeline

        # Just check the method exists and has correct signature
        assert hasattr(RetrievalPipeline, "retrieve_candidates")

        # Check it's a coroutine function
        import inspect
        assert inspect.iscoroutinefunction(RetrievalPipeline.retrieve_candidates)


class TestReasoningPipelineWithDocuments:
    """Tests for two-phase reasoning pipeline."""

    def test_reason_with_documents_method_exists(self) -> None:
        """Test that reason_with_documents method is available."""
        from engram.reasoning.pipeline import ReasoningPipeline

        assert hasattr(ReasoningPipeline, "reason_with_documents")

        import inspect
        assert inspect.iscoroutinefunction(ReasoningPipeline.reason_with_documents)

    def test_reasoning_result_has_new_fields(self) -> None:
        """Test that ReasoningResult has selection_result and source_documents."""
        from engram.reasoning.pipeline import ReasoningResult

        import dataclasses
        fields = {f.name for f in dataclasses.fields(ReasoningResult)}

        assert "selection_result" in fields
        assert "source_documents" in fields
