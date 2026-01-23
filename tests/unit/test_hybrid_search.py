"""Unit tests for hybrid search module."""

from datetime import datetime, timedelta

import pytest

from engram.models import SemanticMemory, EpisodicMemory
from engram.retrieval.fusion import reciprocal_rank_fusion
from engram.retrieval.hybrid_search import (
    ScoredMemory,
    ScoredEpisode,
    hours_since,
    HybridSearch,
)


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def _scores_dict(self, results: list) -> dict[str, float]:
        """Convert FusedResult list to dict for easier testing."""
        return {r.id: r.fused_score for r in results}

    def test_rrf_single_list(self) -> None:
        """Test RRF with a single ranked list."""
        ranked_list = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        results = reciprocal_rank_fusion([ranked_list])
        scores = self._scores_dict(results)

        # RRF score = 1/(k+rank+1), k=60 by default
        assert "a" in scores
        assert "b" in scores
        assert "c" in scores
        # First item has highest score
        assert scores["a"] > scores["b"] > scores["c"]

    def test_rrf_multiple_lists(self) -> None:
        """Test RRF with multiple ranked lists."""
        list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        list2 = [("b", 0.95), ("a", 0.85), ("d", 0.75)]

        results = reciprocal_rank_fusion([list1, list2])
        scores = self._scores_dict(results)

        # 'b' is #1 in list2 and #2 in list1, should have high score
        # 'a' is #1 in list1 and #2 in list2
        # Both should have similar scores
        assert "a" in scores
        assert "b" in scores
        assert "c" in scores
        assert "d" in scores
        # b and a should be close since they're both high in both lists
        assert abs(scores["a"] - scores["b"]) < 0.01

    def test_rrf_empty_lists(self) -> None:
        """Test RRF with empty input."""
        results = reciprocal_rank_fusion([])
        assert results == []

    def test_rrf_disjoint_lists(self) -> None:
        """Test RRF with lists that have no overlap."""
        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("c", 0.9), ("d", 0.8)]

        results = reciprocal_rank_fusion([list1, list2])
        scores = self._scores_dict(results)

        assert len(scores) == 4
        # All items should have similar scores since they appear in only one list
        assert abs(scores["a"] - scores["c"]) < 0.001

    def test_rrf_custom_k(self) -> None:
        """Test RRF with custom k parameter."""
        ranked_list = [("a", 0.9), ("b", 0.8)]

        # With k=10, scores are higher overall
        results_k10 = reciprocal_rank_fusion([ranked_list], k=10)
        scores_k10 = self._scores_dict(results_k10)
        # With k=100, scores are lower overall
        results_k100 = reciprocal_rank_fusion([ranked_list], k=100)
        scores_k100 = self._scores_dict(results_k100)

        assert scores_k10["a"] > scores_k100["a"]


class TestHoursSince:
    """Tests for hours_since function."""

    def test_hours_since_now(self) -> None:
        """Test hours since now is approximately 0."""
        result = hours_since(datetime.utcnow())
        assert result < 0.01  # Less than a minute

    def test_hours_since_one_hour_ago(self) -> None:
        """Test hours since one hour ago."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        result = hours_since(one_hour_ago)
        assert 0.99 < result < 1.01

    def test_hours_since_one_day_ago(self) -> None:
        """Test hours since one day ago."""
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        result = hours_since(one_day_ago)
        assert 23.9 < result < 24.1

    def test_hours_since_none(self) -> None:
        """Test hours since None returns large value."""
        result = hours_since(None)
        assert result == 1000.0


class TestScoredMemory:
    """Tests for ScoredMemory dataclass."""

    def test_scored_memory_creation(self, sample_semantic_memory) -> None:
        """Test creating a scored memory."""
        scored = ScoredMemory(
            memory=sample_semantic_memory,
            score=0.85,
            sources=["vector", "bm25"],
        )

        assert scored.memory == sample_semantic_memory
        assert scored.score == 0.85
        assert scored.sources == ["vector", "bm25"]

    def test_scored_memory_default_sources(self, sample_semantic_memory) -> None:
        """Test default empty sources."""
        scored = ScoredMemory(memory=sample_semantic_memory, score=0.5)
        assert scored.sources == []


class TestScoredEpisode:
    """Tests for ScoredEpisode dataclass."""

    def test_scored_episode_creation(self, sample_episodic_memory) -> None:
        """Test creating a scored episode."""
        scored = ScoredEpisode(
            episode=sample_episodic_memory,
            score=0.75,
        )

        assert scored.episode == sample_episodic_memory
        assert scored.score == 0.75


class TestHybridSearchReranking:
    """Tests for hybrid search reranking logic."""

    @pytest.fixture
    def memories_dict(self) -> dict[str, SemanticMemory]:
        """Create test memories dictionary."""
        return {
            "mem-1": SemanticMemory(
                id="mem-1",
                content="High importance, recent memory",
                memory_type="fact",
                importance=9.0,
                last_accessed=datetime.utcnow(),
                embedding=[0.8, 0.2, 0.0],
            ),
            "mem-2": SemanticMemory(
                id="mem-2",
                content="Low importance, old memory",
                memory_type="fact",
                importance=3.0,
                last_accessed=datetime.utcnow() - timedelta(days=7),
                embedding=[0.1, 0.9, 0.0],
            ),
            "mem-3": SemanticMemory(
                id="mem-3",
                content="Medium importance, no embedding",
                memory_type="fact",
                importance=5.0,
                last_accessed=datetime.utcnow() - timedelta(hours=1),
                embedding=None,
            ),
        }

    def test_rerank_scores_importance(self, memories_dict) -> None:
        """Test that importance affects final score."""
        from unittest.mock import MagicMock

        # Create HybridSearch with mock db
        mock_db = MagicMock()
        search = HybridSearch(db=mock_db)

        fused_scores = {"mem-1": 0.5, "mem-2": 0.5, "mem-3": 0.5}
        query_embedding = [1.0, 0.0, 0.0]
        sources = {"mem-1": ["vector"], "mem-2": ["vector"], "mem-3": ["vector"]}

        results = search._rerank_memories(
            memories=memories_dict,
            fused_scores=fused_scores,
            query_embedding=query_embedding,
            sources=sources,
        )

        # Results should be sorted by score
        assert results[0].memory.id == "mem-1"  # Highest importance and most recent
        assert len(results) == 3

    def test_rerank_considers_recency(self, memories_dict) -> None:
        """Test that recency affects final score."""
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        search = HybridSearch(db=mock_db)

        # Set equal importance and RRF scores
        memories_dict["mem-1"].importance = 5.0
        memories_dict["mem-2"].importance = 5.0

        fused_scores = {"mem-1": 0.5, "mem-2": 0.5}
        query_embedding = [0.5, 0.5, 0.0]  # Similar to both embeddings
        sources = {"mem-1": ["vector"], "mem-2": ["vector"]}

        # Remove mem-3 for this test
        del memories_dict["mem-3"]

        results = search._rerank_memories(
            memories=memories_dict,
            fused_scores=fused_scores,
            query_embedding=query_embedding,
            sources=sources,
        )

        # mem-1 is more recent, should rank higher
        assert results[0].memory.id == "mem-1"

    def test_rerank_handles_no_embedding(self, memories_dict) -> None:
        """Test reranking handles memories without embeddings."""
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        search = HybridSearch(db=mock_db)

        # Only use mem-3 which has no embedding
        memories = {"mem-3": memories_dict["mem-3"]}
        fused_scores = {"mem-3": 0.5}
        query_embedding = [1.0, 0.0, 0.0]
        sources = {"mem-3": ["bm25"]}

        results = search._rerank_memories(
            memories=memories,
            fused_scores=fused_scores,
            query_embedding=query_embedding,
            sources=sources,
        )

        assert len(results) == 1
        # Should use default relevance of 0.5
        assert results[0].score > 0
