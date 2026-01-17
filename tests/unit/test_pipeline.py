"""Unit tests for retrieval pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engram.models import Concept, SemanticMemory, EpisodicMemory
from engram.retrieval.pipeline import RetrievalResult, RetrievalPipeline
from engram.retrieval.hybrid_search import ScoredMemory, ScoredEpisode
from engram.retrieval.spreading_activation import ActivationResult


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self) -> None:
        """Test basic RetrievalResult creation."""
        result = RetrievalResult(
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            query_concepts=[],
            activated_concepts={},
        )

        assert result.query == "test query"
        assert result.query_embedding == [0.1, 0.2, 0.3]
        assert result.memories == []
        assert result.episodes == []
        assert result.retrieval_sources == {}

    def test_retrieval_result_top_memories(self, sample_semantic_memory) -> None:
        """Test top_memories property."""
        scored_memory = ScoredMemory(
            memory=sample_semantic_memory,
            score=0.9,
            sources=["vector"],
        )

        result = RetrievalResult(
            query="test",
            query_embedding=[0.1],
            query_concepts=[],
            activated_concepts={},
            memories=[scored_memory],
        )

        assert len(result.top_memories) == 1
        assert result.top_memories[0] == sample_semantic_memory

    def test_retrieval_result_top_episodes(self, sample_episodic_memory) -> None:
        """Test top_episodes property."""
        scored_episode = ScoredEpisode(
            episode=sample_episodic_memory,
            score=0.8,
        )

        result = RetrievalResult(
            query="test",
            query_embedding=[0.1],
            query_concepts=[],
            activated_concepts={},
            episodes=[scored_episode],
        )

        assert len(result.top_episodes) == 1
        assert result.top_episodes[0] == sample_episodic_memory


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline class."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock Neo4j client."""
        db = MagicMock()
        db.get_concept_by_name = AsyncMock(return_value=None)
        db.vector_search_concepts = AsyncMock(return_value=[])
        db.get_memories_for_concepts = AsyncMock(return_value=[])
        db.update_memory_access = AsyncMock()
        return db

    @pytest.fixture
    def mock_spreading(self) -> MagicMock:
        """Create mock spreading activation."""
        spreading = MagicMock()
        spreading.activate = AsyncMock(return_value=ActivationResult(
            activations={"concept-1": 0.8, "concept-2": 0.5},
            activated_concepts=[],
            seed_concepts=["concept-1"],
            hops_performed=2,
        ))
        return spreading

    @pytest.fixture
    def mock_hybrid(self, sample_semantic_memory, sample_episodic_memory) -> MagicMock:
        """Create mock hybrid search."""
        hybrid = MagicMock()
        hybrid.search_memories = AsyncMock(return_value=[
            ScoredMemory(memory=sample_semantic_memory, score=0.9, sources=["vector", "graph"]),
        ])
        hybrid.search_similar_episodes = AsyncMock(return_value=[
            ScoredEpisode(episode=sample_episodic_memory, score=0.75),
        ])
        return hybrid

    @pytest.mark.asyncio
    async def test_pipeline_basic_retrieval(
        self,
        mock_db,
        mock_embedding_service,
        mock_llm_client,
        mock_spreading,
        mock_hybrid,
    ) -> None:
        """Test basic retrieval pipeline execution."""
        from engram.ingestion.concept_extractor import ConceptExtractor

        # Create mock concept extractor
        mock_extractor = MagicMock(spec=ConceptExtractor)

        class MockExtractionResult:
            concepts = [
                Concept(id="c1", name="docker", type="tool"),
            ]
            relations = []

        mock_extractor.extract = AsyncMock(return_value=MockExtractionResult())

        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
            concept_extractor=mock_extractor,
            spreading_activation=mock_spreading,
            hybrid_search=mock_hybrid,
        )

        result = await pipeline.retrieve(
            query="How do I use Docker containers?",
            top_k_memories=5,
        )

        # Verify result structure
        assert result.query == "How do I use Docker containers?"
        assert len(result.query_embedding) == 384  # Mock embedding dimension
        assert len(result.memories) == 1
        assert len(result.episodes) == 1

        # Verify methods were called
        mock_embedding_service.embed.assert_called_once()
        mock_extractor.extract.assert_called_once()
        mock_hybrid.search_memories.assert_called_once()
        mock_hybrid.search_similar_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_with_matched_concepts(
        self,
        mock_db,
        mock_embedding_service,
        mock_llm_client,
        mock_spreading,
        mock_hybrid,
        sample_concept,
    ) -> None:
        """Test pipeline when concepts match existing ones."""
        from engram.ingestion.concept_extractor import ConceptExtractor

        # Mock database to return existing concept
        mock_db.get_concept_by_name = AsyncMock(return_value=sample_concept)

        mock_extractor = MagicMock(spec=ConceptExtractor)

        class MockExtractionResult:
            concepts = [
                Concept(id="temp-id", name="docker", type="tool"),
            ]
            relations = []

        mock_extractor.extract = AsyncMock(return_value=MockExtractionResult())

        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
            concept_extractor=mock_extractor,
            spreading_activation=mock_spreading,
            hybrid_search=mock_hybrid,
        )

        result = await pipeline.retrieve(query="Docker help")

        # Should use the matched concept from database
        assert len(result.query_concepts) == 1
        assert result.query_concepts[0].id == sample_concept.id

        # Spreading activation should be called with matched concept ID
        mock_spreading.activate.assert_called_once()
        call_args = mock_spreading.activate.call_args
        assert sample_concept.id in call_args[0][0]

    @pytest.mark.asyncio
    async def test_pipeline_without_episodes(
        self,
        mock_db,
        mock_embedding_service,
        mock_llm_client,
        mock_spreading,
        mock_hybrid,
    ) -> None:
        """Test pipeline with episodes disabled."""
        from engram.ingestion.concept_extractor import ConceptExtractor

        mock_extractor = MagicMock(spec=ConceptExtractor)

        class MockExtractionResult:
            concepts = []
            relations = []

        mock_extractor.extract = AsyncMock(return_value=MockExtractionResult())

        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
            concept_extractor=mock_extractor,
            spreading_activation=mock_spreading,
            hybrid_search=mock_hybrid,
        )

        result = await pipeline.retrieve(
            query="test query",
            include_episodes=False,
        )

        # Episodes should not be searched
        mock_hybrid.search_similar_episodes.assert_not_called()
        assert result.episodes == []

    @pytest.mark.asyncio
    async def test_pipeline_tracks_retrieval_sources(
        self,
        mock_db,
        mock_embedding_service,
        mock_llm_client,
        mock_spreading,
        sample_semantic_memory,
    ) -> None:
        """Test that retrieval sources are tracked."""
        from engram.ingestion.concept_extractor import ConceptExtractor

        mock_extractor = MagicMock(spec=ConceptExtractor)

        class MockExtractionResult:
            concepts = []
            relations = []

        mock_extractor.extract = AsyncMock(return_value=MockExtractionResult())

        # Create hybrid mock with specific sources
        mock_hybrid = MagicMock()
        mock_hybrid.search_memories = AsyncMock(return_value=[
            ScoredMemory(memory=sample_semantic_memory, score=0.9, sources=["vector", "graph"]),
            ScoredMemory(
                memory=SemanticMemory(
                    id="mem-2",
                    content="another memory",
                    memory_type="fact",
                ),
                score=0.8,
                sources=["bm25"],
            ),
        ])
        mock_hybrid.search_similar_episodes = AsyncMock(return_value=[])

        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
            concept_extractor=mock_extractor,
            spreading_activation=mock_spreading,
            hybrid_search=mock_hybrid,
        )

        result = await pipeline.retrieve(query="test")

        # Should track all sources
        assert "vector" in result.retrieval_sources
        assert "graph" in result.retrieval_sources
        assert "bm25" in result.retrieval_sources
        assert result.retrieval_sources["vector"] == 1
        assert result.retrieval_sources["graph"] == 1
        assert result.retrieval_sources["bm25"] == 1

    @pytest.mark.asyncio
    async def test_pipeline_vector_fallback(
        self,
        mock_db,
        mock_embedding_service,
        mock_spreading,
        mock_hybrid,
        sample_concept,
    ) -> None:
        """Test vector search fallback when no concepts match."""
        from engram.ingestion.concept_extractor import ConceptExtractor

        # No name match, but vector search finds similar concept
        mock_db.get_concept_by_name = AsyncMock(return_value=None)
        mock_db.vector_search_concepts = AsyncMock(return_value=[
            (sample_concept, 0.75),  # Above 0.5 threshold
        ])

        mock_extractor = MagicMock(spec=ConceptExtractor)

        class MockExtractionResult:
            concepts = [Concept(id="temp", name="containr", type="tool")]  # Typo
            relations = []

        mock_extractor.extract = AsyncMock(return_value=MockExtractionResult())

        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
            concept_extractor=mock_extractor,
            spreading_activation=mock_spreading,
            hybrid_search=mock_hybrid,
        )

        result = await pipeline.retrieve(query="containr help")

        # Should fallback to vector search
        mock_db.vector_search_concepts.assert_called_once()
        # Should find the similar concept
        assert len(result.query_concepts) == 1
        assert result.query_concepts[0].id == sample_concept.id


class TestRetrieveForConcepts:
    """Tests for retrieve_for_concepts method."""

    @pytest.fixture
    def mock_db(self, sample_semantic_memory) -> MagicMock:
        """Create mock DB with memories."""
        db = MagicMock()
        db.get_memories_for_concepts = AsyncMock(return_value=[sample_semantic_memory])
        return db

    @pytest.mark.asyncio
    async def test_retrieve_for_concepts_basic(
        self,
        mock_db,
        mock_embedding_service,
        sample_semantic_memory,
    ) -> None:
        """Test basic concept-based retrieval."""
        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
        )

        results = await pipeline.retrieve_for_concepts(
            concept_ids=["concept-1"],
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].memory.id == sample_semantic_memory.id
        assert "graph" in results[0].sources

    @pytest.mark.asyncio
    async def test_retrieve_for_concepts_with_embedding(
        self,
        mock_db,
        mock_embedding_service,
    ) -> None:
        """Test retrieval with query embedding for relevance scoring."""
        # Create memory with embedding
        memory_with_emb = SemanticMemory(
            id="mem-emb",
            content="test content",
            memory_type="fact",
            importance=5.0,
            embedding=[0.8, 0.2, 0.0],
        )
        mock_db.get_memories_for_concepts = AsyncMock(return_value=[memory_with_emb])

        pipeline = RetrievalPipeline(
            db=mock_db,
            embedding_service=mock_embedding_service,
        )

        results = await pipeline.retrieve_for_concepts(
            concept_ids=["concept-1"],
            query_embedding=[1.0, 0.0, 0.0],
            top_k=5,
        )

        assert len(results) == 1
        # Score should incorporate both importance and relevance
        assert results[0].score > 0
