"""Unit tests for embeddings service."""

import pytest

from engram.retrieval.embeddings import (
    cosine_similarity,
    cosine_similarity_batch,
)


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""

    def test_cosine_similarity_identical(self) -> None:
        """Test similarity of identical vectors."""
        vec = [1.0, 0.0, 0.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_cosine_similarity_opposite(self) -> None:
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.0001

    def test_cosine_similarity_partial(self) -> None:
        """Test partial similarity."""
        vec1 = [1.0, 1.0]
        vec2 = [1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        # cos(45°) ≈ 0.707
        assert 0.7 < sim < 0.72

    def test_cosine_similarity_batch_empty(self) -> None:
        """Test batch similarity with empty vectors."""
        query = [1.0, 0.0]
        similarities = cosine_similarity_batch(query, [])
        assert similarities == []

    def test_cosine_similarity_batch(self) -> None:
        """Test batch similarity computation."""
        query = [1.0, 0.0, 0.0]
        vectors = [
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.5, 0.5, 0.0],  # partial
        ]
        similarities = cosine_similarity_batch(query, vectors)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 0.0001  # identical
        assert abs(similarities[1]) < 0.0001  # orthogonal
        assert 0.7 < similarities[2] < 0.72  # partial


class TestEmbeddingService:
    """Tests for EmbeddingService class (using mock)."""

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, mock_embedding_service) -> None:
        """Test that embed returns a list of floats."""
        embedding = await mock_embedding_service.embed("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_embedding_service) -> None:
        """Test batch embedding."""
        texts = ["text one", "text two", "text three"]
        embeddings = await mock_embedding_service.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, mock_embedding_service) -> None:
        """Test that same text produces same embedding."""
        text = "deterministic test"
        emb1 = await mock_embedding_service.embed(text)
        emb2 = await mock_embedding_service.embed(text)

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, mock_embedding_service) -> None:
        """Test that different texts produce different embeddings."""
        emb1 = await mock_embedding_service.embed("first text")
        emb2 = await mock_embedding_service.embed("second text")

        # At least some values should differ
        assert emb1 != emb2
