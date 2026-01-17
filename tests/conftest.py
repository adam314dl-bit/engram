"""Pytest configuration and fixtures."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import Settings
from engram.ingestion.llm_client import LLMClient
from engram.models import Concept, Document, EpisodicMemory, SemanticMemory
from engram.retrieval.embeddings import EmbeddingService


# Test data directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_DOCS_DIR = TEST_FIXTURES_DIR / "mock_docs"


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with defaults."""
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test_password",
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen3:8b",
    )


@pytest.fixture
def mock_llm_client() -> LLMClient:
    """Mock LLM client for testing without actual LLM."""
    client = MagicMock(spec=LLMClient)

    # Mock concept extraction response
    async def mock_generate_json(*args, **kwargs):
        return {
            "concepts": [
                {"name": "docker", "type": "tool", "description": "контейнеризация"},
                {"name": "container", "type": "resource", "description": "изолированная среда"},
                {"name": "image", "type": "resource", "description": "шаблон контейнера"},
            ],
            "relations": [
                {"source": "docker", "target": "container", "type": "uses"},
                {"source": "container", "target": "image", "type": "needs"},
            ],
        }

    client.generate_json = AsyncMock(side_effect=mock_generate_json)
    client.generate = AsyncMock(return_value="Test response")
    client.close = AsyncMock()

    return client


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Mock embedding service for testing without loading models."""
    service = MagicMock(spec=EmbeddingService)
    service.dimensions = 384

    # Return consistent fake embeddings
    def fake_embed_sync(text: str) -> list[float]:
        # Generate deterministic embedding based on text hash
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        # MD5 produces 32 hex chars, giving us 16 floats, pad to 384
        return [float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)] + [0.0] * (384 - 16)

    def fake_embed_batch_sync(texts: list[str]) -> list[list[float]]:
        return [fake_embed_sync(t) for t in texts]

    async def fake_embed(text: str) -> list[float]:
        return fake_embed_sync(text)

    async def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        return fake_embed_batch_sync(texts)

    service.embed = AsyncMock(side_effect=fake_embed)
    service.embed_batch = AsyncMock(side_effect=fake_embed_batch)
    service.embed_sync = MagicMock(side_effect=fake_embed_sync)
    service.embed_batch_sync = MagicMock(side_effect=fake_embed_batch_sync)

    return service


@pytest.fixture
def sample_concept() -> Concept:
    """Sample concept for testing."""
    return Concept(
        id="concept-docker-123",
        name="docker",
        type="tool",
        description="Container platform",
    )


@pytest.fixture
def sample_semantic_memory() -> SemanticMemory:
    """Sample semantic memory for testing."""
    return SemanticMemory(
        id="memory-123",
        content="Docker использует контейнеры для изоляции приложений",
        concept_ids=["concept-docker-123", "concept-container-456"],
        memory_type="fact",
        importance=8.0,
    )


@pytest.fixture
def sample_episodic_memory() -> EpisodicMemory:
    """Sample episodic memory for testing."""
    return EpisodicMemory(
        id="episode-123",
        query="Как освободить место на диске Docker?",
        concepts_activated=["concept-docker-123", "concept-disk-789"],
        memories_used=["memory-123"],
        behavior_name="check_disk_usage",
        behavior_instruction="Проверить использование диска и выполнить очистку",
        domain="docker",
        answer_summary="Используйте docker system prune",
    )


@pytest.fixture
def sample_document() -> Document:
    """Sample document for testing."""
    return Document(
        id="doc-123",
        title="Docker Basics",
        content="Docker — это платформа контейнеризации...",
        doc_type="markdown",
    )


@pytest.fixture
def mock_docs_path() -> Path:
    """Path to mock documents directory."""
    return MOCK_DOCS_DIR
