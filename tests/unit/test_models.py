"""Unit tests for data models."""

from datetime import datetime

import pytest

from engram.models import (
    Concept,
    ConceptRelation,
    Document,
    EpisodicMemory,
    SemanticMemory,
)


class TestConcept:
    """Tests for Concept model."""

    def test_create_concept(self) -> None:
        """Test creating a concept."""
        concept = Concept(
            id="test-id",
            name="docker",
            type="tool",
            description="Container platform",
        )
        assert concept.id == "test-id"
        assert concept.name == "docker"
        assert concept.type == "tool"
        assert concept.activation_count == 0

    def test_concept_to_dict(self, sample_concept: Concept) -> None:
        """Test converting concept to dictionary."""
        data = sample_concept.to_dict()
        assert data["id"] == sample_concept.id
        assert data["name"] == sample_concept.name
        assert data["type"] == sample_concept.type

    def test_concept_from_dict(self) -> None:
        """Test creating concept from dictionary."""
        data = {
            "id": "test-id",
            "name": "kubernetes",
            "type": "tool",
            "description": "Container orchestration",
            "level": 2,
        }
        concept = Concept.from_dict(data)
        assert concept.id == "test-id"
        assert concept.name == "kubernetes"
        assert concept.type == "tool"


class TestSemanticMemory:
    """Tests for SemanticMemory model."""

    def test_create_memory(self) -> None:
        """Test creating a semantic memory."""
        memory = SemanticMemory(
            id="mem-1",
            content="Docker uses containers",
            concept_ids=["c1", "c2"],
            memory_type="fact",
        )
        assert memory.id == "mem-1"
        assert memory.content == "Docker uses containers"
        assert len(memory.concept_ids) == 2

    def test_memory_is_valid(self, sample_semantic_memory: SemanticMemory) -> None:
        """Test memory validity check."""
        assert sample_semantic_memory.is_valid()

        # Test superseded memory
        sample_semantic_memory.status = "superseded"
        assert not sample_semantic_memory.is_valid()

    def test_memory_to_dict(self, sample_semantic_memory: SemanticMemory) -> None:
        """Test converting memory to dictionary."""
        data = sample_semantic_memory.to_dict()
        assert data["id"] == sample_semantic_memory.id
        assert data["content"] == sample_semantic_memory.content
        assert data["memory_type"] == sample_semantic_memory.memory_type


class TestEpisodicMemory:
    """Tests for EpisodicMemory model."""

    def test_create_episode(self) -> None:
        """Test creating an episodic memory."""
        episode = EpisodicMemory(
            id="ep-1",
            query="How to free disk space?",
            behavior_name="check_disk_usage",
            behavior_instruction="Check disk and clean up",
        )
        assert episode.id == "ep-1"
        assert episode.success_count == 0
        assert episode.failure_count == 0

    def test_episode_success_rate(self, sample_episodic_memory: EpisodicMemory) -> None:
        """Test success rate calculation."""
        # Initially 0 since no feedback
        assert sample_episodic_memory.success_rate == 0.0

        sample_episodic_memory.success_count = 3
        sample_episodic_memory.failure_count = 1
        assert sample_episodic_memory.success_rate == 0.75

    def test_episode_is_successful(self, sample_episodic_memory: EpisodicMemory) -> None:
        """Test is_successful property."""
        sample_episodic_memory.success_count = 5
        sample_episodic_memory.failure_count = 2
        assert sample_episodic_memory.is_successful

        sample_episodic_memory.failure_count = 6
        assert not sample_episodic_memory.is_successful


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self) -> None:
        """Test creating a document."""
        doc = Document(
            id="doc-1",
            title="Test Doc",
            content="Some content",
        )
        assert doc.id == "doc-1"
        assert doc.status == "pending"
        assert doc.doc_type == "markdown"

    def test_document_to_dict(self, sample_document: Document) -> None:
        """Test converting document to dictionary."""
        data = sample_document.to_dict()
        assert data["id"] == sample_document.id
        assert data["title"] == sample_document.title


class TestConceptRelation:
    """Tests for ConceptRelation model."""

    def test_create_relation(self) -> None:
        """Test creating a concept relation."""
        relation = ConceptRelation(
            source_id="c1",
            target_id="c2",
            relation_type="uses",
            weight=0.8,
        )
        assert relation.source_id == "c1"
        assert relation.target_id == "c2"
        assert relation.relation_type == "uses"
        assert relation.weight == 0.8

    def test_relation_to_dict(self) -> None:
        """Test converting relation to dictionary."""
        relation = ConceptRelation(
            source_id="c1",
            target_id="c2",
            relation_type="contains",
        )
        data = relation.to_dict()
        assert data["source_id"] == "c1"
        assert data["target_id"] == "c2"
        assert data["relation_type"] == "contains"
