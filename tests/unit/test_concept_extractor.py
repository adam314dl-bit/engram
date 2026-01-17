"""Unit tests for concept extractor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.ingestion.concept_extractor import (
    ConceptExtractor,
    normalize_concept_name,
    validate_concept_type,
    validate_relation_type,
)
from engram.ingestion.llm_client import LLMClient


class TestConceptExtractorHelpers:
    """Tests for concept extractor helper functions."""

    def test_normalize_concept_name(self) -> None:
        """Test concept name normalization."""
        assert normalize_concept_name("Docker") == "docker"
        assert normalize_concept_name("  KUBERNETES  ") == "kubernetes"
        assert normalize_concept_name("disk space") == "disk space"

    def test_validate_concept_type_valid(self) -> None:
        """Test valid concept types."""
        assert validate_concept_type("tool") == "tool"
        assert validate_concept_type("RESOURCE") == "resource"
        assert validate_concept_type("  action  ") == "action"

    def test_validate_concept_type_invalid(self) -> None:
        """Test invalid concept type defaults to general."""
        assert validate_concept_type("unknown") == "general"
        assert validate_concept_type("") == "general"

    def test_validate_relation_type_valid(self) -> None:
        """Test valid relation types."""
        assert validate_relation_type("uses") == "uses"
        assert validate_relation_type("CONTAINS") == "contains"

    def test_validate_relation_type_invalid(self) -> None:
        """Test invalid relation type defaults to related_to."""
        assert validate_relation_type("unknown") == "related_to"


class TestConceptExtractor:
    """Tests for ConceptExtractor class."""

    @pytest.mark.asyncio
    async def test_extract_concepts(self, mock_llm_client: LLMClient) -> None:
        """Test concept extraction."""
        extractor = ConceptExtractor(llm_client=mock_llm_client)

        result = await extractor.extract("Docker uses containers")

        assert len(result.concepts) == 3
        assert any(c.name == "docker" for c in result.concepts)
        assert len(result.relations) == 2

    @pytest.mark.asyncio
    async def test_extract_handles_error(self) -> None:
        """Test extraction handles LLM errors gracefully."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.generate_json = AsyncMock(side_effect=ValueError("Parse error"))

        extractor = ConceptExtractor(llm_client=mock_client)
        result = await extractor.extract("Some content")

        # Should return empty result on error
        assert len(result.concepts) == 0
        assert len(result.relations) == 0

    def test_get_or_create_concept_new(self) -> None:
        """Test creating new concept."""
        extractor = ConceptExtractor()
        concept = extractor.get_or_create_concept("docker", "tool")

        assert concept.name == "docker"
        assert concept.type == "tool"
        assert concept.id is not None

    def test_get_or_create_concept_cached(self) -> None:
        """Test getting cached concept."""
        extractor = ConceptExtractor()

        concept1 = extractor.get_or_create_concept("docker", "tool")
        concept2 = extractor.get_or_create_concept("DOCKER", "resource")  # Different case

        # Should return same concept
        assert concept1.id == concept2.id
        assert concept1.type == "tool"  # First type wins

    def test_clear_cache(self) -> None:
        """Test clearing concept cache."""
        extractor = ConceptExtractor()

        concept1 = extractor.get_or_create_concept("docker")
        extractor.clear_cache()
        concept2 = extractor.get_or_create_concept("docker")

        # After clearing, should get new concept
        assert concept1.id != concept2.id

    @pytest.mark.asyncio
    async def test_parse_concepts_filters_empty(self, mock_llm_client: LLMClient) -> None:
        """Test that empty concept names are filtered."""
        mock_llm_client.generate_json = AsyncMock(return_value={
            "concepts": [
                {"name": "", "type": "tool"},
                {"name": "valid", "type": "tool"},
                {"name": "   ", "type": "tool"},
            ],
            "relations": [],
        })

        extractor = ConceptExtractor(llm_client=mock_llm_client)
        result = await extractor.extract("content")

        # Only valid concept should be included
        assert len(result.concepts) == 1
        assert result.concepts[0].name == "valid"
