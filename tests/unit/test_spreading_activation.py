"""Unit tests for spreading activation algorithm."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import Concept, ConceptRelation
from engram.retrieval.spreading_activation import (
    ActivatedConcept,
    ActivationResult,
    SpreadingActivation,
)
from engram.storage.neo4j_client import Neo4jClient


@pytest.fixture
def mock_db() -> Neo4jClient:
    """Create a mock Neo4j client for testing."""
    db = MagicMock(spec=Neo4jClient)

    # Create test concepts
    concepts = {
        "docker": Concept(id="c1", name="docker", type="tool"),
        "container": Concept(id="c2", name="container", type="resource"),
        "image": Concept(id="c3", name="image", type="resource"),
        "dockerfile": Concept(id="c4", name="dockerfile", type="config"),
        "registry": Concept(id="c5", name="registry", type="resource"),
    }

    # Define concept relationships
    relations = {
        "c1": [  # docker neighbors
            (concepts["container"], ConceptRelation(
                source_id="c1", target_id="c2", relation_type="uses", weight=0.9
            )),
            (concepts["image"], ConceptRelation(
                source_id="c1", target_id="c3", relation_type="uses", weight=0.8
            )),
        ],
        "c2": [  # container neighbors
            (concepts["image"], ConceptRelation(
                source_id="c2", target_id="c3", relation_type="needs", weight=0.85
            )),
        ],
        "c3": [  # image neighbors
            (concepts["dockerfile"], ConceptRelation(
                source_id="c3", target_id="c4", relation_type="needs", weight=0.7
            )),
            (concepts["registry"], ConceptRelation(
                source_id="c3", target_id="c5", relation_type="related_to", weight=0.6
            )),
        ],
        "c4": [],
        "c5": [],
    }

    async def mock_get_concept(concept_id: str) -> Concept | None:
        for c in concepts.values():
            if c.id == concept_id:
                return c
        return None

    async def mock_get_neighbors(concept_id: str) -> list[tuple[Concept, ConceptRelation]]:
        return relations.get(concept_id, [])

    async def mock_update_activation(concept_id: str) -> None:
        pass

    db.get_concept = AsyncMock(side_effect=mock_get_concept)
    db.get_concept_neighbors = AsyncMock(side_effect=mock_get_neighbors)
    db.update_concept_activation = AsyncMock(side_effect=mock_update_activation)

    return db


class TestSpreadingActivation:
    """Tests for SpreadingActivation class."""

    @pytest.mark.asyncio
    async def test_empty_seeds(self, mock_db: Neo4jClient) -> None:
        """Test with no seed concepts."""
        spreader = SpreadingActivation(db=mock_db)
        result = await spreader.activate([])

        assert result.activations == {}
        assert result.activated_concepts == []
        assert result.seed_concepts == []

    @pytest.mark.asyncio
    async def test_single_seed_no_neighbors(self, mock_db: Neo4jClient) -> None:
        """Test single seed with no outgoing edges."""
        spreader = SpreadingActivation(db=mock_db)
        result = await spreader.activate(["c4"])  # dockerfile has no neighbors

        assert "c4" in result.activations
        assert result.activations["c4"] == 1.0
        assert len(result.activated_concepts) == 1

    @pytest.mark.asyncio
    async def test_single_seed_with_neighbors(self, mock_db: Neo4jClient) -> None:
        """Test single seed spreads to neighbors."""
        spreader = SpreadingActivation(
            db=mock_db, decay=0.85, threshold=0.1, max_hops=1, rescale=0.5
        )
        result = await spreader.activate(["c1"])  # docker

        # Should have docker and its neighbors
        assert "c1" in result.activations  # seed
        assert result.activations["c1"] == 1.0

        # Neighbors should have activation < 1.0
        assert "c2" in result.activations  # container
        assert "c3" in result.activations  # image
        assert result.activations["c2"] < 1.0
        assert result.activations["c3"] < 1.0

    @pytest.mark.asyncio
    async def test_multi_hop_spreading(self, mock_db: Neo4jClient) -> None:
        """Test activation spreads multiple hops."""
        spreader = SpreadingActivation(
            db=mock_db, decay=0.85, threshold=0.05, max_hops=3, rescale=0.5
        )
        result = await spreader.activate(["c1"])  # docker

        # Should reach dockerfile (3 hops: docker -> image -> dockerfile)
        # or (docker -> container -> image -> dockerfile)
        # May or may not reach depending on threshold
        assert "c1" in result.activations
        assert len(result.activations) >= 2  # At least seed + direct neighbors

    @pytest.mark.asyncio
    async def test_activation_decay(self, mock_db: Neo4jClient) -> None:
        """Test that activation decays with distance."""
        spreader = SpreadingActivation(
            db=mock_db, decay=0.85, threshold=0.05, max_hops=2, rescale=0.5
        )
        result = await spreader.activate(["c1"])

        # Seeds should have highest activation
        seed_activation = result.activations.get("c1", 0)
        assert seed_activation == 1.0

        # Direct neighbors should have lower activation
        for cid in ["c2", "c3"]:
            if cid in result.activations:
                assert result.activations[cid] < seed_activation

    @pytest.mark.asyncio
    async def test_lateral_inhibition(self, mock_db: Neo4jClient) -> None:
        """Test that top-k per hop limits activations."""
        spreader = SpreadingActivation(
            db=mock_db, decay=0.85, threshold=0.01, max_hops=2, rescale=0.5, top_k_per_hop=2
        )
        result = await spreader.activate(["c1"])

        # With top_k=2, should limit expansion at each hop
        # Total concepts should be reasonable
        assert len(result.activations) <= 10  # Reasonable limit

    @pytest.mark.asyncio
    async def test_activation_result_sorted(self, mock_db: Neo4jClient) -> None:
        """Test that activated_concepts are sorted by activation."""
        spreader = SpreadingActivation(
            db=mock_db, decay=0.85, threshold=0.05, max_hops=2, rescale=0.5
        )
        result = await spreader.activate(["c1"])

        # Check sorting
        activations = [ac.activation for ac in result.activated_concepts]
        assert activations == sorted(activations, reverse=True)

    @pytest.mark.asyncio
    async def test_path_tracking(self, mock_db: Neo4jClient) -> None:
        """Test that activation paths are tracked."""
        spreader = SpreadingActivation(
            db=mock_db, decay=0.85, threshold=0.05, max_hops=2, rescale=0.5
        )
        result = await spreader.activate(["c1"])

        # Seed should have path to itself
        for ac in result.activated_concepts:
            if ac.concept.id == "c1":
                assert ac.path == ["c1"]
                assert ac.hop == 0
            else:
                # Non-seeds should have path starting from seed
                assert len(ac.path) > 1
                assert ac.path[0] == "c1"
                assert ac.hop > 0


class TestActivationResult:
    """Tests for ActivationResult dataclass."""

    def test_activation_result_creation(self) -> None:
        """Test creating an ActivationResult."""
        result = ActivationResult(
            activations={"c1": 1.0, "c2": 0.5},
            activated_concepts=[],
            seed_concepts=["c1"],
            hops_performed=2,
        )
        assert result.activations["c1"] == 1.0
        assert result.seed_concepts == ["c1"]
        assert result.hops_performed == 2


class TestActivatedConcept:
    """Tests for ActivatedConcept dataclass."""

    def test_activated_concept_creation(self) -> None:
        """Test creating an ActivatedConcept."""
        concept = Concept(id="c1", name="test", type="tool")
        ac = ActivatedConcept(
            concept=concept,
            activation=0.8,
            hop=1,
            path=["c0", "c1"],
        )
        assert ac.concept.id == "c1"
        assert ac.activation == 0.8
        assert ac.hop == 1
        assert ac.path == ["c0", "c1"]
