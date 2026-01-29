"""Spreading activation algorithm for concept network traversal.

Based on research: decay 0.85, max 3 hops, query-oriented edge filtering.
Implements brain-like associative retrieval through concept networks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from engram.config import settings
from engram.models import Concept, ConceptRelation
from engram.retrieval.embeddings import cosine_similarity
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class ActivatedConcept:
    """A concept with its activation level and path info."""

    concept: Concept
    activation: float
    hop: int  # How many hops from seed
    path: list[str] = field(default_factory=list)  # Concept IDs in activation path


@dataclass
class ActivationResult:
    """Result of spreading activation."""

    activations: dict[str, float]  # concept_id -> activation level
    activated_concepts: list[ActivatedConcept]  # Sorted by activation (descending)
    seed_concepts: list[str]  # Original seed concept IDs
    hops_performed: int


class SpreadingActivation:
    """
    Spreading activation through concept network.

    Algorithm:
    1. Initialize seeds with activation = 1.0
    2. For each hop:
       a. Get neighbors of active concepts
       b. Filter edges by query relevance (cosine sim > threshold)
       c. Transfer: neighbor += current × edge_weight × decay × rescale
       d. Apply lateral inhibition (keep top-k per hop)
    3. Return concept_id -> activation mapping
    """

    def __init__(
        self,
        db: Neo4jClient,
        decay: float | None = None,
        threshold: float | None = None,
        max_hops: int | None = None,
        rescale: float | None = None,
        top_k_per_hop: int | None = None,
        semantic_edge_boost: float | None = None,
    ) -> None:
        self.db = db
        self.decay = decay or settings.activation_decay
        self.threshold = threshold or settings.activation_threshold
        self.max_hops = max_hops or settings.activation_max_hops
        self.rescale = rescale or settings.activation_rescale
        self.top_k_per_hop = top_k_per_hop or settings.activation_top_k_per_hop
        # v4.4: Boost for semantic + universal edges
        self.semantic_edge_boost = semantic_edge_boost or settings.semantic_edge_boost

    async def activate(
        self,
        seed_concept_ids: list[str],
        query_embedding: list[float] | None = None,
    ) -> ActivationResult:
        """
        Spread activation through concept network from seed concepts.

        Args:
            seed_concept_ids: IDs of concepts to start activation from
            query_embedding: Optional query embedding for relevance filtering

        Returns:
            ActivationResult with activation levels for all reached concepts
        """
        if not seed_concept_ids:
            return ActivationResult(
                activations={},
                activated_concepts=[],
                seed_concepts=[],
                hops_performed=0,
            )

        # Initialize activation with seeds at 1.0
        activation: dict[str, float] = {cid: 1.0 for cid in seed_concept_ids}
        paths: dict[str, list[str]] = {cid: [cid] for cid in seed_concept_ids}
        hop_levels: dict[str, int] = {cid: 0 for cid in seed_concept_ids}

        # Track concepts to expand at each hop
        frontier = set(seed_concept_ids)

        for hop in range(self.max_hops):
            if not frontier:
                break

            next_frontier: set[str] = set()
            hop_activations: dict[str, float] = {}

            for concept_id in frontier:
                current_act = activation.get(concept_id, 0)

                # Skip if below firing threshold
                if current_act < self.threshold:
                    continue

                # Get neighbors with edge info
                neighbors = await self.db.get_concept_neighbors(concept_id)

                for neighbor_concept, relation in neighbors:
                    neighbor_id = neighbor_concept.id

                    # Skip if already a seed (don't propagate back to seeds)
                    if neighbor_id in seed_concept_ids:
                        continue

                    # Query-oriented edge filtering
                    edge_relevance = 1.0
                    if query_embedding and relation.edge_embedding:
                        # Skip cosine similarity if dimensions don't match
                        # (can happen after embedding model change)
                        if len(query_embedding) == len(relation.edge_embedding):
                            edge_relevance = cosine_similarity(
                                query_embedding, relation.edge_embedding
                            )
                            # Filter out low-relevance edges
                            if edge_relevance < self.threshold:
                                continue

                    # v4.4: Apply semantic edge boost for universal semantic edges
                    semantic_boost = 1.0
                    if getattr(relation, 'is_universal', False) and getattr(relation, 'is_semantic', False):
                        semantic_boost = self.semantic_edge_boost

                    # Calculate activation transfer
                    transfer = (
                        current_act
                        * relation.weight
                        * edge_relevance
                        * self.decay
                        * self.rescale
                        * semantic_boost
                    )

                    # Accumulate activation (can receive from multiple sources)
                    hop_activations[neighbor_id] = (
                        hop_activations.get(neighbor_id, 0) + transfer
                    )

                    # Track path (use path with highest activation)
                    if neighbor_id not in paths or hop_activations[neighbor_id] > activation.get(neighbor_id, 0):
                        paths[neighbor_id] = paths[concept_id] + [neighbor_id]
                        hop_levels[neighbor_id] = hop + 1

            # Lateral inhibition: keep top-k per hop
            sorted_hop = sorted(
                hop_activations.items(), key=lambda x: x[1], reverse=True
            )
            top_k = dict(sorted_hop[: self.top_k_per_hop])

            # Merge with existing activations (keep max)
            for cid, act in top_k.items():
                if act > activation.get(cid, 0):
                    activation[cid] = act
                    next_frontier.add(cid)

            frontier = next_frontier

            logger.debug(
                f"Hop {hop + 1}: {len(top_k)} concepts activated, "
                f"top activation: {max(top_k.values()) if top_k else 0:.3f}"
            )

        # Update activation counts in database
        for concept_id in activation:
            if concept_id not in seed_concept_ids:
                await self.db.update_concept_activation(concept_id)

        # Build result with concept objects
        activated_concepts: list[ActivatedConcept] = []
        for cid, act in activation.items():
            concept = await self.db.get_concept(cid)
            if concept:
                activated_concepts.append(
                    ActivatedConcept(
                        concept=concept,
                        activation=act,
                        hop=hop_levels.get(cid, 0),
                        path=paths.get(cid, [cid]),
                    )
                )

        # Sort by activation level (descending)
        activated_concepts.sort(key=lambda x: x.activation, reverse=True)

        return ActivationResult(
            activations=activation,
            activated_concepts=activated_concepts,
            seed_concepts=seed_concept_ids,
            hops_performed=min(hop + 1, self.max_hops) if frontier or hop_activations else hop,
        )

    async def activate_from_query(
        self,
        query: str,
        query_embedding: list[float],
        top_k_seeds: int = 5,
    ) -> ActivationResult:
        """
        Activate concepts starting from query-relevant seeds.

        First finds concepts similar to the query via vector search,
        then spreads activation from those seeds.
        """
        # Find seed concepts via vector similarity
        seed_results = await self.db.vector_search_concepts(
            embedding=query_embedding, k=top_k_seeds
        )

        seed_ids = [concept.id for concept, score in seed_results]

        logger.info(
            f"Query '{query[:50]}...' activated {len(seed_ids)} seed concepts"
        )

        return await self.activate(seed_ids, query_embedding)


async def spread_activation(
    db: Neo4jClient,
    seed_concepts: list[str],
    query_embedding: list[float] | None = None,
    decay: float = 0.85,
    threshold: float = 0.3,
    max_hops: int = 3,
    rescale: float = 0.4,
) -> dict[str, float]:
    """
    Convenience function for spreading activation.

    Returns concept_id -> activation mapping.
    """
    spreader = SpreadingActivation(
        db=db,
        decay=decay,
        threshold=threshold,
        max_hops=max_hops,
        rescale=rescale,
    )
    result = await spreader.activate(seed_concepts, query_embedding)
    return result.activations
