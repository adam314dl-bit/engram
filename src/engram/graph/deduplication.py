"""Concept deduplication with cross-lingual duplicate detection.

Uses LaBSE for semantic similarity, phonetic matching via transliteration,
and Jaro-Winkler string similarity for robust duplicate detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import jellyfish

from engram.graph.config import DeduplicationConfig
from engram.graph.models import DuplicateCandidate, MatchConfidence
from engram.models import Concept
from engram.preprocessing.transliteration import to_latin

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of a concept merge operation."""

    canonical_id: str
    merged_ids: list[str]
    aliases_added: list[str]
    edges_redirected: int
    memories_redirected: int


@dataclass
class DeduplicationReport:
    """Report from a deduplication run."""

    total_concepts: int
    duplicates_found: int
    high_confidence: int  # Auto-merged
    medium_confidence: int  # POSSIBLE_DUPLICATE edges created
    low_confidence: int  # Tracked but not acted on
    merges_performed: int
    errors: list[str]


class ConceptDeduplicator:
    """Detects and merges duplicate concepts across languages."""

    def __init__(
        self,
        db: "Neo4jClient",
        config: DeduplicationConfig | None = None,
    ) -> None:
        self.db = db
        self.config = config or DeduplicationConfig()
        self._labse_model: "SentenceTransformer | None" = None

    def load_labse_model(self) -> "SentenceTransformer":
        """Load LaBSE model for cross-lingual embeddings."""
        if self._labse_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading LaBSE model: {self.config.labse_model}")
            self._labse_model = SentenceTransformer(self.config.labse_model)
            logger.info("LaBSE model loaded")
        return self._labse_model

    async def compute_all_embeddings(
        self,
        concepts: list[Concept],
    ) -> dict[str, list[float]]:
        """Compute LaBSE embeddings for all concepts.

        Args:
            concepts: Concepts to embed

        Returns:
            Mapping of concept_id -> LaBSE embedding
        """
        model = self.load_labse_model()
        texts = [c.name for c in concepts]

        logger.info(f"Computing LaBSE embeddings for {len(texts)} concepts")
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return {
            concept.id: embedding.tolist()
            for concept, embedding in zip(concepts, embeddings)
        }

    def compute_phonetic_similarity(self, name1: str, name2: str) -> float:
        """Compute phonetic similarity using transliteration.

        Converts both names to Latin and compares using Jaro-Winkler.

        Args:
            name1: First concept name
            name2: Second concept name

        Returns:
            Similarity score (0.0 - 1.0)
        """
        latin1 = to_latin(name1.lower())
        latin2 = to_latin(name2.lower())
        return jellyfish.jaro_winkler_similarity(latin1, latin2)

    def compute_string_similarity(self, name1: str, name2: str) -> float:
        """Compute Jaro-Winkler string similarity.

        Args:
            name1: First concept name
            name2: Second concept name

        Returns:
            Similarity score (0.0 - 1.0)
        """
        return jellyfish.jaro_winkler_similarity(name1.lower(), name2.lower())

    def _compute_combined_similarity(
        self,
        labse_sim: float,
        phonetic_sim: float,
        string_sim: float,
    ) -> float:
        """Compute weighted combined similarity score."""
        return (
            self.config.labse_weight * labse_sim
            + self.config.phonetic_weight * phonetic_sim
            + self.config.string_weight * string_sim
        )

    def _classify_confidence(self, combined_sim: float) -> MatchConfidence:
        """Classify match confidence based on combined similarity."""
        if combined_sim >= self.config.auto_merge_threshold:
            return MatchConfidence.HIGH
        elif combined_sim >= self.config.review_threshold:
            return MatchConfidence.MEDIUM
        elif combined_sim >= self.config.possible_threshold:
            return MatchConfidence.LOW
        return MatchConfidence.NONE

    async def find_duplicates(
        self,
        concepts: list[Concept] | None = None,
    ) -> list[DuplicateCandidate]:
        """Find duplicate concepts in the graph.

        Args:
            concepts: Concepts to check (loads all if None)

        Returns:
            List of duplicate candidates sorted by similarity
        """
        if concepts is None:
            # Load all active concepts
            results = await self.db.execute_query(
                "MATCH (c:Concept) WHERE c.status IS NULL OR c.status = 'active' RETURN c"
            )
            concepts = [Concept.from_dict(dict(r["c"])) for r in results]

        if len(concepts) < 2:
            return []

        logger.info(f"Finding duplicates among {len(concepts)} concepts")

        # Compute LaBSE embeddings
        embeddings = await self.compute_all_embeddings(concepts)

        # Store embeddings in concepts for later use
        for concept in concepts:
            concept.labse_embedding = embeddings.get(concept.id)

        # Find duplicates using cosine similarity
        import numpy as np

        # Convert to numpy array for efficient computation
        concept_ids = [c.id for c in concepts]
        concept_names = {c.id: c.name for c in concepts}
        embedding_matrix = np.array([embeddings[cid] for cid in concept_ids])

        # Normalize for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized = embedding_matrix / (norms + 1e-10)

        # Compute similarity matrix
        similarity_matrix = normalized @ normalized.T

        duplicates: list[DuplicateCandidate] = []

        # Find pairs above threshold (only upper triangle to avoid duplicates)
        for i in range(len(concepts)):
            candidates_for_i: list[tuple[int, float]] = []

            for j in range(i + 1, len(concepts)):
                labse_sim = float(similarity_matrix[i, j])

                # Quick filter: skip if LaBSE similarity is too low
                if labse_sim < self.config.possible_threshold * 0.5:
                    continue

                candidates_for_i.append((j, labse_sim))

            # Sort by LaBSE similarity and keep top candidates
            candidates_for_i.sort(key=lambda x: x[1], reverse=True)
            candidates_for_i = candidates_for_i[: self.config.max_candidates_per_concept]

            for j, labse_sim in candidates_for_i:
                name_i = concept_names[concept_ids[i]]
                name_j = concept_names[concept_ids[j]]

                phonetic_sim = self.compute_phonetic_similarity(name_i, name_j)
                string_sim = self.compute_string_similarity(name_i, name_j)
                combined_sim = self._compute_combined_similarity(
                    labse_sim, phonetic_sim, string_sim
                )

                confidence = self._classify_confidence(combined_sim)
                if confidence == MatchConfidence.NONE:
                    continue

                duplicates.append(
                    DuplicateCandidate(
                        source_id=concept_ids[i],
                        source_name=name_i,
                        target_id=concept_ids[j],
                        target_name=name_j,
                        labse_similarity=labse_sim,
                        phonetic_similarity=phonetic_sim,
                        string_similarity=string_sim,
                        combined_similarity=combined_sim,
                        confidence=confidence,
                    )
                )

        # Sort by combined similarity (descending)
        duplicates.sort(key=lambda d: d.combined_similarity, reverse=True)

        logger.info(
            f"Found {len(duplicates)} duplicate candidates: "
            f"{sum(1 for d in duplicates if d.confidence == MatchConfidence.HIGH)} high, "
            f"{sum(1 for d in duplicates if d.confidence == MatchConfidence.MEDIUM)} medium, "
            f"{sum(1 for d in duplicates if d.confidence == MatchConfidence.LOW)} low"
        )

        return duplicates

    async def merge_concepts(
        self,
        canonical_id: str,
        merge_ids: list[str],
        dry_run: bool = False,
    ) -> MergeResult:
        """Merge duplicate concepts into a canonical concept.

        Args:
            canonical_id: ID of the canonical concept to keep
            merge_ids: IDs of concepts to merge into canonical
            dry_run: If True, only preview changes

        Returns:
            MergeResult with details of the merge
        """
        if not merge_ids:
            return MergeResult(
                canonical_id=canonical_id,
                merged_ids=[],
                aliases_added=[],
                edges_redirected=0,
                memories_redirected=0,
            )

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Merging {len(merge_ids)} concepts into {canonical_id}"
        )

        # Get canonical concept
        canonical = await self.db.get_concept(canonical_id)
        if not canonical:
            raise ValueError(f"Canonical concept not found: {canonical_id}")

        # Get concepts to merge
        aliases_added: list[str] = []
        for merge_id in merge_ids:
            concept = await self.db.get_concept(merge_id)
            if concept:
                aliases_added.append(concept.name)

        if dry_run:
            # Count what would be affected
            edges_query = """
            MATCH (c:Concept)-[r:RELATED_TO]-()
            WHERE c.id IN $merge_ids
            RETURN count(r) as count
            """
            memories_query = """
            MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
            WHERE c.id IN $merge_ids
            RETURN count(s) as count
            """
            edges_result = await self.db.execute_query(edges_query, merge_ids=merge_ids)
            memories_result = await self.db.execute_query(memories_query, merge_ids=merge_ids)

            return MergeResult(
                canonical_id=canonical_id,
                merged_ids=merge_ids,
                aliases_added=aliases_added,
                edges_redirected=edges_result[0]["count"] if edges_result else 0,
                memories_redirected=memories_result[0]["count"] if memories_result else 0,
            )

        # Perform actual merge
        result = await self.db.merge_concepts(canonical_id, merge_ids, aliases_added)

        logger.info(
            f"Merged {len(merge_ids)} concepts: "
            f"{result['edges_redirected']} edges, {result['memories_redirected']} memories redirected"
        )

        return MergeResult(
            canonical_id=canonical_id,
            merged_ids=merge_ids,
            aliases_added=aliases_added,
            edges_redirected=result["edges_redirected"],
            memories_redirected=result["memories_redirected"],
        )

    async def create_possible_duplicate_edge(
        self,
        candidate: DuplicateCandidate,
    ) -> None:
        """Create POSSIBLE_DUPLICATE edge for medium-confidence matches.

        Args:
            candidate: Duplicate candidate to track
        """
        query = """
        MATCH (a:Concept {id: $source_id})
        MATCH (b:Concept {id: $target_id})
        MERGE (a)-[r:POSSIBLE_DUPLICATE]->(b)
        SET r.combined_similarity = $combined_similarity,
            r.labse_similarity = $labse_similarity,
            r.phonetic_similarity = $phonetic_similarity,
            r.string_similarity = $string_similarity,
            r.confidence = $confidence,
            r.detected_at = datetime()
        """
        await self.db.execute_query(
            query,
            source_id=candidate.source_id,
            target_id=candidate.target_id,
            combined_similarity=candidate.combined_similarity,
            labse_similarity=candidate.labse_similarity,
            phonetic_similarity=candidate.phonetic_similarity,
            string_similarity=candidate.string_similarity,
            confidence=candidate.confidence.value,
        )

    async def run_deduplication(
        self,
        dry_run: bool = False,
        auto_merge: bool = True,
    ) -> DeduplicationReport:
        """Run full deduplication process.

        Args:
            dry_run: If True, only report what would be done
            auto_merge: If True, automatically merge high-confidence duplicates

        Returns:
            DeduplicationReport with results
        """
        errors: list[str] = []
        merges_performed = 0

        # Find all duplicates
        duplicates = await self.find_duplicates()

        # Count by confidence
        high_confidence = [d for d in duplicates if d.confidence == MatchConfidence.HIGH]
        medium_confidence = [d for d in duplicates if d.confidence == MatchConfidence.MEDIUM]
        low_confidence = [d for d in duplicates if d.confidence == MatchConfidence.LOW]

        # Auto-merge high confidence if enabled
        if auto_merge and not dry_run:
            # Group by canonical (first seen ID)
            merge_groups: dict[str, list[str]] = {}
            processed: set[str] = set()

            for dup in high_confidence:
                # Skip if either concept already processed
                if dup.source_id in processed or dup.target_id in processed:
                    continue

                # Use first ID as canonical
                canonical = dup.source_id
                if canonical not in merge_groups:
                    merge_groups[canonical] = []
                merge_groups[canonical].append(dup.target_id)
                processed.add(dup.target_id)

            # Perform merges
            for canonical_id, merge_ids in merge_groups.items():
                try:
                    await self.merge_concepts(canonical_id, merge_ids)
                    merges_performed += len(merge_ids)
                except Exception as e:
                    errors.append(f"Failed to merge {merge_ids} into {canonical_id}: {e}")

        # Create POSSIBLE_DUPLICATE edges for medium confidence
        if not dry_run:
            for dup in medium_confidence:
                try:
                    await self.create_possible_duplicate_edge(dup)
                except Exception as e:
                    errors.append(f"Failed to create edge for {dup.source_id}-{dup.target_id}: {e}")

        # Get total concept count
        count_result = await self.db.execute_query(
            "MATCH (c:Concept) WHERE c.status IS NULL OR c.status = 'active' RETURN count(c) as count"
        )
        total_concepts = count_result[0]["count"] if count_result else 0

        return DeduplicationReport(
            total_concepts=total_concepts,
            duplicates_found=len(duplicates),
            high_confidence=len(high_confidence),
            medium_confidence=len(medium_confidence),
            low_confidence=len(low_confidence),
            merges_performed=merges_performed,
            errors=errors,
        )


class DeduplicationSafetyWrapper:
    """Safety wrapper for destructive deduplication operations."""

    def __init__(self, db: "Neo4jClient") -> None:
        self.db = db
        self._backup_id: str | None = None

    async def create_backup(self) -> str:
        """Create a backup before destructive operations.

        Returns:
            Backup identifier
        """
        backup_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._backup_id = backup_id

        # Export concepts and relations to a backup label
        backup_query = """
        // Backup concepts
        MATCH (c:Concept)
        WITH c, properties(c) as props
        CREATE (b:ConceptBackup {backup_id: $backup_id})
        SET b += props

        // Backup relations
        MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
        CREATE (rb:RelationBackup {
            backup_id: $backup_id,
            source_id: c1.id,
            target_id: c2.id,
            type: r.type,
            weight: r.weight
        })
        """
        await self.db.execute_query(backup_query, backup_id=backup_id)

        logger.info(f"Created backup: {backup_id}")
        return backup_id

    async def execute_merge_dry_run(
        self,
        deduplicator: ConceptDeduplicator,
    ) -> DeduplicationReport:
        """Execute a dry run to preview changes.

        Args:
            deduplicator: Deduplicator to use

        Returns:
            Report of what would be done
        """
        return await deduplicator.run_deduplication(dry_run=True, auto_merge=True)

    async def rollback_to_backup(self, backup_id: str) -> None:
        """Rollback to a previous backup.

        Args:
            backup_id: Backup identifier to restore
        """
        rollback_query = """
        // Delete current concepts
        MATCH (c:Concept) DETACH DELETE c

        // Restore from backup
        MATCH (b:ConceptBackup {backup_id: $backup_id})
        CREATE (c:Concept)
        SET c = properties(b)
        REMOVE c.backup_id

        // Restore relations
        MATCH (rb:RelationBackup {backup_id: $backup_id})
        MATCH (c1:Concept {id: rb.source_id})
        MATCH (c2:Concept {id: rb.target_id})
        CREATE (c1)-[r:RELATED_TO]->(c2)
        SET r.type = rb.type, r.weight = rb.weight

        // Clean up backup
        MATCH (b:ConceptBackup {backup_id: $backup_id}) DELETE b
        MATCH (rb:RelationBackup {backup_id: $backup_id}) DELETE rb
        """
        await self.db.execute_query(rollback_query, backup_id=backup_id)
        logger.info(f"Rolled back to backup: {backup_id}")
