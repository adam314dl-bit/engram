"""Graph quality metrics for monitoring and evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class StructuralMetrics:
    """Structural metrics for the graph."""

    total_concepts: int = 0
    active_concepts: int = 0
    merged_concepts: int = 0
    alias_concepts: int = 0

    total_edges: int = 0
    orphaned_concepts: int = 0  # Concepts with no edges
    avg_degree: float = 0.0
    max_degree: int = 0


@dataclass
class EdgeQualityMetrics:
    """Quality metrics for edges."""

    semantic_edges: int = 0
    contextual_edges: int = 0
    universal_edges: int = 0

    semantic_ratio: float = 0.0  # semantic / total

    edges_by_source: dict[str, int] = field(default_factory=dict)
    # document, world_knowledge, ontology, inference, user

    edges_by_type: dict[str, int] = field(default_factory=dict)
    # is_a, contains, uses, needs, related_to, etc.


@dataclass
class EnrichmentMetrics:
    """Metrics for enrichment coverage."""

    concepts_with_definition: int = 0
    concepts_without_definition: int = 0
    enrichment_ratio: float = 0.0

    concepts_with_aliases: int = 0
    total_aliases: int = 0


@dataclass
class DeduplicationMetrics:
    """Metrics for deduplication status."""

    possible_duplicates: int = 0  # POSSIBLE_DUPLICATE edges
    pending_review: int = 0  # Concepts with pending_review status
    merged_count: int = 0


@dataclass
class GraphQualityReport:
    """Complete graph quality report."""

    structural: StructuralMetrics
    edge_quality: EdgeQualityMetrics
    enrichment: EnrichmentMetrics
    deduplication: DeduplicationMetrics


class GraphQualityMetrics:
    """Computes quality metrics for the knowledge graph."""

    def __init__(self, db: "Neo4jClient") -> None:
        self.db = db

    async def compute_all(self) -> GraphQualityReport:
        """Compute all quality metrics.

        Returns:
            Complete GraphQualityReport
        """
        structural = await self._compute_structural_metrics()
        edge_quality = await self._compute_edge_quality_metrics()
        enrichment = await self._compute_enrichment_metrics()
        deduplication = await self._compute_deduplication_metrics()

        return GraphQualityReport(
            structural=structural,
            edge_quality=edge_quality,
            enrichment=enrichment,
            deduplication=deduplication,
        )

    async def _compute_structural_metrics(self) -> StructuralMetrics:
        """Compute structural metrics."""
        metrics = StructuralMetrics()

        # Count concepts by status
        query = """
        MATCH (c:Concept)
        RETURN
            count(c) as total,
            sum(CASE WHEN c.status IS NULL OR c.status = 'active' THEN 1 ELSE 0 END) as active,
            sum(CASE WHEN c.status = 'merged' THEN 1 ELSE 0 END) as merged,
            sum(CASE WHEN c.status = 'alias' THEN 1 ELSE 0 END) as alias
        """
        result = await self.db.execute_query(query)
        if result:
            metrics.total_concepts = result[0]["total"]
            metrics.active_concepts = result[0]["active"]
            metrics.merged_concepts = result[0]["merged"]
            metrics.alias_concepts = result[0]["alias"]

        # Count edges
        edge_query = "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count"
        edge_result = await self.db.execute_query(edge_query)
        if edge_result:
            metrics.total_edges = edge_result[0]["count"]

        # Count orphaned concepts
        orphan_query = """
        MATCH (c:Concept)
        WHERE (c.status IS NULL OR c.status = 'active')
          AND NOT EXISTS { (c)-[:RELATED_TO]-() }
          AND NOT EXISTS { ()-[:RELATED_TO]->(c) }
        RETURN count(c) as count
        """
        orphan_result = await self.db.execute_query(orphan_query)
        if orphan_result:
            metrics.orphaned_concepts = orphan_result[0]["count"]

        # Compute degree statistics
        degree_query = """
        MATCH (c:Concept)
        WHERE c.status IS NULL OR c.status = 'active'
        WITH c, COUNT { (c)-[:RELATED_TO]-() } + COUNT { ()-[:RELATED_TO]->(c) } as degree
        RETURN avg(degree) as avg_degree, max(degree) as max_degree
        """
        degree_result = await self.db.execute_query(degree_query)
        if degree_result:
            metrics.avg_degree = float(degree_result[0]["avg_degree"] or 0)
            metrics.max_degree = int(degree_result[0]["max_degree"] or 0)

        return metrics

    async def _compute_edge_quality_metrics(self) -> EdgeQualityMetrics:
        """Compute edge quality metrics."""
        metrics = EdgeQualityMetrics()

        # Count semantic vs contextual
        semantic_query = """
        MATCH ()-[r:RELATED_TO]->()
        RETURN
            sum(CASE WHEN r.is_semantic = true THEN 1 ELSE 0 END) as semantic,
            sum(CASE WHEN r.is_semantic IS NULL OR r.is_semantic = false THEN 1 ELSE 0 END) as contextual,
            sum(CASE WHEN r.is_universal = true THEN 1 ELSE 0 END) as universal,
            count(r) as total
        """
        result = await self.db.execute_query(semantic_query)
        if result:
            metrics.semantic_edges = result[0]["semantic"]
            metrics.contextual_edges = result[0]["contextual"]
            metrics.universal_edges = result[0]["universal"]
            total = result[0]["total"]
            if total > 0:
                metrics.semantic_ratio = metrics.semantic_edges / total

        # Count by source type
        source_query = """
        MATCH ()-[r:RELATED_TO]->()
        RETURN coalesce(r.source_type, 'document') as source_type, count(r) as count
        """
        source_result = await self.db.execute_query(source_query)
        for row in source_result:
            metrics.edges_by_source[row["source_type"]] = row["count"]

        # Count by relation type
        type_query = """
        MATCH ()-[r:RELATED_TO]->()
        RETURN coalesce(r.type, 'related_to') as rel_type, count(r) as count
        ORDER BY count DESC
        """
        type_result = await self.db.execute_query(type_query)
        for row in type_result:
            metrics.edges_by_type[row["rel_type"]] = row["count"]

        return metrics

    async def _compute_enrichment_metrics(self) -> EnrichmentMetrics:
        """Compute enrichment metrics."""
        metrics = EnrichmentMetrics()

        # Count enriched concepts
        enrichment_query = """
        MATCH (c:Concept)
        WHERE c.status IS NULL OR c.status = 'active'
        RETURN
            sum(CASE WHEN c.enriched_definition IS NOT NULL THEN 1 ELSE 0 END) as with_def,
            sum(CASE WHEN c.enriched_definition IS NULL THEN 1 ELSE 0 END) as without_def,
            count(c) as total
        """
        result = await self.db.execute_query(enrichment_query)
        if result:
            metrics.concepts_with_definition = result[0]["with_def"]
            metrics.concepts_without_definition = result[0]["without_def"]
            total = result[0]["total"]
            if total > 0:
                metrics.enrichment_ratio = metrics.concepts_with_definition / total

        # Count aliases
        alias_query = """
        MATCH (c:Concept)
        WHERE c.status IS NULL OR c.status = 'active'
        WITH c, size(coalesce(c.aliases, [])) as alias_count
        RETURN
            sum(CASE WHEN alias_count > 0 THEN 1 ELSE 0 END) as with_aliases,
            sum(alias_count) as total_aliases
        """
        alias_result = await self.db.execute_query(alias_query)
        if alias_result:
            metrics.concepts_with_aliases = alias_result[0]["with_aliases"]
            metrics.total_aliases = alias_result[0]["total_aliases"]

        return metrics

    async def _compute_deduplication_metrics(self) -> DeduplicationMetrics:
        """Compute deduplication metrics."""
        metrics = DeduplicationMetrics()

        # Count possible duplicates
        dup_query = """
        MATCH ()-[r:POSSIBLE_DUPLICATE]->()
        RETURN count(r) as count
        """
        dup_result = await self.db.execute_query(dup_query)
        if dup_result:
            metrics.possible_duplicates = dup_result[0]["count"]

        # Count pending review
        pending_query = """
        MATCH (c:Concept)
        WHERE c.status = 'pending_review'
        RETURN count(c) as count
        """
        pending_result = await self.db.execute_query(pending_query)
        if pending_result:
            metrics.pending_review = pending_result[0]["count"]

        # Count merged
        merged_query = """
        MATCH (c:Concept)
        WHERE c.status = 'merged'
        RETURN count(c) as count
        """
        merged_result = await self.db.execute_query(merged_query)
        if merged_result:
            metrics.merged_count = merged_result[0]["count"]

        return metrics

    def format_report(self, report: GraphQualityReport) -> str:
        """Format report as human-readable string."""
        lines = [
            "=== Graph Quality Report ===",
            "",
            "Structural Metrics:",
            f"  Total concepts: {report.structural.total_concepts}",
            f"  Active concepts: {report.structural.active_concepts}",
            f"  Merged concepts: {report.structural.merged_concepts}",
            f"  Alias concepts: {report.structural.alias_concepts}",
            f"  Total edges: {report.structural.total_edges}",
            f"  Orphaned concepts: {report.structural.orphaned_concepts}",
            f"  Average degree: {report.structural.avg_degree:.2f}",
            f"  Max degree: {report.structural.max_degree}",
            "",
            "Edge Quality Metrics:",
            f"  Semantic edges: {report.edge_quality.semantic_edges}",
            f"  Contextual edges: {report.edge_quality.contextual_edges}",
            f"  Universal edges: {report.edge_quality.universal_edges}",
            f"  Semantic ratio: {report.edge_quality.semantic_ratio:.1%}",
            "",
            "  Edges by source:",
        ]

        for source, count in sorted(report.edge_quality.edges_by_source.items()):
            lines.append(f"    {source}: {count}")

        lines.extend([
            "",
            "  Top edge types:",
        ])

        for rel_type, count in list(report.edge_quality.edges_by_type.items())[:10]:
            lines.append(f"    {rel_type}: {count}")

        lines.extend([
            "",
            "Enrichment Metrics:",
            f"  Concepts with definition: {report.enrichment.concepts_with_definition}",
            f"  Concepts without definition: {report.enrichment.concepts_without_definition}",
            f"  Enrichment ratio: {report.enrichment.enrichment_ratio:.1%}",
            f"  Concepts with aliases: {report.enrichment.concepts_with_aliases}",
            f"  Total aliases: {report.enrichment.total_aliases}",
            "",
            "Deduplication Metrics:",
            f"  Possible duplicates: {report.deduplication.possible_duplicates}",
            f"  Pending review: {report.deduplication.pending_review}",
            f"  Merged count: {report.deduplication.merged_count}",
        ])

        return "\n".join(lines)
