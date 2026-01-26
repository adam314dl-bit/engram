#!/usr/bin/env python3
"""CLI for graph quality optimization.

Usage:
    uv run python scripts/improve_graph_quality.py dedup       # Run deduplication
    uv run python scripts/improve_graph_quality.py enrich      # Run enrichment
    uv run python scripts/improve_graph_quality.py all         # Run both
    uv run python scripts/improve_graph_quality.py stats       # Show statistics
    uv run python scripts/improve_graph_quality.py --dry-run   # Preview changes
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

# Add src to path
sys.path.insert(0, "src")

from engram.graph.config import DeduplicationConfig, EnrichmentConfig, GraphQualityConfig
from engram.graph.deduplication import ConceptDeduplicator, DeduplicationSafetyWrapper
from engram.graph.enrichment import EdgeClassifier, SemanticEnricher
from engram.graph.metrics import GraphQualityMetrics
from engram.storage.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_deduplication(
    db: Neo4jClient,
    config: GraphQualityConfig,
    dry_run: bool = False,
) -> None:
    """Run concept deduplication."""
    print("\n=== Running Concept Deduplication ===\n")

    dedup_config = DeduplicationConfig(
        auto_merge_threshold=config.deduplication.auto_merge_threshold,
        review_threshold=config.deduplication.review_threshold,
        possible_threshold=config.deduplication.possible_threshold,
    )

    deduplicator = ConceptDeduplicator(db, dedup_config)

    if config.backup_before_merge and not dry_run:
        wrapper = DeduplicationSafetyWrapper(db)
        backup_id = await wrapper.create_backup()
        print(f"Created backup: {backup_id}")

    report = await deduplicator.run_deduplication(
        dry_run=dry_run,
        auto_merge=not dry_run,
    )

    print(f"\nDeduplication {'Preview' if dry_run else 'Report'}:")
    print(f"  Total concepts: {report.total_concepts}")
    print(f"  Duplicates found: {report.duplicates_found}")
    print(f"    High confidence (auto-merge): {report.high_confidence}")
    print(f"    Medium confidence (review): {report.medium_confidence}")
    print(f"    Low confidence (tracked): {report.low_confidence}")
    print(f"  Merges performed: {report.merges_performed}")

    if report.errors:
        print(f"\n  Errors ({len(report.errors)}):")
        for error in report.errors[:5]:
            print(f"    - {error}")
        if len(report.errors) > 5:
            print(f"    ... and {len(report.errors) - 5} more")


async def run_enrichment(
    db: Neo4jClient,
    config: GraphQualityConfig,
    dry_run: bool = False,
    limit: int | None = None,
    min_degree: int = 0,
) -> None:
    """Run semantic enrichment."""
    print("\n=== Running Semantic Enrichment ===\n")

    if dry_run:
        # Count what would be enriched
        result = await db.execute_query(
            "MATCH (c:Concept) WHERE c.enriched_at IS NULL AND (c.status IS NULL OR c.status = 'active') RETURN count(c) as count"
        )
        count = result[0]["count"] if result else 0
        print(f"Would enrich {count} concepts")
        return

    enrich_config = EnrichmentConfig(
        semantic_edge_boost=config.enrichment.semantic_edge_boost,
    )

    enricher = SemanticEnricher(db, enrich_config)
    result = await enricher.enrich_all_concepts(limit=limit, min_degree=min_degree)

    print(f"\nEnrichment Report:")
    print(f"  Concepts enriched: {result.concepts_enriched}")
    print(f"  Definitions generated: {result.definitions_generated}")
    print(f"  World knowledge edges: {result.world_knowledge_edges}")

    # Classify edges
    print("\n  Classifying existing edges...")
    classifier = EdgeClassifier(db)
    classified = await classifier.classify_all_edges()
    print(f"  Edges classified: {result.edges_classified + classified}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"    - {error}")


async def run_classify_edges(db: Neo4jClient) -> None:
    """Run edge classification only."""
    print("\n=== Running Edge Classification ===\n")

    classifier = EdgeClassifier(db)
    classified = await classifier.classify_all_edges()
    print(f"\nEdges classified: {classified}")


async def show_stats(db: Neo4jClient) -> None:
    """Show graph quality statistics."""
    print("\n=== Graph Quality Statistics ===\n")

    metrics = GraphQualityMetrics(db)
    report = await metrics.compute_all()

    print(metrics.format_report(report))


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Graph quality optimization for Engram v4.4"
    )
    parser.add_argument(
        "command",
        choices=["dedup", "enrich", "classify", "all", "stats"],
        help="Command to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup before destructive operations",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of concepts to enrich (for testing)",
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=0,
        help="Only enrich concepts with at least N edges",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("engram").setLevel(logging.DEBUG)

    config = GraphQualityConfig(
        dry_run=args.dry_run,
        backup_before_merge=not args.no_backup,
        verbose=args.verbose,
    )

    db = Neo4jClient()
    await db.connect()

    try:
        if args.command == "dedup":
            await run_deduplication(db, config, args.dry_run)

        elif args.command == "enrich":
            await run_enrichment(db, config, args.dry_run, args.limit, args.min_degree)

        elif args.command == "classify":
            await run_classify_edges(db)

        elif args.command == "all":
            await run_deduplication(db, config, args.dry_run)
            await run_enrichment(db, config, args.dry_run, args.limit, args.min_degree)

        elif args.command == "stats":
            await show_stats(db)

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
