#!/usr/bin/env python3
"""Interactive CLI for reviewing medium-confidence duplicates.

Usage:
    uv run python scripts/review_duplicates.py
    uv run python scripts/review_duplicates.py --auto-skip-low  # Skip low similarity pairs
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# Add src to path
sys.path.insert(0, "src")

from engram.graph.deduplication import ConceptDeduplicator
from engram.graph.models import MatchConfidence
from engram.storage.neo4j_client import Neo4jClient


async def review_duplicates(
    db: Neo4jClient,
    auto_skip_low: bool = False,
) -> None:
    """Interactive review of duplicate candidates."""
    print("\n=== Duplicate Review CLI ===\n")

    deduplicator = ConceptDeduplicator(db)

    # Find duplicates
    print("Finding duplicate candidates...")
    duplicates = await deduplicator.find_duplicates()

    # Filter to medium/low confidence (high are auto-merged)
    review_candidates = [
        d for d in duplicates
        if d.confidence in (MatchConfidence.MEDIUM, MatchConfidence.LOW)
    ]

    if auto_skip_low:
        review_candidates = [d for d in review_candidates if d.confidence == MatchConfidence.MEDIUM]

    if not review_candidates:
        print("No duplicates to review!")
        return

    print(f"Found {len(review_candidates)} candidates to review\n")
    print("Commands: [m]erge, [s]kip, [n]ot duplicate, [q]uit\n")

    merged_count = 0
    skipped_count = 0
    rejected_count = 0

    for i, candidate in enumerate(review_candidates):
        print(f"--- Candidate {i + 1}/{len(review_candidates)} ---")
        print(f"  Source: {candidate.source_name} ({candidate.source_id[:8]}...)")
        print(f"  Target: {candidate.target_name} ({candidate.target_id[:8]}...)")
        print(f"  Combined similarity: {candidate.combined_similarity:.2%}")
        print(f"    LaBSE: {candidate.labse_similarity:.2%}")
        print(f"    Phonetic: {candidate.phonetic_similarity:.2%}")
        print(f"    String: {candidate.string_similarity:.2%}")
        print(f"  Confidence: {candidate.confidence.value}")
        print()

        while True:
            try:
                choice = input("Action [m/s/n/q]: ").strip().lower()
            except EOFError:
                choice = "q"

            if choice == "m":
                # Merge (use source as canonical)
                await deduplicator.merge_concepts(
                    canonical_id=candidate.source_id,
                    merge_ids=[candidate.target_id],
                )
                merged_count += 1
                print(f"  -> Merged '{candidate.target_name}' into '{candidate.source_name}'")
                break

            elif choice == "s":
                # Skip for now (keep POSSIBLE_DUPLICATE edge)
                skipped_count += 1
                print("  -> Skipped")
                break

            elif choice == "n":
                # Not a duplicate - remove POSSIBLE_DUPLICATE edge
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $source_id})-[r:POSSIBLE_DUPLICATE]-(b:Concept {id: $target_id})
                    DELETE r
                    """,
                    source_id=candidate.source_id,
                    target_id=candidate.target_id,
                )
                rejected_count += 1
                print("  -> Marked as NOT duplicate")
                break

            elif choice == "q":
                print("\nQuitting review...")
                print(f"\nSummary:")
                print(f"  Merged: {merged_count}")
                print(f"  Skipped: {skipped_count}")
                print(f"  Rejected: {rejected_count}")
                return

            else:
                print("  Invalid choice. Use m/s/n/q")

        print()

    print("\n=== Review Complete ===")
    print(f"  Merged: {merged_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Rejected: {rejected_count}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive review of duplicate concepts"
    )
    parser.add_argument(
        "--auto-skip-low",
        action="store_true",
        help="Skip low-confidence candidates (only review medium)",
    )

    args = parser.parse_args()

    db = Neo4jClient()
    await db.connect()

    try:
        await review_duplicates(db, args.auto_skip_low)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
