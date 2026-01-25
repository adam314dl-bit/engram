#!/usr/bin/env python
"""Generate KB summary for query enrichment.

Run this script after ingestion to generate a summary of the knowledge base
for use in query enrichment.

Usage:
    uv run python scripts/generate_kb_summary.py
"""

import asyncio
import logging
import sys

# Add src to path
sys.path.insert(0, "src")

from engram.indexing.kb_summary import KBSummaryGenerator, KBSummaryStore
from engram.storage.neo4j_client import Neo4jClient, close_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Generate and save KB summary."""
    logger.info("Starting KB summary generation...")

    db = Neo4jClient()
    await db.connect()

    try:
        # Generate summary
        generator = KBSummaryGenerator(db)
        summary = await generator.generate()

        # Display summary
        print("\n" + "=" * 60)
        print("KB SUMMARY")
        print("=" * 60)
        print(f"\nDomains ({len(summary.domains)}):")
        for domain in summary.domains[:10]:
            print(f"  - {domain}")
        if len(summary.domains) > 10:
            print(f"  ... and {len(summary.domains) - 10} more")

        print(f"\nEntity Types ({len(summary.entity_types)}):")
        for etype, entities in summary.entity_types.items():
            sample = entities[:3]
            print(f"  - {etype}: {', '.join(sample)}" + (" ..." if len(entities) > 3 else ""))

        print(f"\nInfo Types ({len(summary.info_types)}):")
        for itype in summary.info_types[:10]:
            print(f"  - {itype}")

        print(f"\nKey Terms (top 20 of {len(summary.key_terms)}):")
        top_terms = sorted(summary.key_terms.items(), key=lambda x: x[1], reverse=True)[:20]
        for term, count in top_terms:
            print(f"  - {term}: {count}")

        print(f"\nStatistics:")
        for key, value in summary.statistics.items():
            print(f"  - {key}: {value}")

        print("\n" + "=" * 60)
        print("PROMPT TEXT PREVIEW")
        print("=" * 60)
        print(summary.to_prompt_text(max_tokens=400))
        print("=" * 60)

        # Save to Neo4j
        store = KBSummaryStore(db)
        await store.save(summary)

        print("\n✓ KB summary saved to Neo4j")

    finally:
        await db.close()
        await close_client()


if __name__ == "__main__":
    asyncio.run(main())
