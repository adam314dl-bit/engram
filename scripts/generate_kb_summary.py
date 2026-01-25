#!/usr/bin/env python
"""Generate KB summary for query enrichment.

Run this script after ingestion to generate a summary of the knowledge base
for use in query enrichment.

Usage:
    # Default (uses KB_SUMMARY_USE_LLM from config)
    uv run python scripts/generate_kb_summary.py

    # Force LLM enhancement
    uv run python scripts/generate_kb_summary.py --llm

    # Force no LLM (Neo4j only)
    uv run python scripts/generate_kb_summary.py --no-llm
"""

import argparse
import asyncio
import logging
import sys

# Add src to path
sys.path.insert(0, "src")

from engram.config import settings
from engram.indexing.kb_summary import (
    KBSummaryGenerator,
    KBSummaryStore,
    LLMKBSummaryEnhancer,
)
from engram.storage.neo4j_client import Neo4jClient, close_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate KB summary for query enrichment"
    )
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Force LLM enhancement (override config)",
    )
    llm_group.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM enhancement (Neo4j only)",
    )
    return parser.parse_args()


async def main() -> None:
    """Generate and save KB summary."""
    args = parse_args()

    # Determine LLM usage
    if args.llm:
        use_llm = True
        logger.info("LLM enhancement: ENABLED (--llm flag)")
    elif args.no_llm:
        use_llm = False
        logger.info("LLM enhancement: DISABLED (--no-llm flag)")
    else:
        use_llm = settings.kb_summary_use_llm
        logger.info(f"LLM enhancement: {'ENABLED' if use_llm else 'DISABLED'} (from config)")

    logger.info("Starting KB summary generation...")

    db = Neo4jClient()
    await db.connect()

    try:
        # Generate base summary from Neo4j
        generator = KBSummaryGenerator(db)
        summary = await generator.generate()

        # Optionally enhance with LLM
        if use_llm:
            logger.info("Enhancing summary with LLM...")
            enhancer = LLMKBSummaryEnhancer(db)
            summary = await enhancer.enhance(summary)

        # Display summary
        print("\n" + "=" * 60)
        print("KB SUMMARY")
        print("=" * 60)

        # v4.4 LLM-enhanced fields (if present)
        if summary.domain_description:
            print(f"\n🌐 Domain Description:")
            print(f"   {summary.domain_description}")

        if summary.capabilities:
            print(f"\n✅ Capabilities ({len(summary.capabilities)}):")
            for cap in summary.capabilities:
                print(f"   - {cap}")

        if summary.limitations:
            print(f"\n❌ Limitations ({len(summary.limitations)}):")
            for lim in summary.limitations:
                print(f"   - {lim}")

        if summary.sample_questions:
            print(f"\n❓ Sample Questions ({len(summary.sample_questions)}):")
            for q in summary.sample_questions:
                print(f"   - {q}")

        # Original fields
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
        print(summary.to_prompt_text(max_tokens=500))
        print("=" * 60)

        # Save to Neo4j
        store = KBSummaryStore(db)
        await store.save(summary)

        enhanced_marker = " (LLM-enhanced)" if use_llm and summary.domain_description else ""
        print(f"\n✓ KB summary saved to Neo4j{enhanced_marker}")

    finally:
        await db.close()
        await close_client()


if __name__ == "__main__":
    asyncio.run(main())
