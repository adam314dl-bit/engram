#!/usr/bin/env python3
"""Ingest mock documents into Engram."""

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_ingestion():
    """Ingest all mock documents."""
    from engram.ingestion.pipeline import IngestionPipeline
    from engram.storage.neo4j_client import Neo4jClient

    docs_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_docs"

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        logger.info("Run 'python scripts/generate_mock_docs.py' first to create documents.")
        return False

    # Count documents
    md_files = list(docs_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} documents to ingest")

    # Connect to Neo4j
    db = Neo4jClient()
    await db.connect()

    try:
        # Setup schema if needed
        await db.setup_schema()

        # Create pipeline
        pipeline = IngestionPipeline(neo4j_client=db)

        # Ingest directory
        logger.info(f"Ingesting documents from {docs_dir}...")
        results = await pipeline.ingest_directory(docs_dir, extensions=[".md"])

        # Summary
        total_concepts = sum(r.concepts_created for r in results)
        total_memories = sum(r.memories_created for r in results)
        total_relations = sum(r.relations_created for r in results)
        failed = [r for r in results if r.errors]

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Documents processed: {len(results)}")
        print(f"Concepts created:    {total_concepts}")
        print(f"Memories created:    {total_memories}")
        print(f"Relations created:   {total_relations}")

        if failed:
            print(f"\nFailed documents: {len(failed)}")
            for r in failed:
                print(f"  - {r.document.title}: {r.errors}")

        print("=" * 60)

        return len(failed) == 0

    finally:
        await db.close()


async def main():
    """Main entry point."""
    try:
        success = await run_ingestion()
        return success
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
