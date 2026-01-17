#!/usr/bin/env python3
"""Seed test data into Engram for development."""

import asyncio
import logging
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engram.ingestion import IngestionPipeline
from engram.storage import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    # Initialize Neo4j client
    db = Neo4jClient()
    await db.connect()

    try:
        # Setup schema
        logger.info("Setting up Neo4j schema...")
        await db.setup_schema()

        # Create pipeline
        pipeline = IngestionPipeline(neo4j_client=db)

        # Find test documents
        docs_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_docs"

        if not docs_dir.exists():
            logger.error(f"Mock docs directory not found: {docs_dir}")
            return

        # Ingest documents
        logger.info(f"Ingesting documents from {docs_dir}")
        results = await pipeline.ingest_directory(docs_dir)

        # Summary
        total_concepts = sum(r.concepts_created for r in results)
        total_memories = sum(r.memories_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info(
            f"\nIngestion complete:\n"
            f"  Documents: {len(results)}\n"
            f"  Concepts: {total_concepts}\n"
            f"  Memories: {total_memories}\n"
            f"  Errors: {total_errors}"
        )

        # Print any errors
        for result in results:
            for error in result.errors:
                logger.error(f"  {result.document.title}: {error}")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
