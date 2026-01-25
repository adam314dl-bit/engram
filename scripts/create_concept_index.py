#!/usr/bin/env python3
"""Create concept_content fulltext index for BM25+Graph mode (v4.7).

This index is required for concept matching without vector search.
Run once after deploying v4.7 code.

Usage:
    uv run python scripts/create_concept_index.py
"""

import asyncio
import logging

from engram.storage.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_concept_fulltext_index() -> None:
    """Create the concept_content fulltext index if it doesn't exist."""
    db = Neo4jClient()
    await db.connect()
    logger.info("Connected to Neo4j")

    try:
        # Check if index already exists
        check_query = "SHOW FULLTEXT INDEXES WHERE name = 'concept_content'"
        async with db.session() as session:
            result = await session.run(check_query)
            existing = [record async for record in result]

            if existing:
                logger.info("Index 'concept_content' already exists, skipping creation")
                return

        # Create the index
        create_query = """
        CREATE FULLTEXT INDEX concept_content IF NOT EXISTS
        FOR (c:Concept) ON EACH [c.name]
        """
        async with db.session() as session:
            await session.run(create_query)
            logger.info("Created fulltext index 'concept_content' on Concept.name")

        # Verify creation
        async with db.session() as session:
            result = await session.run(check_query)
            existing = [record async for record in result]
            if existing:
                logger.info("Index verified successfully")
            else:
                logger.warning("Index creation may have failed - please check manually")

    finally:
        await db.close()
        logger.info("Disconnected from Neo4j")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Creating concept_content fulltext index for v4.7")
    print("=" * 60)
    asyncio.run(create_concept_fulltext_index())
    print("Done!")


if __name__ == "__main__":
    main()
