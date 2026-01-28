#!/usr/bin/env python3
"""Migration script for Engram v5.

Re-embeds all memories with BGE-M3 and builds FAISS index.

Usage:
    # Full migration
    uv run python scripts/migrate_to_v5.py

    # With options
    uv run python scripts/migrate_to_v5.py --batch-size 100 --index-path ./data/vector_index
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engram.config import settings
from engram.embeddings.bge_service import get_bge_embedding_service
from engram.embeddings.vector_index import VectorIndex
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


async def get_all_memories(db: Neo4jClient) -> list[dict]:
    """Fetch all memories from Neo4j."""
    query = """
    MATCH (m:SemanticMemory)
    WHERE m.status IS NULL OR m.status <> 'archived'
    RETURN m.id AS id, m.content AS content, m.search_content AS search_content
    """
    results = await db.execute_query(query)
    return [dict(r) for r in results]


async def migrate_to_v5(
    batch_size: int = 100,
    index_path: str | None = None,
    skip_existing: bool = False,
) -> None:
    """
    Migrate Engram to v5 by re-embedding memories with BGE-M3.

    Args:
        batch_size: Number of memories to embed per batch
        index_path: Path to save FAISS index
        skip_existing: Skip if index already exists
    """
    index_path = index_path or settings.vector_index_path

    # Check if index already exists
    index = VectorIndex()
    if skip_existing and index.exists(index_path):
        index.load(index_path)
        logger.info(f"Index already exists with {index.count} vectors, skipping migration")
        return

    # Connect to Neo4j
    db = Neo4jClient()
    await db.connect()

    try:
        # Get all memories
        logger.info("Fetching memories from Neo4j...")
        memories = await get_all_memories(db)
        logger.info(f"Found {len(memories)} memories to embed")

        if not memories:
            logger.warning("No memories found, nothing to migrate")
            return

        # Initialize BGE-M3 service
        logger.info("Loading BGE-M3 model...")
        bge_service = get_bge_embedding_service()
        bge_service.load_model()
        logger.info("BGE-M3 model ready")

        # Create new index
        index = VectorIndex(index_type=settings.vector_index_type)

        # Process in batches
        total_start = time.perf_counter()
        total_embedded = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(memories) + batch_size - 1) // batch_size

            # Extract texts (prefer search_content if available)
            texts = []
            chunk_ids = []
            for m in batch:
                text = m.get("search_content") or m.get("content") or ""
                if text.strip():
                    texts.append(text)
                    chunk_ids.append(m["id"])

            if not texts:
                continue

            # Embed batch
            batch_start = time.perf_counter()
            embeddings = bge_service.embed_batch_sync(texts)
            batch_time = time.perf_counter() - batch_start

            # Add to index
            index.add(embeddings, chunk_ids)
            total_embedded += len(texts)

            logger.info(
                f"Batch {batch_num}/{total_batches}: "
                f"embedded {len(texts)} memories in {batch_time:.1f}s "
                f"({len(texts) / batch_time:.1f} docs/s)"
            )

        total_time = time.perf_counter() - total_start
        logger.info(
            f"Migration complete: {total_embedded} memories embedded in {total_time:.1f}s "
            f"({total_embedded / total_time:.1f} docs/s overall)"
        )

        # Save index
        logger.info(f"Saving index to {index_path}...")
        index.save(index_path)
        logger.info(f"Index saved with {index.count} vectors")

    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate Engram to v5 (BGE-M3 + FAISS)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding (default: 100)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        help=f"Path to save FAISS index (default: {settings.vector_index_path})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip migration if index already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(migrate_to_v5(
        batch_size=args.batch_size,
        index_path=args.index_path,
        skip_existing=args.skip_existing,
    ))


if __name__ == "__main__":
    main()
