#!/usr/bin/env python3
"""Standalone script to build FAISS vector index.

Fetches all memories from Neo4j, embeds with BGE-M3, and builds FAISS index.
Can be run independently of full migration.

Usage:
    # Build index
    uv run python scripts/build_vector_index.py

    # With options
    uv run python scripts/build_vector_index.py --batch-size 200 --output ./data/my_index

    # Recreate Neo4j vector indexes (after changing embedding dimensions)
    uv run python scripts/build_vector_index.py --recreate-neo4j-indexes --force
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


async def recreate_neo4j_vector_indexes(db: Neo4jClient, dimensions: int = 1024) -> None:
    """
    Drop and recreate Neo4j vector indexes with specified dimensions.

    Args:
        db: Neo4j client
        dimensions: Vector dimensions (default 1024 for BGE-M3)
    """
    logger.info(f"Recreating Neo4j vector indexes with {dimensions} dimensions...")

    # Drop existing indexes
    drop_queries = [
        "DROP INDEX concept_embeddings IF EXISTS",
        "DROP INDEX semantic_embeddings IF EXISTS",
        "DROP INDEX episodic_embeddings IF EXISTS",
    ]

    for query in drop_queries:
        try:
            await db.execute_query(query)
            logger.info(f"Dropped: {query.split()[2]}")
        except Exception as e:
            logger.warning(f"Drop warning (may not exist): {e}")

    # Recreate with new dimensions
    create_queries = [
        f"""
        CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
        FOR (c:Concept) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
        f"""
        CREATE VECTOR INDEX semantic_embeddings IF NOT EXISTS
        FOR (s:SemanticMemory) ON (s.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
        f"""
        CREATE VECTOR INDEX episodic_embeddings IF NOT EXISTS
        FOR (e:EpisodicMemory) ON (e.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
    ]

    for query in create_queries:
        try:
            await db.execute_query(query)
            index_name = query.split("INDEX")[1].split("IF")[0].strip()
            logger.info(f"Created: {index_name} ({dimensions} dims)")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    logger.info("Neo4j vector indexes recreated successfully")


async def fetch_memories(
    db: Neo4jClient,
    status_filter: str | None = None,
) -> list[dict]:
    """
    Fetch memories from Neo4j.

    Args:
        db: Neo4j client
        status_filter: Optional status to filter by (None = all non-archived)

    Returns:
        List of memory dicts with id, content, search_content
    """
    if status_filter:
        query = """
        MATCH (m:SemanticMemory)
        WHERE m.status = $status
        RETURN m.id AS id, m.content AS content, m.search_content AS search_content
        """
        results = await db.execute_query(query, status=status_filter)
    else:
        query = """
        MATCH (m:SemanticMemory)
        WHERE m.status IS NULL OR m.status <> 'archived'
        RETURN m.id AS id, m.content AS content, m.search_content AS search_content
        """
        results = await db.execute_query(query)

    return [dict(r) for r in results]


async def build_index(
    output_path: str | None = None,
    batch_size: int = 100,
    index_type: str = "flat",
    force: bool = False,
    recreate_neo4j_indexes: bool = False,
) -> None:
    """
    Build FAISS vector index from Neo4j memories.

    Args:
        output_path: Path to save index
        batch_size: Embedding batch size
        index_type: FAISS index type (flat or ivf)
        force: Overwrite existing index
        recreate_neo4j_indexes: Drop and recreate Neo4j vector indexes with 1024 dims
    """
    output_path = output_path or settings.vector_index_path

    # Check existing
    if not force and Path(output_path).exists() and (Path(output_path) / "index.faiss").exists():
        logger.error(f"Index already exists at {output_path}. Use --force to overwrite.")
        return

    # Connect to Neo4j
    logger.info("Connecting to Neo4j...")
    db = Neo4jClient()
    await db.connect()

    # Recreate Neo4j vector indexes if requested
    if recreate_neo4j_indexes:
        await recreate_neo4j_vector_indexes(db, dimensions=1024)

    try:
        # Fetch memories
        logger.info("Fetching memories...")
        memories = await fetch_memories(db)
        logger.info(f"Found {len(memories)} memories")

        if not memories:
            logger.warning("No memories to index")
            return

        # Load BGE-M3
        logger.info("Loading BGE-M3 model...")
        bge = get_bge_embedding_service()
        bge.load_model()

        # Create index
        index = VectorIndex(index_type=index_type)

        # Process batches
        start_time = time.perf_counter()
        total = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]

            # Extract texts
            texts = []
            ids = []
            for m in batch:
                text = m.get("search_content") or m.get("content") or ""
                if text.strip():
                    texts.append(text)
                    ids.append(m["id"])

            if not texts:
                continue

            # Embed
            embeddings = bge.embed_batch_sync(texts)

            # Add to index
            index.add(embeddings, ids)
            total += len(texts)

            progress = (i + len(batch)) / len(memories) * 100
            logger.info(f"Progress: {progress:.1f}% ({total} vectors)")

        elapsed = time.perf_counter() - start_time
        logger.info(f"Embedded {total} memories in {elapsed:.1f}s")

        # Save
        logger.info(f"Saving index to {output_path}...")
        index.save(output_path)
        logger.info(f"Done! Index has {index.count} vectors")

    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index from Neo4j memories")
    parser.add_argument(
        "-o", "--output",
        type=str,
        help=f"Output path (default: {settings.vector_index_path})",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Embedding batch size (default: 100)",
    )
    parser.add_argument(
        "-t", "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="FAISS index type (default: flat)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing index",
    )
    parser.add_argument(
        "--recreate-neo4j-indexes",
        action="store_true",
        help="Drop and recreate Neo4j vector indexes with 1024 dimensions (BGE-M3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    asyncio.run(build_index(
        output_path=args.output,
        batch_size=args.batch_size,
        index_type=args.index_type,
        force=args.force,
        recreate_neo4j_indexes=args.recreate_neo4j_indexes,
    ))


if __name__ == "__main__":
    main()
