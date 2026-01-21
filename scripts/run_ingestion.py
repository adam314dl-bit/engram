#!/usr/bin/env python3
"""Ingest documents into Engram.

Usage:
    python scripts/run_ingestion.py [directory]
    python scripts/run_ingestion.py --clear [directory]

If no directory is specified, uses tests/fixtures/mock_docs.
Recursively finds all .md and .txt files in the directory.

Options:
    --clear     Reset database before ingesting (deletes all data)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during progress bar
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def reset_database(db) -> bool:
    """Reset the database by deleting all nodes, relationships, and indexes."""
    print("\n" + "=" * 60)
    print("RESETTING DATABASE")
    print("=" * 60)

    try:
        # Drop vector indexes (they have dimension locked in)
        vector_indexes = [
            "concept_embeddings",
            "semantic_embeddings",
            "episodic_embeddings",
        ]
        for idx in vector_indexes:
            try:
                await db.execute_query(f"DROP INDEX {idx} IF EXISTS")
                print(f"  Dropped index: {idx}")
            except Exception as e:
                logger.warning(f"Could not drop index {idx}: {e}")

        # Drop fulltext index
        try:
            await db.execute_query("DROP INDEX semantic_content IF EXISTS")
            print("  Dropped index: semantic_content")
        except Exception:
            pass

        # Delete all nodes and relationships
        print("  Deleting all nodes and relationships...")
        await db.execute_query("MATCH (n) DETACH DELETE n")

        # Recreate schema (including vector indexes with correct dimensions)
        print("  Recreating schema...")
        await db.setup_schema()

        print("Database reset completed")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        logger.exception(f"Error resetting database: {e}")
        return False


async def run_ingestion(docs_dir: Path | None = None, clear: bool = False):
    """Ingest all documents from directory."""
    from engram.ingestion.pipeline import IngestionPipeline
    from engram.storage.neo4j_client import Neo4jClient

    # Use provided directory or default to mock_docs
    if docs_dir is None:
        docs_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_docs"

    if not docs_dir.exists():
        print(f"Documents directory not found: {docs_dir}")
        return False

    # Connect to Neo4j first (needed for reset)
    db = Neo4jClient()
    await db.connect()

    try:
        # Reset database if requested
        if clear:
            success = await reset_database(db)
            if not success:
                print("Database reset failed, aborting ingestion")
                return False

        print(f"Scanning directory (recursive): {docs_dir}")

        # Count documents (recursive)
        extensions = [".md", ".txt", ".markdown"]
        all_files = []
        for ext in extensions:
            all_files.extend(docs_dir.rglob(f"*{ext}"))
        print(f"Found {len(all_files)} documents to ingest")

        # Setup schema if needed (in case not already done by reset)
        await db.setup_schema()

        # Create pipeline
        pipeline = IngestionPipeline(neo4j_client=db)

        # Create progress bar
        pbar = tqdm(total=len(all_files), desc="Ingesting documents", unit="doc")

        # Track totals
        totals = {"concepts": 0, "memories": 0, "relations": 0, "errors": []}

        def on_progress(result):
            """Update progress bar after each document."""
            totals["concepts"] += result.concepts_created
            totals["memories"] += result.memories_created
            totals["relations"] += result.relations_created
            if result.errors:
                totals["errors"].append(result)
            pbar.set_postfix(
                concepts=totals["concepts"],
                memories=totals["memories"],
                relations=totals["relations"],
            )
            pbar.update(1)

        # Ingest directory with progress callback (recursive)
        results = await pipeline.ingest_directory(
            docs_dir,
            extensions=extensions,
            progress_callback=on_progress,
        )

        pbar.close()

        # Summary
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Documents processed: {len(results)}")
        print(f"Concepts created:    {totals['concepts']}")
        print(f"Memories created:    {totals['memories']}")
        print(f"Relations created:   {totals['relations']}")

        if totals["errors"]:
            print(f"\nFailed documents: {len(totals['errors'])}")
            for r in totals["errors"]:
                print(f"  - {r.document.title}: {r.errors}")

        print("=" * 60)

        return len(totals["errors"]) == 0

    finally:
        await db.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Engram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_ingestion.py                    # Ingest from default directory
    python scripts/run_ingestion.py /path/to/docs     # Ingest from custom directory
    python scripts/run_ingestion.py --clear           # Reset database, then ingest
    python scripts/run_ingestion.py --clear /path     # Reset database, then ingest from path
        """,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory containing documents to ingest (default: tests/fixtures/mock_docs)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Reset database before ingesting (deletes ALL data)",
    )

    args = parser.parse_args()

    # Validate directory if provided
    if args.directory and not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        return False

    try:
        success = await run_ingestion(args.directory, clear=args.clear)
        return success
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
