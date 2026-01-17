#!/usr/bin/env python3
"""Ingest mock documents into Engram."""

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


async def run_ingestion():
    """Ingest all mock documents."""
    from engram.ingestion.pipeline import IngestionPipeline
    from engram.storage.neo4j_client import Neo4jClient

    docs_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_docs"

    if not docs_dir.exists():
        print(f"Documents directory not found: {docs_dir}")
        print("Run 'python scripts/generate_mock_docs.py' first to create documents.")
        return False

    # Count documents
    md_files = list(docs_dir.glob("*.md"))
    print(f"Found {len(md_files)} documents to ingest")

    # Connect to Neo4j
    db = Neo4jClient()
    await db.connect()

    try:
        # Setup schema if needed
        await db.setup_schema()

        # Create pipeline
        pipeline = IngestionPipeline(neo4j_client=db)

        # Create progress bar
        pbar = tqdm(total=len(md_files), desc="Ingesting documents", unit="doc")

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

        # Ingest directory with progress callback
        results = await pipeline.ingest_directory(
            docs_dir,
            extensions=[".md"],
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
    try:
        success = await run_ingestion()
        return success
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
