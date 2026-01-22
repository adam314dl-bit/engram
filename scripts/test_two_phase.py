#!/usr/bin/env python3
"""Test script for two-phase retrieval with LLM selection."""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_two_phase_retrieval():
    """Test the two-phase retrieval flow."""
    from engram.storage.neo4j_client import Neo4jClient
    from engram.reasoning.pipeline import ReasoningPipeline

    logger.info("Connecting to Neo4j...")
    db = Neo4jClient()
    await db.connect()

    try:
        # Check if we have any data
        stats = await db.get_stats()
        logger.info(f"Database stats: {stats}")

        if stats.get("semantic_memories", 0) == 0:
            logger.warning("No memories in database. Run ingestion first.")
            return

        # Create reasoning pipeline
        pipeline = ReasoningPipeline(db=db)

        # Test query
        query = "Что такое Docker?"
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}")

        # Test 1: Standard retrieval (for comparison)
        logger.info("\n--- Standard Retrieval ---")
        standard_result = await pipeline.reason(
            query=query,
            top_k_memories=10,
        )
        logger.info(f"Retrieved {len(standard_result.retrieval.memories)} memories")
        logger.info(f"Answer length: {len(standard_result.answer)} chars")
        logger.info(f"Confidence: {standard_result.confidence:.1%}")

        # Test 2: Two-phase retrieval with documents
        logger.info("\n--- Two-Phase Retrieval ---")
        doc_result = await pipeline.reason_with_documents(
            query=query,
            top_k_candidates=50,  # Use 50 for faster testing
        )

        logger.info(f"Retrieved {len(doc_result.retrieval.memories)} candidates")
        if doc_result.selection_result:
            logger.info(
                f"LLM selected {len(doc_result.selection_result.selected_ids)} memories "
                f"({doc_result.selection_result.selection_ratio:.1%})"
            )
        logger.info(f"Fetched {len(doc_result.source_documents)} source documents")
        logger.info(f"Answer length: {len(doc_result.answer)} chars")
        logger.info(f"Confidence: {doc_result.confidence:.1%}")

        # Show source documents
        if doc_result.source_documents:
            logger.info("\nSource documents used:")
            for doc in doc_result.source_documents[:5]:
                logger.info(f"  - {doc.title} ({len(doc.content)} chars)")

        # Show answers
        logger.info(f"\n{'='*60}")
        logger.info("STANDARD ANSWER:")
        logger.info(f"{'='*60}")
        print(standard_result.answer[:500] + "..." if len(standard_result.answer) > 500 else standard_result.answer)

        logger.info(f"\n{'='*60}")
        logger.info("TWO-PHASE ANSWER:")
        logger.info(f"{'='*60}")
        print(doc_result.answer[:500] + "..." if len(doc_result.answer) > 500 else doc_result.answer)

    finally:
        await db.close()


async def test_selector_only():
    """Test just the memory selector."""
    from engram.storage.neo4j_client import Neo4jClient
    from engram.retrieval.pipeline import RetrievalPipeline
    from engram.reasoning.selector import MemorySelector

    logger.info("Testing memory selector...")
    db = Neo4jClient()
    await db.connect()

    try:
        retrieval = RetrievalPipeline(db=db)
        selector = MemorySelector()

        query = "Что такое Docker?"

        # Get candidates
        logger.info(f"Retrieving candidates for: {query}")
        result = await retrieval.retrieve_candidates(query=query, top_k_memories=30)
        logger.info(f"Got {len(result.memories)} candidates")

        # Show some candidates
        logger.info("\nTop 5 candidates:")
        for i, sm in enumerate(result.memories[:5], 1):
            content = sm.memory.content[:100] + "..." if len(sm.memory.content) > 100 else sm.memory.content
            logger.info(f"  {i}. [{sm.memory.id}] {content}")

        # Select with LLM
        logger.info("\nAsking LLM to select relevant memories...")
        selection = await selector.select(query=query, candidates=result.memories)

        logger.info(f"Selected {len(selection.selected_ids)} memories:")
        for mem_id in selection.selected_ids[:10]:
            logger.info(f"  - {mem_id}")

    finally:
        await db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selector":
        asyncio.run(test_selector_only())
    else:
        asyncio.run(test_two_phase_retrieval())
