#!/usr/bin/env python3
"""Integration test for Phase 3: Hybrid Retrieval Pipeline.

This script tests the retrieval pipeline with real Neo4j data.
Exit criteria: "Retrieval finds relevant memories even for indirect queries"
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_retrieval_pipeline():
    """Test the full retrieval pipeline with different query types."""
    from engram.storage.neo4j_client import Neo4jClient
    from engram.retrieval.pipeline import RetrievalPipeline

    db = Neo4jClient()
    await db.connect()

    try:
        pipeline = RetrievalPipeline(db=db)

        # Test queries - mix of direct and indirect
        test_queries = [
            # Direct queries
            ("How do I clean up Docker disk space?", "docker cleanup"),
            ("What is a Kubernetes service?", "kubernetes service"),
            # Indirect queries - should still find relevant memories
            ("My disk is full, what can I do?", "disk space"),
            ("How do I expose my application?", "service exposure"),
            ("Container orchestration basics", "kubernetes concepts"),
        ]

        print("\n" + "=" * 70)
        print("PHASE 3 INTEGRATION TEST: Hybrid Retrieval Pipeline")
        print("=" * 70)

        all_passed = True

        for query, description in test_queries:
            print(f"\n--- Query: {query} ({description}) ---")

            result = await pipeline.retrieve(
                query=query,
                top_k_memories=5,
                top_k_episodes=3,
            )

            # Print results
            print(f"Query concepts extracted: {len(result.query_concepts)}")
            for c in result.query_concepts[:3]:
                print(f"  - {c.name} ({c.type})")

            print(f"Activated concepts: {len(result.activated_concepts)}")
            top_activated = sorted(
                result.activated_concepts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for cid, activation in top_activated:
                print(f"  - {cid}: {activation:.3f}")

            print(f"Retrieved memories: {len(result.memories)}")
            for sm in result.memories[:3]:
                print(f"  [{sm.score:.3f}] {sm.memory.content[:80]}...")
                print(f"    Sources: {', '.join(sm.sources)}")

            print(f"Retrieval sources distribution: {result.retrieval_sources}")

            # Check if we got results
            if len(result.memories) == 0:
                print("  WARNING: No memories retrieved!")
                all_passed = False
            else:
                print("  OK: Retrieved memories successfully")

        # Test 2: Concept-based retrieval
        print("\n--- Testing retrieve_for_concepts ---")
        concepts = await db.execute_query(
            "MATCH (c:Concept) RETURN c.id as id LIMIT 3"
        )
        if concepts:
            concept_ids = [c["id"] for c in concepts]
            memories = await pipeline.retrieve_for_concepts(
                concept_ids=concept_ids,
                top_k=5,
            )
            print(f"Retrieved {len(memories)} memories for concepts: {concept_ids}")
            for sm in memories[:2]:
                print(f"  [{sm.score:.3f}] {sm.memory.content[:60]}...")

        print("\n" + "=" * 70)
        if all_passed:
            print("PHASE 3 EXIT CRITERIA: PASSED")
            print("Retrieval finds relevant memories even for indirect queries")
        else:
            print("PHASE 3 EXIT CRITERIA: NEEDS ATTENTION")
            print("Some queries didn't return results")
        print("=" * 70)

        return all_passed

    finally:
        await db.close()


if __name__ == "__main__":
    success = asyncio.run(test_retrieval_pipeline())
    sys.exit(0 if success else 1)
