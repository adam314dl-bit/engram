#!/usr/bin/env python3
"""Integration test for Phase 4: Reasoning & Synthesis.

This script tests the reasoning pipeline with real Neo4j data.
Exit criteria: "System answers questions, stores reasoning traces, can try alternatives."
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_reasoning_pipeline():
    """Test the full reasoning pipeline."""
    from engram.storage.neo4j_client import Neo4jClient
    from engram.reasoning.pipeline import ReasoningPipeline

    db = Neo4jClient()
    await db.connect()

    try:
        pipeline = ReasoningPipeline(db=db)

        print("\n" + "=" * 70)
        print("PHASE 4 INTEGRATION TEST: Reasoning & Synthesis")
        print("=" * 70)

        # Test 1: Basic reasoning
        print("\n--- Test 1: Basic Reasoning ---")
        query1 = "Что такое Docker и как он работает?"

        result1 = await pipeline.reason(query=query1)

        print(f"Query: {query1}")
        print(f"Answer: {result1.answer[:200]}...")
        print(f"Behavior: {result1.episode.behavior_name}")
        print(f"Behavior instruction: {result1.episode.behavior_instruction}")
        print(f"Domain: {result1.episode.domain}")
        print(f"Episode ID: {result1.episode_id}")
        print(f"Confidence: {result1.confidence:.0%}")
        print(f"Memories used: {len(result1.synthesis.memories_used)}")
        print(f"Concepts activated: {len(result1.synthesis.concepts_activated)}")

        # Verify episode was created
        stored_episode = await pipeline.episode_manager.get_episode(result1.episode_id)
        assert stored_episode is not None, "Episode was not stored"
        print("  Episode stored successfully")

        # Test 2: Different query type
        print("\n--- Test 2: How-to Query ---")
        query2 = "Как освободить место на диске в Docker?"

        result2 = await pipeline.reason(query=query2)

        print(f"Query: {query2}")
        print(f"Answer: {result2.answer[:200]}...")
        print(f"Behavior: {result2.episode.behavior_name}")
        print(f"Episode ID: {result2.episode_id}")

        # Test 3: Feedback recording
        print("\n--- Test 3: Feedback Recording ---")

        # Record positive feedback for first episode
        await pipeline.record_feedback(result1.episode_id, positive=True)
        updated_episode1 = await pipeline.episode_manager.get_episode(result1.episode_id)
        print(f"Episode 1 success count after positive feedback: {updated_episode1.success_count}")
        assert updated_episode1.success_count > 0, "Success not recorded"
        print("  Positive feedback recorded successfully")

        # Record negative feedback for second episode
        await pipeline.record_feedback(result2.episode_id, positive=False)
        updated_episode2 = await pipeline.episode_manager.get_episode(result2.episode_id)
        print(f"Episode 2 failure count after negative feedback: {updated_episode2.failure_count}")
        assert updated_episode2.failure_count > 0, "Failure not recorded"
        print("  Negative feedback recorded successfully")

        # Test 4: Re-reasoning after failure
        print("\n--- Test 4: Re-reasoning After Failure ---")

        re_result = await pipeline.re_reason(
            episode_id=result2.episode_id,
            user_feedback="Команда prune не помогла, образов слишком много",
        )

        print(f"Alternative answer: {re_result.answer[:200]}...")
        print(f"Alternative memories found: {len(re_result.alternative_memories)}")
        print(f"Alternative episodes found: {len(re_result.alternative_episodes)}")
        print(f"Approach changed: {re_result.approach_changed}")

        # Test 5: Finding similar episodes
        print("\n--- Test 5: Finding Similar Episodes ---")

        similar = await pipeline.episode_manager.find_similar_episodes(
            behavior_instruction="Объяснить концепцию Docker",
            k=5,
        )
        print(f"Found {len(similar)} similar episodes")
        for episode, similarity in similar[:3]:
            print(f"  [{similarity:.2f}] {episode.behavior_name}: {episode.behavior_instruction[:50]}...")

        print("\n" + "=" * 70)
        print("PHASE 4 EXIT CRITERIA: PASSED")
        print("- System answers questions")
        print("- Stores reasoning traces (episodes)")
        print("- Can try alternatives on failure")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await db.close()


if __name__ == "__main__":
    success = asyncio.run(test_reasoning_pipeline())
    sys.exit(0 if success else 1)
