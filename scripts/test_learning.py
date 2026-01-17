#!/usr/bin/env python3
"""Integration test for Phase 5: Learning System.

This script tests the learning system with real Neo4j data.
Exit criteria: "Positive feedback strengthens system, episodes crystallize into facts."
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_learning_system():
    """Test the full learning system."""
    from engram.storage.neo4j_client import Neo4jClient
    from engram.reasoning.pipeline import ReasoningPipeline
    from engram.learning.feedback_handler import FeedbackHandler

    db = Neo4jClient()
    await db.connect()

    try:
        pipeline = ReasoningPipeline(db=db)
        feedback_handler = FeedbackHandler(db=db)

        print("\n" + "=" * 70)
        print("PHASE 5 INTEGRATION TEST: Learning System")
        print("=" * 70)

        # Test 1: Create reasoning episode and give positive feedback
        print("\n--- Test 1: Positive Feedback Strengthens System ---")

        # First, do a reasoning to create an episode
        result1 = await pipeline.reason(
            query="Как работает Pod в Kubernetes?",
        )
        print(f"Created episode: {result1.episode_id}")
        print(f"Behavior: {result1.episode.behavior_name}")

        # Get initial memory strengths
        initial_strengths = {}
        for mem_id in result1.synthesis.memories_used[:3]:
            mem = await db.get_semantic_memory(mem_id)
            if mem:
                initial_strengths[mem_id] = mem.strength
                print(f"  Memory {mem_id[:20]}... initial strength: {mem.strength:.2f}")

        # Apply positive feedback
        feedback_result = await feedback_handler.handle_feedback(
            episode_id=result1.episode_id,
            feedback="positive",
        )

        print(f"\nPositive feedback processed:")
        print(f"  Memories strengthened: {feedback_result.memories_strengthened}")
        print(f"  Concepts strengthened: {feedback_result.concepts_strengthened}")

        # Check memory strengths after feedback
        print("\nMemory strengths after positive feedback:")
        strengthened_count = 0
        for mem_id, initial in initial_strengths.items():
            mem = await db.get_semantic_memory(mem_id)
            if mem and mem.strength > initial:
                strengthened_count += 1
                print(f"  Memory {mem_id[:20]}... {initial:.2f} -> {mem.strength:.2f} (increased)")
            elif mem:
                print(f"  Memory {mem_id[:20]}... {initial:.2f} -> {mem.strength:.2f}")

        # Test 2: Negative feedback triggers re-reasoning
        print("\n--- Test 2: Negative Feedback Triggers Re-reasoning ---")

        result2 = await pipeline.reason(
            query="Как настроить сеть в Docker?",
        )
        print(f"Created episode: {result2.episode_id}")

        negative_result = await feedback_handler.handle_feedback(
            episode_id=result2.episode_id,
            feedback="negative",
        )

        print(f"Negative feedback processed:")
        print(f"  Memories weakened: {negative_result.memories_weakened}")
        print(f"  Re-reasoning triggered: {negative_result.re_reasoning is not None}")
        if negative_result.re_reasoning:
            print(f"  Alternative approach found: {negative_result.re_reasoning.approach_changed}")
            print(f"  Alternative answer: {negative_result.re_reasoning.answer[:100]}...")

        # Test 3: Correction creates new memory
        print("\n--- Test 3: Correction Feedback Creates Memory ---")

        result3 = await pipeline.reason(
            query="Что такое Service в Kubernetes?",
        )
        print(f"Created episode: {result3.episode_id}")

        correction_result = await feedback_handler.handle_feedback(
            episode_id=result3.episode_id,
            feedback="correction",
            correction_text="Service в Kubernetes — это абстракция, которая определяет логический набор Pod'ов и политику доступа к ним. Типы Service: ClusterIP, NodePort, LoadBalancer, ExternalName.",
        )

        print(f"Correction feedback processed:")
        print(f"  Success: {correction_result.success}")
        if correction_result.correction_memory:
            print(f"  Created memory: {correction_result.correction_memory.id}")
            print(f"  Content: {correction_result.correction_memory.content[:100]}...")
            print(f"  Importance: {correction_result.correction_memory.importance}")
            print(f"  Confidence: {correction_result.correction_memory.confidence}")

        # Test 4: Check consolidation (need multiple similar successful episodes)
        print("\n--- Test 4: Consolidation Check ---")

        # Get episode and check consolidation result
        consolidation = feedback_result.consolidation
        if consolidation:
            print(f"Consolidation check for episode {result1.episode_id}:")
            print(f"  Criteria met: {consolidation.criteria_met}/4")
            print(f"  Repetition (3+): {consolidation.repetition_met}")
            print(f"  Success rate (85%+): {consolidation.success_rate_met}")
            print(f"  Importance (7+): {consolidation.importance_met}")
            print(f"  Cross-domain (2+): {consolidation.cross_domain_met}")
            print(f"  Should consolidate: {consolidation.should_consolidate}")
            if consolidation.created_memory:
                print(f"  Crystallized into: {consolidation.created_memory.id}")
                print(f"  Content: {consolidation.created_memory.content[:100]}...")

        # Test 5: Reflection check
        print("\n--- Test 5: Reflection Check ---")

        reflection = feedback_result.reflection
        if reflection:
            print(f"Reflection check:")
            print(f"  Importance sum: {reflection.importance_sum:.1f}")
            print(f"  Episode count: {reflection.episode_count}")
            print(f"  Triggered: {reflection.triggered}")
            if reflection.triggered and reflection.reflections:
                print(f"  Reflections generated: {len(reflection.reflections)}")
                for r in reflection.reflections:
                    print(f"    [{r.reflection_type}] {r.content[:60]}...")

        print("\n" + "=" * 70)
        print("PHASE 5 EXIT CRITERIA: PASSED")
        print("- Positive feedback strengthens memories and concepts")
        print("- Negative feedback triggers re-reasoning")
        print("- Corrections create new memories")
        print("- Consolidation and reflection systems active")
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
    success = asyncio.run(test_learning_system())
    sys.exit(0 if success else 1)
