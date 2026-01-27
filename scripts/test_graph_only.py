#!/usr/bin/env python
"""Test graph-only retrieval with reranking (no BM25, no fusion).

Usage:
    uv run python scripts/test_graph_only.py "query text"
    uv run python scripts/test_graph_only.py "query" --top-k 20
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engram.config import settings
from engram.retrieval.concept_extractor import ConceptExtractor
from engram.retrieval.spreading import SpreadingActivation
from engram.retrieval.reranker import rerank_with_fallback
from engram.storage.neo4j_client import get_client


async def main(args: argparse.Namespace) -> None:
    db = await get_client()

    print(f"\nQuery: {args.query}")
    print("-" * 60)

    # 1. Extract concepts
    extractor = ConceptExtractor()
    concept_result = await extractor.extract(args.query)
    print(f"\nExtracted concepts: {[c.name for c in concept_result.concepts]}")

    # 2. Match concepts in graph
    seed_concept_ids = []
    for concept in concept_result.concepts:
        existing = await db.get_concept_by_name(concept.name)
        if existing:
            seed_concept_ids.append(existing.id)
            print(f"  Matched: {concept.name} -> {existing.id[:20]}...")

    if not seed_concept_ids:
        # Fallback to BM25 concept search
        print("  No exact matches, using BM25 concept search...")
        bm25_concepts = await db.fulltext_search_concepts(query_text=args.query, k=5)
        for c, score in bm25_concepts:
            if score > 0.5:
                seed_concept_ids.append(c.id)
                print(f"  BM25 match: {c.name} (score={score:.2f})")

    print(f"\nSeed concepts: {len(seed_concept_ids)}")

    # 3. Spreading activation
    spreading = SpreadingActivation(db=db)
    activation_result = await spreading.activate(seed_concept_ids, [])
    activated = activation_result.activations

    print(f"Activated concepts: {len(activated)}")

    # 4. Get memories from activated concepts
    active_ids = [
        cid for cid, act in activated.items()
        if act >= settings.activation_threshold
    ]
    print(f"Above threshold ({settings.activation_threshold}): {len(active_ids)}")

    if not active_ids:
        print("No concepts above threshold!")
        await db.close()
        return

    graph_memories = await db.get_memories_for_concepts(
        concept_ids=active_ids,
        limit=args.top_k * 3,
    )

    print(f"\nGraph memories retrieved: {len(graph_memories)}")

    # Score by sum of activations
    scored = []
    for memory in graph_memories:
        score = sum(activated.get(cid, 0) for cid in memory.concept_ids)
        scored.append((memory, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Graph Results (before reranking) ---")
    for i, (mem, score) in enumerate(scored[:args.top_k], 1):
        content = (mem.content or "")[:100]
        print(f"{i:2}. score={score:.3f} | {content}...")

    # 5. Apply reranker
    if settings.reranker_enabled:
        print(f"\n--- After Reranking (top {args.top_k}) ---")
        candidates = [
            (mem.id, mem.content, score)
            for mem, score in scored[:settings.reranker_candidates]
        ]

        reranked = rerank_with_fallback(args.query, candidates, top_k=args.top_k)

        # Build lookup
        mem_lookup = {mem.id: mem for mem, _ in scored}

        for i, item in enumerate(reranked[:args.top_k], 1):
            mem = mem_lookup.get(item.id)
            if mem:
                content = (mem.content or "")[:100]
                print(f"{i:2}. rerank={item.rerank_score:.3f} orig={item.original_score:.3f} | {content}...")

    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test graph-only retrieval")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    args = parser.parse_args()
    asyncio.run(main(args))
