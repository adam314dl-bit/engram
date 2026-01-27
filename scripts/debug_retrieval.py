#!/usr/bin/env python
"""Debug retrieval for specific queries.

Traces chunks through every pipeline stage to understand where
relevant chunks appear and disappear.

Usage:
    uv run python scripts/debug_retrieval.py "query text"
    uv run python scripts/debug_retrieval.py "query" --search-text "Epic"
    uv run python scripts/debug_retrieval.py "query" --find-chunk "mem-123"
    uv run python scripts/debug_retrieval.py "query" -v --show-context
    uv run python scripts/debug_retrieval.py "query" --save-trace

Examples:
    # Basic query debug
    uv run python scripts/debug_retrieval.py "какие типы задач в jira"

    # Find where "Epic" chunks appear
    uv run python scripts/debug_retrieval.py "типы задач в jira" --search-text "Epic"

    # Track a specific chunk through the pipeline
    uv run python scripts/debug_retrieval.py "jira" --find-chunk "mem-abc123"

    # Verbose output with full context
    uv run python scripts/debug_retrieval.py "jira" -v --show-context

    # Save trace to file for later analysis
    uv run python scripts/debug_retrieval.py "jira" --save-trace
"""

# IMPORTANT: Set HuggingFace offline mode BEFORE any imports
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engram.retrieval.traced_retriever import TracedRetriever
from engram.storage.neo4j_client import get_client


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from neo4j driver
    logging.getLogger("neo4j").setLevel(logging.WARNING)


async def main(args: argparse.Namespace) -> None:
    """Run retrieval debug."""
    setup_logging(args.verbose)

    print(f"\nQuery: {args.query}")
    print("-" * 60)

    # Get database client
    db = await get_client()

    # Create traced retriever
    traced = TracedRetriever(db)

    # Run retrieval with tracing
    print("\nRunning traced retrieval...")
    result, trace = await traced.retrieve_with_trace(
        query=args.query,
        top_k_memories=args.top_k,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(trace.summary())
    print("=" * 60)

    # Search for specific text if requested
    if args.search_text:
        print(f"\n--- Chunks containing '{args.search_text}' ---")
        matching_chunks = traced.find_chunks_by_text(trace, args.search_text)

        if not matching_chunks:
            print(f"  No chunks found containing '{args.search_text}'")
        else:
            print(f"  Found {len(matching_chunks)} matching chunks:\n")
            for i, chunk in enumerate(matching_chunks, 1):
                print(f"  [{i}] ID: {chunk.memory_id}")
                print(f"  Sources: {', '.join(chunk.sources)}")
                print(f"  Included: {chunk.included}")
                if chunk.included:
                    print(f"  Final rank: {chunk.final_rank}")
                else:
                    print(f"  Dropped at: {chunk.dropped_at()}")
                print()
                print(f"  --- Full Content ---")
                print(f"  {chunk.full_content}")
                print(f"  --- End Content ---")
                print()

                # Show journey through pipeline
                if args.verbose:
                    journey = trace.find_chunk_journey(chunk.memory_id)
                    print("  Journey:")
                    for step_info in journey.get("journey", []):
                        marker = "+" if step_info["present"] else "-"
                        if step_info["present"]:
                            print(f"    {marker} {step_info['step']}: "
                                  f"score={step_info['score']:.4f}, rank={step_info['rank']}")
                        else:
                            print(f"    {marker} {step_info['step']}")
                    print()
                print("-" * 60)

    # Find specific chunk if requested
    if args.find_chunk:
        print(f"\n--- Journey for chunk '{args.find_chunk}' ---")
        journey = trace.find_chunk_journey(args.find_chunk)

        if "error" in journey:
            print(f"  {journey['error']}")
        else:
            print(f"  ID: {journey['memory_id']}")
            print(f"  Preview: {journey['content_preview'][:80]}...")
            print(f"  Sources: {', '.join(journey['sources'])}")
            print(f"  Included: {journey['included']}")
            if journey['included']:
                print(f"  Final rank: {journey['final_rank']}")
            print("\n  Pipeline journey:")
            for step_info in journey["journey"]:
                marker = "+" if step_info["present"] else "-"
                if step_info["present"]:
                    print(f"    {marker} {step_info['step']}: "
                          f"score={step_info['score']:.4f}, rank={step_info['rank']}")
                else:
                    print(f"    {marker} {step_info['step']}")

    # Show final results with context
    if args.show_context:
        print("\n--- Final Results ---")
        for i, sm in enumerate(result.memories[:10], 1):
            chunk = trace.chunk_traces.get(sm.memory.id)
            sources = chunk.sources if chunk else []
            print(f"\n{i}. Score: {sm.score:.4f} | Sources: {', '.join(sources)}")
            print(f"   ID: {sm.memory.id[:30]}...")
            content = sm.memory.content[:200] if sm.memory.content else "N/A"
            print(f"   Content: {content}...")

    # Show path retrieval details
    if result.path_result and args.verbose:
        print("\n--- Path Retrieval Details ---")
        pr = result.path_result
        print(f"  Paths found: {len(pr.paths)}")
        print(f"  Bridge concepts: {len(pr.bridge_concepts)}")
        print(f"  Shared memories: {len(pr.shared_memories)}")
        print(f"  Path memories: {len(pr.path_memories)}")

        if pr.paths:
            print("\n  Paths between concepts:")
            for path in pr.paths[:5]:
                print(f"    {path['src_id'][:15]}... -> {path['tgt_id'][:15]}... "
                      f"(len={path['length']})")

        if pr.bridge_concepts:
            print("\n  Top bridge concepts:")
            for bridge in pr.bridge_concepts[:5]:
                print(f"    {bridge['name']} (connects {bridge['count']} query concepts)")

    # Save trace if requested
    if args.save_trace:
        filepath = trace.save()
        print(f"\nTrace saved to: {filepath}")

    # Close database
    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug retrieval by tracing chunks through the pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "query",
        help="Query text to debug",
    )

    parser.add_argument(
        "--search-text",
        help="Search for chunks containing this text",
    )

    parser.add_argument(
        "--find-chunk",
        help="Track a specific chunk ID through the pipeline",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of results to return (default: 20)",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show full content context for top results",
    )

    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save trace to file",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with full journey details",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
