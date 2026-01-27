#!/usr/bin/env python
"""Debug full retrieval + generation pipeline.

Traces chunks through every pipeline stage and generates final answer.
Shows all timings to identify bottlenecks.

Usage:
    uv run python scripts/debug_retrieval.py "query text"
    uv run python scripts/debug_retrieval.py "query" --search-text "Epic"
    uv run python scripts/debug_retrieval.py "query" --find-chunk "mem-123"
    uv run python scripts/debug_retrieval.py "query" -v --show-context
    uv run python scripts/debug_retrieval.py "query" --save-trace
    uv run python scripts/debug_retrieval.py "query" --no-answer  # Skip answer generation

Examples:
    # Full pipeline with answer
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
import time
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


def format_duration(ms: float) -> str:
    """Format duration in human-readable form."""
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def print_timing_bar(label: str, duration_ms: float, max_ms: float, width: int = 40) -> None:
    """Print a visual timing bar."""
    bar_len = int((duration_ms / max_ms) * width) if max_ms > 0 else 0
    bar = "█" * bar_len + "░" * (width - bar_len)
    print(f"  {label:25} {bar} {format_duration(duration_ms):>8}")


async def main(args: argparse.Namespace) -> None:
    """Run full pipeline debug."""
    setup_logging(args.verbose)

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    print(f"\n{'='*60}")
    print(f"  QUERY: {args.query}")
    print(f"{'='*60}")

    # Get database client
    db_start = time.perf_counter()
    db = await get_client()
    timings["db_connect"] = (time.perf_counter() - db_start) * 1000

    # Create traced retriever
    traced = TracedRetriever(db)

    # Run retrieval with tracing
    print("\n[1/2] Running retrieval pipeline...")
    retrieval_start = time.perf_counter()
    result, trace = await traced.retrieve_with_trace(
        query=args.query,
        top_k_memories=args.top_k,
    )
    timings["retrieval_total"] = (time.perf_counter() - retrieval_start) * 1000

    # Extract step timings from trace
    for step in trace.steps:
        timings[f"  {step.step_name}"] = step.duration_ms

    # Generate answer (unless --no-answer)
    answer = None
    if not args.no_answer:
        print("[2/2] Generating answer...")

        from engram.reasoning.synthesizer import ResponseSynthesizer
        from engram.ingestion.llm_client import get_llm_client

        synth_start = time.perf_counter()
        synthesizer = ResponseSynthesizer(llm_client=get_llm_client())

        answer_start = time.perf_counter()
        answer = await synthesizer.synthesize(
            query=args.query,
            retrieval=result,
        )
        timings["answer_generation"] = (time.perf_counter() - answer_start) * 1000
        timings["synthesizer_init"] = (synth_start - retrieval_start) * 1000  # Minimal
    else:
        print("[2/2] Skipping answer generation (--no-answer)")

    total_duration = (time.perf_counter() - total_start) * 1000
    timings["total"] = total_duration

    # Print retrieval trace summary
    print(f"\n{'='*60}")
    print("  RETRIEVAL TRACE")
    print(f"{'='*60}")
    print(trace.summary())

    # Print timing breakdown
    print(f"\n{'='*60}")
    print("  TIMING BREAKDOWN")
    print(f"{'='*60}")

    max_time = max(timings.values())

    # Retrieval steps
    print("\n  Retrieval Pipeline:")
    for step in trace.steps:
        print_timing_bar(step.step_name, step.duration_ms, max_time)

    print(f"\n  {'─'*55}")
    print_timing_bar("retrieval_total", timings["retrieval_total"], max_time)

    # Answer generation
    if not args.no_answer:
        print("\n  Answer Generation:")
        print_timing_bar("answer_generation", timings["answer_generation"], max_time)

    # Total
    print(f"\n  {'─'*55}")
    print_timing_bar("TOTAL", total_duration, max_time)

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Concepts extracted:     {', '.join(trace.extracted_concepts) or 'None'}")
    print(f"  Memories retrieved:     {len(result.memories)}")
    print(f"  Sources:                {dict(result.retrieval_sources)}")
    if result.path_result:
        pr = result.path_result
        print(f"  Path retrieval:         {len(pr.shared_memories)} shared, {len(pr.path_memories)} path")
    print(f"  Retrieval time:         {format_duration(timings['retrieval_total'])}")
    if not args.no_answer:
        print(f"  Answer generation:      {format_duration(timings['answer_generation'])}")
    print(f"  Total time:             {format_duration(total_duration)}")

    # Print answer
    if answer and not args.no_answer:
        print(f"\n{'='*60}")
        print("  ANSWER")
        print(f"{'='*60}")
        print(f"\n{answer.response}\n")

        if answer.sources:
            print("  Sources:")
            for src in answer.sources[:5]:
                title = src.get("title", "Unknown")
                url = src.get("url", "")
                print(f"    - {title}")
                if url:
                    print(f"      {url}")

    # Search for specific text if requested
    if args.search_text:
        print(f"\n{'='*60}")
        print(f"  CHUNKS CONTAINING '{args.search_text}'")
        print(f"{'='*60}")
        matching_chunks = traced.find_chunks_by_text(trace, args.search_text)

        if not matching_chunks:
            print(f"\n  No chunks found containing '{args.search_text}'")
        else:
            print(f"\n  Found {len(matching_chunks)} matching chunks:\n")
            for i, chunk in enumerate(matching_chunks, 1):
                print(f"  [{i}] ID: {chunk.memory_id}")
                print(f"      Sources: {', '.join(chunk.sources)}")
                print(f"      Included: {chunk.included}")
                if chunk.included:
                    print(f"      Final rank: {chunk.final_rank}")
                else:
                    print(f"      Dropped at: {chunk.dropped_at()}")

                # Show journey through pipeline
                if args.verbose:
                    journey = trace.find_chunk_journey(chunk.memory_id)
                    print("      Journey:")
                    for step_info in journey.get("journey", []):
                        marker = "+" if step_info["present"] else "-"
                        if step_info["present"]:
                            print(f"        {marker} {step_info['step']}: "
                                  f"score={step_info['score']:.4f}, rank={step_info['rank']}")
                        else:
                            print(f"        {marker} {step_info['step']}")

                print(f"\n      Content preview: {chunk.content_preview}...")
                print()

    # Find specific chunk if requested
    if args.find_chunk:
        print(f"\n{'='*60}")
        print(f"  JOURNEY FOR CHUNK '{args.find_chunk}'")
        print(f"{'='*60}")
        journey = trace.find_chunk_journey(args.find_chunk)

        if "error" in journey:
            print(f"\n  {journey['error']}")
        else:
            print(f"\n  ID: {journey['memory_id']}")
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
        print(f"\n{'='*60}")
        print("  TOP RESULTS WITH CONTEXT")
        print(f"{'='*60}")
        for i, sm in enumerate(result.memories[:10], 1):
            chunk = trace.chunk_traces.get(sm.memory.id)
            sources = chunk.sources if chunk else []
            print(f"\n  {i}. Score: {sm.score:.4f} | Sources: {', '.join(sources)}")
            print(f"     ID: {sm.memory.id[:40]}...")
            content = sm.memory.content[:300] if sm.memory.content else "N/A"
            print(f"     Content: {content}...")

    # Show path retrieval details
    if result.path_result and args.verbose:
        print(f"\n{'='*60}")
        print("  PATH RETRIEVAL DETAILS")
        print(f"{'='*60}")
        pr = result.path_result
        print(f"\n  Paths found: {len(pr.paths)}")
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
        print(f"\n  Trace saved to: {filepath}")

    # Close database
    await db.close()

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug full retrieval + generation pipeline with timing",
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
        "--no-answer",
        action="store_true",
        help="Skip answer generation (retrieval only)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with full journey details",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
