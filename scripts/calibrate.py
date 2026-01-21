#!/usr/bin/env python3
"""Engram v3 Calibration CLI.

Phase 0 calibration tool for tuning retrieval and memory parameters.
Helps find optimal values for:
- RRF k parameter
- Reranker thresholds
- ACT-R decay and threshold parameters
- BM25 preprocessing settings

Usage:
    uv run python scripts/calibrate.py --help
    uv run python scripts/calibrate.py test-russian
    uv run python scripts/calibrate.py test-reranker
    uv run python scripts/calibrate.py test-actr
    uv run python scripts/calibrate.py benchmark
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engram.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_russian_preprocessing() -> None:
    """Test Russian NLP preprocessing."""
    print("\n=== Testing Russian Preprocessing ===\n")

    try:
        from engram.preprocessing.russian import (
            get_morph_analyzer,
            is_russian_text,
            lemmatize_text,
            lemmatize_word,
            prepare_bm25_query,
            tokenize,
        )
    except ImportError as e:
        print(f"Error importing preprocessing module: {e}")
        print("Make sure pymorphy3 is installed: uv sync")
        return

    # Test text samples
    samples = [
        "Привет, как дела?",
        "Контейнеры Docker используют изолированные namespace",
        "Машинное обучение и нейронные сети",
        "Python — интерпретируемый язык программирования",
        "Mixed text with русскими словами and English words",
    ]

    print("1. Testing tokenization:")
    for sample in samples[:2]:
        tokens = tokenize(sample)
        print(f"   Input:  {sample}")
        print(f"   Tokens: {tokens}\n")

    print("\n2. Testing lemmatization:")
    words = ["контейнеры", "используют", "изолированные", "нейронные", "программирования"]
    for word in words:
        lemma = lemmatize_word(word)
        print(f"   {word} -> {lemma}")

    print("\n3. Testing full text lemmatization:")
    for sample in samples[:3]:
        lemmatized = lemmatize_text(sample, remove_stopwords=True)
        print(f"   Input:  {sample}")
        print(f"   Output: {lemmatized}\n")

    print("\n4. Testing BM25 query preparation:")
    queries = [
        "Как настроить Docker контейнер?",
        "Проблема с памятью в Python",
    ]
    for query in queries:
        prepared = prepare_bm25_query(query)
        print(f"   Query:    {query}")
        print(f"   Prepared: {prepared}\n")

    print("\n5. Testing language detection:")
    texts = [
        ("Это русский текст", True),
        ("This is English text", False),
        ("Русский текст with English", True),  # ~55% Cyrillic
    ]
    for text, expected in texts:
        is_russian = is_russian_text(text)
        status = "✓" if is_russian == expected else "✗"
        print(f"   {status} '{text}' -> is_russian={is_russian}")

    print("\n6. Testing MorphAnalyzer caching:")
    import time
    start = time.time()
    _ = get_morph_analyzer()
    first_load = time.time() - start

    start = time.time()
    _ = get_morph_analyzer()
    cached_load = time.time() - start

    print(f"   First load:  {first_load:.3f}s")
    print(f"   Cached load: {cached_load:.6f}s")
    if cached_load > 0:
        print(f"   Speedup:     {first_load / cached_load:.0f}x")
    else:
        print("   Speedup:     (instant - cache hit)")

    print("\n✓ Russian preprocessing tests complete!")


def test_reranker() -> None:
    """Test BGE reranker."""
    print("\n=== Testing BGE Reranker ===\n")

    if not settings.reranker_enabled:
        print("Reranker is disabled in settings. Enable with RERANKER_ENABLED=true")
        return

    try:
        from engram.retrieval.reranker import (
            clear_reranker_cache,
            get_reranker,
            is_reranker_available,
            rerank,
        )
    except ImportError as e:
        print(f"Error importing reranker module: {e}")
        print("Make sure FlagEmbedding is installed: uv sync")
        return

    print(f"1. Checking reranker availability (model: {settings.reranker_model})...")
    available = is_reranker_available()
    if not available:
        print("   Reranker not available. Check FlagEmbedding installation.")
        return
    print("   ✓ Reranker available")

    print("\n2. Testing reranking...")
    query = "Как освободить место на диске в Docker?"
    candidates = [
        ("mem1", "Docker использует слои для хранения образов", 0.8),
        ("mem2", "Команда docker system prune удаляет неиспользуемые данные", 0.75),
        ("mem3", "Контейнеры изолированы друг от друга", 0.7),
        ("mem4", "Для очистки диска используйте docker system prune -a", 0.65),
        ("mem5", "Python — популярный язык программирования", 0.6),
    ]

    print(f"   Query: {query}")
    print("\n   Original ranking:")
    for id_, content, score in candidates:
        print(f"   {id_}: {score:.2f} - {content[:50]}...")

    results = rerank(query, candidates, top_k=5)
    print("\n   After reranking:")
    for r in results:
        print(f"   {r.id}: {r.rerank_score:.3f} (was {r.original_score:.2f}, rank {r.original_rank})")
        print(f"        {r.content[:50]}...")

    print("\n3. Testing reranker caching...")
    import time
    clear_reranker_cache()

    start = time.time()
    _ = get_reranker()
    first_load = time.time() - start

    start = time.time()
    _ = get_reranker()
    cached_load = time.time() - start

    print(f"   First load:  {first_load:.2f}s")
    print(f"   Cached load: {cached_load:.6f}s")

    print("\n✓ Reranker tests complete!")


def test_actr() -> None:
    """Test ACT-R activation calculations."""
    print("\n=== Testing ACT-R Activation ===\n")

    from engram.learning.forgetting import (
        compute_base_level_activation,
        compute_retrieval_probability,
        determine_memory_status,
    )

    print("1. Testing base-level activation with different access patterns:")

    now = datetime.utcnow()

    # Simulate different access patterns
    patterns = [
        ("Recent single access", [now - timedelta(hours=1)]),
        ("Recent frequent access", [now - timedelta(hours=i) for i in range(1, 6)]),
        ("Old single access", [now - timedelta(days=30)]),
        ("Old frequent access", [now - timedelta(days=i) for i in range(1, 31)]),
        ("Mixed pattern", [
            now - timedelta(hours=1),
            now - timedelta(days=7),
            now - timedelta(days=30),
        ]),
    ]

    print(f"\n   Using decay_d={settings.actr_decay_d}")
    for name, access_times in patterns:
        activation = compute_base_level_activation(access_times, now)
        prob = compute_retrieval_probability(activation)
        status = determine_memory_status(activation)
        print(f"\n   {name}:")
        print(f"      Accesses: {len(access_times)}")
        print(f"      Base-level activation: {activation:.3f}")
        print(f"      Retrieval probability: {prob:.3f}")
        print(f"      Status: {status}")

    print("\n2. Testing status thresholds:")
    print(f"   Deprioritize threshold: {settings.forgetting_deprioritize_threshold}")
    print(f"   Archive threshold: {settings.forgetting_archive_threshold}")

    test_activations = [2.0, 0.0, -0.5, -1.5, -2.0, -3.0, -5.0]
    for activation in test_activations:
        status = determine_memory_status(activation)
        prob = compute_retrieval_probability(activation)
        print(f"   B={activation:+.1f} -> status={status:15s} prob={prob:.3f}")

    print("\n3. Testing decay over time:")
    # Single access, measure activation decay
    access_time = now - timedelta(hours=1)
    times = [1, 6, 24, 24*7, 24*30]  # hours
    print("\n   Time since access -> Activation -> Probability")
    for hours in times:
        test_now = access_time + timedelta(hours=hours)
        activation = compute_base_level_activation([access_time], test_now)
        prob = compute_retrieval_probability(activation)
        time_str = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"   {time_str:6s} -> B={activation:+.3f} -> P={prob:.3f}")

    print("\n✓ ACT-R tests complete!")


async def run_benchmark() -> None:
    """Run retrieval benchmark with sample queries."""
    print("\n=== Running Retrieval Benchmark ===\n")

    print("Note: This requires a running Neo4j database with ingested data.")
    print("Start Neo4j: docker start engram-neo4j")
    print("Start API:   uv run python -m engram.api.main\n")

    try:
        from engram.storage.neo4j_client import Neo4jClient
    except ImportError as e:
        print(f"Error importing Neo4j client: {e}")
        return

    try:
        db = Neo4jClient()
        await db.verify_connection()
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"✗ Could not connect to Neo4j: {e}")
        return

    # Check if we have data
    result = await db.execute_query(
        "MATCH (m:SemanticMemory) RETURN count(m) as count"
    )
    memory_count = result[0]["count"] if result else 0

    result = await db.execute_query(
        "MATCH (c:Concept) RETURN count(c) as count"
    )
    concept_count = result[0]["count"] if result else 0

    print(f"   Memories: {memory_count}")
    print(f"   Concepts: {concept_count}")

    if memory_count == 0:
        print("\nNo data in database. Run ingestion first:")
        print("   uv run python scripts/run_ingestion.py")
        await db.close()
        return

    print("\nBenchmark results would be displayed here...")
    print("(Full benchmark implementation requires test query set)")

    await db.close()
    print("\n✓ Benchmark complete!")


def show_config() -> None:
    """Show current v3 configuration."""
    print("\n=== Engram v3 Configuration ===\n")

    sections = [
        ("RRF Fusion", [
            ("rrf_k", settings.rrf_k),
        ]),
        ("Reranker", [
            ("reranker_enabled", settings.reranker_enabled),
            ("reranker_model", settings.reranker_model),
            ("reranker_candidates", settings.reranker_candidates),
        ]),
        ("BM25", [
            ("bm25_lemmatize", settings.bm25_lemmatize),
            ("bm25_remove_stopwords", settings.bm25_remove_stopwords),
        ]),
        ("Retrieval", [
            ("retrieval_top_k", settings.retrieval_top_k),
            ("retrieval_bm25_k", settings.retrieval_bm25_k),
            ("retrieval_vector_k", settings.retrieval_vector_k),
        ]),
        ("Embeddings", [
            ("embedding_model", settings.embedding_model),
            ("embedding_dimensions", settings.embedding_dimensions),
            ("embedding_query_prefix", settings.embedding_query_prefix or "(none)"),
        ]),
        ("SM-2 Spaced Repetition", [
            ("sm2_initial_ef", settings.sm2_initial_ef),
            ("sm2_ef_min", settings.sm2_ef_min),
            ("sm2_ef_max", settings.sm2_ef_max),
        ]),
        ("ACT-R Activation", [
            ("actr_decay_d", settings.actr_decay_d),
            ("actr_threshold_tau", settings.actr_threshold_tau),
        ]),
        ("Forgetting", [
            ("forgetting_deprioritize_threshold", settings.forgetting_deprioritize_threshold),
            ("forgetting_archive_threshold", settings.forgetting_archive_threshold),
        ]),
        ("Contradiction", [
            ("contradiction_auto_resolve_gap", settings.contradiction_auto_resolve_gap),
        ]),
    ]

    for section_name, params in sections:
        print(f"{section_name}:")
        for name, value in params:
            print(f"  {name}: {value}")
        print()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Engram v3 Calibration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config               Show current configuration
  %(prog)s test-russian         Test Russian NLP preprocessing
  %(prog)s test-reranker        Test BGE reranker
  %(prog)s test-actr            Test ACT-R activation calculations
  %(prog)s benchmark            Run retrieval benchmark
        """,
    )

    parser.add_argument(
        "command",
        choices=["config", "test-russian", "test-reranker", "test-actr", "benchmark"],
        help="Calibration command to run",
    )

    args = parser.parse_args()

    if args.command == "config":
        show_config()
    elif args.command == "test-russian":
        test_russian_preprocessing()
    elif args.command == "test-reranker":
        test_reranker()
    elif args.command == "test-actr":
        test_actr()
    elif args.command == "benchmark":
        asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
