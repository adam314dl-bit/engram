"""Evaluation runner for retrieval quality testing.

v5: Runs retrieval against a golden test set and calculates metrics.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from engram.config import settings
from engram.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    evaluate_retrieval,
    format_metrics,
)
from engram.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class GoldenQuery:
    """A golden test query with expected results."""

    query: str
    relevant_chunk_ids: list[str]
    notes: str = ""
    tags: list[str] | None = None


@dataclass
class QueryResult:
    """Result of evaluating a single query."""

    query: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    metrics: RetrievalMetrics
    notes: str = ""


class EvaluationRunner:
    """
    Runner for retrieval evaluation against golden queries.

    Usage:
        runner = EvaluationRunner(db)
        results = await runner.run_evaluation("tests/golden_queries.json")
        print(runner.format_results(results))
    """

    def __init__(
        self,
        db: Neo4jClient,
        pipeline: RetrievalPipeline | None = None,
        k_values: list[int] | None = None,
    ) -> None:
        """
        Initialize evaluation runner.

        Args:
            db: Neo4j client
            pipeline: Retrieval pipeline (created if not provided)
            k_values: K values for @K metrics
        """
        self.db = db
        self.pipeline = pipeline or RetrievalPipeline(db=db)
        self.k_values = k_values or [1, 3, 5, 10, 20, 50, 100]

    def load_golden_queries(self, path: str | Path) -> list[GoldenQuery]:
        """
        Load golden queries from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of GoldenQuery objects
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append(GoldenQuery(
                query=item["query"],
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                notes=item.get("notes", ""),
                tags=item.get("tags"),
            ))

        return queries

    async def evaluate_query(
        self,
        golden: GoldenQuery,
    ) -> QueryResult:
        """
        Evaluate a single golden query.

        Args:
            golden: Golden query with expected results

        Returns:
            QueryResult with retrieved results and metrics
        """
        # Run retrieval
        result = await self.pipeline.retrieve(
            query=golden.query,
            top_k_memories=max(self.k_values),
        )

        # Extract retrieved IDs in order
        retrieved_ids = [sm.memory.id for sm in result.memories]
        relevant_ids = set(golden.relevant_chunk_ids)

        # Calculate metrics
        metrics = evaluate_retrieval(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k_values=self.k_values,
        )

        return QueryResult(
            query=golden.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            metrics=metrics,
            notes=golden.notes,
        )

    async def run_evaluation(
        self,
        golden_path: str | Path,
        max_queries: int | None = None,
    ) -> list[QueryResult]:
        """
        Run evaluation on all golden queries.

        Args:
            golden_path: Path to golden queries JSON
            max_queries: Maximum queries to evaluate (for testing)

        Returns:
            List of QueryResult for each query
        """
        golden_queries = self.load_golden_queries(golden_path)

        if max_queries:
            golden_queries = golden_queries[:max_queries]

        logger.info(f"Evaluating {len(golden_queries)} queries...")

        results = []
        for i, golden in enumerate(golden_queries, 1):
            try:
                result = await self.evaluate_query(golden)
                results.append(result)
                logger.debug(
                    f"[{i}/{len(golden_queries)}] {golden.query[:50]}... "
                    f"MRR={result.metrics.mrr:.4f}"
                )
            except Exception as e:
                logger.error(f"Error evaluating query '{golden.query}': {e}")

        return results

    def aggregate_results(self, results: list[QueryResult]) -> RetrievalMetrics:
        """Aggregate metrics across all results."""
        all_metrics = [r.metrics for r in results]
        return aggregate_metrics(all_metrics)

    def format_results(
        self,
        results: list[QueryResult],
        show_per_query: bool = False,
    ) -> str:
        """
        Format evaluation results for display.

        Args:
            results: List of QueryResult
            show_per_query: Show individual query results

        Returns:
            Formatted string
        """
        lines = []

        # Aggregate metrics
        aggregated = self.aggregate_results(results)
        lines.append(format_metrics(aggregated))

        # Per-query results if requested
        if show_per_query:
            lines.append("")
            lines.append("Per-Query Results:")
            lines.append("-" * 60)

            for i, r in enumerate(results, 1):
                lines.append(f"\n{i}. {r.query[:60]}{'...' if len(r.query) > 60 else ''}")
                lines.append(f"   MRR: {r.metrics.mrr:.4f}")
                lines.append(f"   Recall@10: {r.metrics.recall_at_k.get(10, 0):.4f}")
                if r.notes:
                    lines.append(f"   Notes: {r.notes}")

        return "\n".join(lines)

    def save_results(
        self,
        results: list[QueryResult],
        output_path: str | Path,
    ) -> None:
        """
        Save evaluation results to JSON.

        Args:
            results: List of QueryResult
            output_path: Output file path
        """
        output_path = Path(output_path)

        # Convert to serializable format
        output = {
            "summary": asdict(self.aggregate_results(results)),
            "queries": [
                {
                    "query": r.query,
                    "notes": r.notes,
                    "retrieved_ids": r.retrieved_ids[:20],  # Limit for readability
                    "relevant_ids": list(r.relevant_ids),
                    "metrics": {
                        "mrr": r.metrics.mrr,
                        "recall_at_10": r.metrics.recall_at_k.get(10, 0),
                        "recall_at_50": r.metrics.recall_at_k.get(50, 0),
                        "ndcg_at_10": r.metrics.ndcg_at_k.get(10, 0),
                    },
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation results to {output_path}")


async def main():
    """CLI entry point for evaluation runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("golden_path", help="Path to golden queries JSON")
    parser.add_argument("-n", "--max-queries", type=int, help="Max queries to evaluate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-query results")
    parser.add_argument("-o", "--output", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize database
    db = Neo4jClient()
    await db.connect()

    try:
        runner = EvaluationRunner(db)
        results = await runner.run_evaluation(
            args.golden_path,
            max_queries=args.max_queries,
        )

        print("\n" + runner.format_results(results, show_per_query=args.verbose))

        if args.output:
            runner.save_results(results, args.output)

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
