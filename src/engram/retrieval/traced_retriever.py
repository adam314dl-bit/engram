"""Traced retriever wrapper for retrieval observability.

v4.5: Wraps RetrievalPipeline to add detailed tracing of chunks
through every pipeline stage.
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator

from engram.config import settings
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.hybrid_search import ScoredMemory
from engram.retrieval.observability import (
    ChunkTrace,
    RetrievalTrace,
    StepTrace,
    create_trace,
)
from engram.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class TracedRetriever:
    """
    Wrapper that adds tracing to RetrievalPipeline.

    Tracks every chunk through each pipeline stage, recording:
    - Where chunks first appear
    - Scores at each stage
    - Where chunks get dropped
    - Final ranking

    Usage:
        traced = TracedRetriever(db)
        result, trace = await traced.retrieve_with_trace(query)
        print(trace.summary())
    """

    def __init__(
        self,
        db: Neo4jClient,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.db = db
        self.embeddings = embedding_service or get_embedding_service()
        self.pipeline = RetrievalPipeline(
            db=db,
            embedding_service=self.embeddings,
        )

    @contextmanager
    def _trace_step(
        self,
        trace: RetrievalTrace,
        step_name: str,
    ) -> Generator[StepTrace, None, None]:
        """Context manager for timing a pipeline step."""
        step = StepTrace(
            step_name=step_name,
            duration_ms=0,
            input_count=0,
            output_count=0,
        )
        start = time.perf_counter()
        try:
            yield step
        finally:
            step.duration_ms = (time.perf_counter() - start) * 1000
            trace.steps.append(step)

    def _record_chunks(
        self,
        trace: RetrievalTrace,
        step: StepTrace,
        scored_memories: list[ScoredMemory],
        is_final: bool = False,
    ) -> None:
        """Record chunk scores and ranks for a step."""
        step.output_count = len(scored_memories)

        for rank, sm in enumerate(scored_memories, 1):
            memory_id = sm.memory.id

            # Get or create chunk trace
            if memory_id not in trace.chunk_traces:
                full_content = sm.memory.content or ""
                content_preview = full_content[:100] if full_content else ""
                trace.chunk_traces[memory_id] = ChunkTrace(
                    memory_id=memory_id,
                    content_preview=content_preview,
                    full_content=full_content,
                )

            chunk = trace.chunk_traces[memory_id]

            # Record score and rank for this step
            chunk.stage_scores[step.step_name] = sm.score
            chunk.stage_ranks[step.step_name] = rank
            step.chunk_scores[memory_id] = sm.score

            # Record sources
            for source in sm.sources:
                if source not in chunk.sources:
                    chunk.sources.append(source)

            # Mark as included in final result
            if is_final:
                chunk.included = True
                chunk.final_rank = rank

    async def retrieve_with_trace(
        self,
        query: str,
        top_k_memories: int | None = None,
        top_k_episodes: int | None = None,
        include_episodes: bool = True,
    ) -> tuple[RetrievalResult, RetrievalTrace]:
        """
        Execute retrieval with full tracing.

        Args:
            query: User query text
            top_k_memories: Number of memories to return
            top_k_episodes: Number of episodes to return
            include_episodes: Whether to include similar episodes

        Returns:
            Tuple of (RetrievalResult, RetrievalTrace)
        """
        trace = create_trace(query)
        start_time = time.perf_counter()

        # We'll instrument the pipeline by calling components directly
        # and recording results at each step

        top_k_memories = top_k_memories or settings.retrieval_top_k
        top_k_episodes = top_k_episodes or 3

        # Check retrieval mode
        use_vector = settings.retrieval_mode != "bm25_graph"

        # Step 1: Embed query
        query_embedding: list[float] = []
        with self._trace_step(trace, "query_embedding") as step:
            if use_vector:
                query_embedding = await self.embeddings.embed(query)
                step.metadata["embedding_dim"] = len(query_embedding)
            else:
                step.metadata["skipped"] = "bm25_graph mode"

        # Step 2: Extract concepts
        with self._trace_step(trace, "concept_extraction") as step:
            concept_result = await self.pipeline.concept_extractor.extract(query)
            trace.extracted_concepts = [c.name for c in concept_result.concepts]
            step.output_count = len(concept_result.concepts)
            step.metadata["concepts"] = trace.extracted_concepts

        # Step 3: Match concepts in graph
        seed_concept_ids: list[str] = []
        with self._trace_step(trace, "concept_matching") as step:
            step.input_count = len(concept_result.concepts)
            for concept in concept_result.concepts:
                existing = await self.db.get_concept_by_name(concept.name)
                if existing:
                    seed_concept_ids.append(existing.id)

            # Fallback search if no matches
            if not seed_concept_ids:
                if use_vector and query_embedding:
                    vector_concepts = await self.db.vector_search_concepts(
                        embedding=query_embedding, k=5
                    )
                    for c, score in vector_concepts:
                        if score > 0.3:
                            seed_concept_ids.append(c.id)
                else:
                    bm25_concepts = await self.db.fulltext_search_concepts(
                        query_text=query, k=5
                    )
                    for c, score in bm25_concepts:
                        if score > 0.5:
                            seed_concept_ids.append(c.id)

            step.output_count = len(seed_concept_ids)
            step.metadata["matched_ids"] = seed_concept_ids

        # Step 4: Spreading activation
        activated_concepts: dict[str, float] = {}
        with self._trace_step(trace, "spreading_activation") as step:
            step.input_count = len(seed_concept_ids)
            if seed_concept_ids:
                activation_result = await self.pipeline.spreading.activate(
                    seed_concept_ids,
                    query_embedding if query_embedding else [],
                )
                activated_concepts = activation_result.activations
                step.output_count = len(activated_concepts)
                step.metadata["max_activation"] = max(activated_concepts.values()) if activated_concepts else 0

        # Step 5: Get graph memories
        graph_memories = []
        graph_memory_scores: dict[str, float] = {}
        with self._trace_step(trace, "graph_retrieval") as step:
            if activated_concepts:
                active_ids = [
                    cid for cid, act in activated_concepts.items()
                    if act >= settings.activation_threshold
                ]
                step.input_count = len(active_ids)

                if active_ids:
                    graph_memories = await self.db.get_memories_for_concepts(
                        concept_ids=active_ids,
                        limit=top_k_memories * 2,
                    )
                    for memory in graph_memories:
                        score = sum(
                            activated_concepts.get(cid, 0)
                            for cid in memory.concept_ids
                        )
                        graph_memory_scores[memory.id] = score

            step.output_count = len(graph_memories)

            # Record graph memories
            for rank, memory in enumerate(graph_memories, 1):
                if memory.id not in trace.chunk_traces:
                    full_content = memory.content or ""
                    trace.chunk_traces[memory.id] = ChunkTrace(
                        memory_id=memory.id,
                        content_preview=full_content[:100] if full_content else "",
                        full_content=full_content,
                    )
                chunk = trace.chunk_traces[memory.id]
                chunk.stage_scores["graph_retrieval"] = graph_memory_scores.get(memory.id, 0)
                chunk.stage_ranks["graph_retrieval"] = rank
                step.chunk_scores[memory.id] = graph_memory_scores.get(memory.id, 0)
                if "G" not in chunk.sources:
                    chunk.sources.append("G")

        # Step 6: Path-based retrieval
        path_memories: list[ScoredMemory] = []
        path_memory_scores: dict[str, float] = {}
        with self._trace_step(trace, "path_retrieval") as step:
            step.input_count = len(seed_concept_ids)
            if len(seed_concept_ids) >= 2:
                path_result = await self.pipeline.path_retriever.retrieve(seed_concept_ids)
                path_memories = path_result.all_memories

                for sm in path_memories:
                    path_memory_scores[sm.memory.id] = sm.score

                step.output_count = len(path_memories)
                step.metadata["shared_memories"] = len(path_result.shared_memories)
                step.metadata["path_memories"] = len(path_result.path_memories)
                step.metadata["paths_found"] = len(path_result.paths)
                step.metadata["bridge_concepts"] = len(path_result.bridge_concepts)

                # Record path memories
                for rank, sm in enumerate(path_memories, 1):
                    if sm.memory.id not in trace.chunk_traces:
                        full_content = sm.memory.content or ""
                        trace.chunk_traces[sm.memory.id] = ChunkTrace(
                            memory_id=sm.memory.id,
                            content_preview=full_content[:100] if full_content else "",
                            full_content=full_content,
                        )
                    chunk = trace.chunk_traces[sm.memory.id]
                    chunk.stage_scores["path_retrieval"] = sm.score
                    chunk.stage_ranks["path_retrieval"] = rank
                    step.chunk_scores[sm.memory.id] = sm.score
                    if "P" not in chunk.sources:
                        chunk.sources.append("P")
            else:
                step.metadata["skipped"] = "need >= 2 concepts"

        # Step 7: BM25 search
        with self._trace_step(trace, "bm25_search") as step:
            bm25_results = await self.db.fulltext_search_memories(
                query_text=query, k=settings.retrieval_bm25_k
            )
            step.output_count = len(bm25_results)

            for rank, (memory, score) in enumerate(bm25_results, 1):
                if memory.id not in trace.chunk_traces:
                    full_content = memory.content or ""
                    trace.chunk_traces[memory.id] = ChunkTrace(
                        memory_id=memory.id,
                        content_preview=full_content[:100] if full_content else "",
                        full_content=full_content,
                    )
                chunk = trace.chunk_traces[memory.id]
                chunk.stage_scores["bm25_search"] = score
                chunk.stage_ranks["bm25_search"] = rank
                step.chunk_scores[memory.id] = score
                if "B" not in chunk.sources:
                    chunk.sources.append("B")

        # Step 8: Vector search (if enabled)
        if use_vector and query_embedding:
            with self._trace_step(trace, "vector_search") as step:
                vector_results = await self.db.vector_search_memories(
                    embedding=query_embedding, k=settings.retrieval_vector_k
                )
                step.output_count = len(vector_results)

                for rank, (memory, score) in enumerate(vector_results, 1):
                    if memory.id not in trace.chunk_traces:
                        full_content = memory.content or ""
                        trace.chunk_traces[memory.id] = ChunkTrace(
                            memory_id=memory.id,
                            content_preview=full_content[:100] if full_content else "",
                            full_content=full_content,
                        )
                    chunk = trace.chunk_traces[memory.id]
                    chunk.stage_scores["vector_search"] = score
                    chunk.stage_ranks["vector_search"] = rank
                    step.chunk_scores[memory.id] = score
                    if "V" not in chunk.sources:
                        chunk.sources.append("V")

        # Step 9: Hybrid search (full pipeline with fusion)
        with self._trace_step(trace, "hybrid_fusion") as step:
            # Count unique chunks before fusion
            all_chunk_ids = set(trace.chunk_traces.keys())
            step.input_count = len(all_chunk_ids)

            # Call hybrid search
            scored_memories = await self.pipeline.hybrid.search_memories(
                query=query,
                query_embedding=query_embedding if query_embedding else None,
                graph_memories=graph_memories,
                graph_memory_scores=graph_memory_scores,
                path_memories=[sm.memory for sm in path_memories],
                path_memory_scores=path_memory_scores,
                use_dynamic_k=False,
            )

            step.output_count = len(scored_memories)
            self._record_chunks(trace, step, scored_memories, is_final=True)

        # Calculate total duration
        trace.total_duration_ms = (time.perf_counter() - start_time) * 1000

        # Save trace if configured
        if settings.observability_save_traces:
            filepath = trace.save()
            logger.debug(f"Saved trace to {filepath}")

        # Now get the full result using the standard pipeline
        # (we've already computed everything, but this gives us the full RetrievalResult)
        result = await self.pipeline.retrieve(
            query=query,
            top_k_memories=top_k_memories,
            top_k_episodes=top_k_episodes,
            include_episodes=include_episodes,
        )

        return result, trace

    def find_chunks_by_text(
        self,
        trace: RetrievalTrace,
        search_text: str,
    ) -> list[ChunkTrace]:
        """Find chunks containing specific text.

        Args:
            trace: The retrieval trace
            search_text: Text to search for (case-insensitive)

        Returns:
            List of ChunkTrace objects containing the text
        """
        search_lower = search_text.lower()
        return [
            chunk for chunk in trace.chunk_traces.values()
            if search_lower in chunk.full_content.lower()
        ]
