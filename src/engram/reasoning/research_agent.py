"""Async research mode for long-running complex queries.

Handles complex research tasks by:
1. Planning subtasks
2. Executing subtasks in parallel
3. Checkpointing progress
4. Synthesizing results

Useful for queries like:
- "Research all options for X and compare them"
- "Investigate the history of Y"
- "Compile a comprehensive guide on Z"
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.reasoning.ircot import IRCoTReasoner, IRCoTResult
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.hybrid_search import HybridSearch, ScoredMemory
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class ResearchStatus(str, Enum):
    """Status of research task."""

    PLANNING = "planning"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ResearchSubtask:
    """A subtask in the research plan."""

    id: str
    query: str
    priority: int  # Lower is higher priority
    status: str = "pending"  # pending, running, complete, failed
    result: str | None = None
    memories_used: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ResearchProgress:
    """Progress tracking for research task."""

    task_id: str
    original_query: str
    status: ResearchStatus
    subtasks: list[ResearchSubtask]
    current_subtask_id: str | None = None
    partial_results: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    error: str | None = None

    @property
    def completed_count(self) -> int:
        """Number of completed subtasks."""
        return sum(1 for s in self.subtasks if s.status == "complete")

    @property
    def total_count(self) -> int:
        """Total number of subtasks."""
        return len(self.subtasks)

    @property
    def progress_percent(self) -> float:
        """Progress percentage."""
        if not self.subtasks:
            return 0.0
        return (self.completed_count / self.total_count) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "original_query": self.original_query,
            "status": self.status.value,
            "subtasks": [
                {
                    "id": s.id,
                    "query": s.query,
                    "priority": s.priority,
                    "status": s.status,
                    "has_result": s.result is not None,
                }
                for s in self.subtasks
            ],
            "current_subtask_id": self.current_subtask_id,
            "completed_count": self.completed_count,
            "total_count": self.total_count,
            "progress_percent": self.progress_percent,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "error": self.error,
        }


@dataclass
class ResearchResult:
    """Final result of research task."""

    task_id: str
    query: str
    final_report: str
    subtask_results: list[ResearchSubtask]
    total_memories: list[ScoredMemory]
    duration_seconds: float
    status: ResearchStatus


PLANNING_PROMPT = """Спланируй исследование по запросу пользователя.

Запрос: {query}

Разбей исследование на 3-5 независимых подзадач.
Каждая подзадача должна исследовать конкретный аспект вопроса.

Верни JSON:
{{
  "subtasks": [
    {{"query": "подзадача 1", "priority": 1}},
    {{"query": "подзадача 2", "priority": 2}},
    ...
  ],
  "reasoning": "логика разбиения"
}}

JSON:"""


SYNTHESIS_PROMPT = """Синтезируй итоговый отчёт по результатам исследования.

Исходный запрос: {query}

Результаты подзадач:
{subtask_results}

Создай связный, структурированный отчёт, объединяющий все найденные факты.
Используй заголовки и списки для организации информации.
Отметь если какие-то аспекты не были полностью исследованы.

Отчёт:"""


class ResearchAgent:
    """
    Async research agent for complex queries.

    Supports:
    - Task planning
    - Parallel subtask execution
    - Progress checkpointing
    - Result synthesis
    """

    def __init__(
        self,
        db: Neo4jClient,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
        checkpoint_dir: str | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self.db = db
        self.llm = llm_client or get_llm_client()
        self.embeddings = embedding_service or get_embedding_service()
        self.hybrid_search = HybridSearch(db=db)
        self.ircot = IRCoTReasoner(
            db=db,
            llm_client=self.llm,
            embedding_service=self.embeddings,
        )
        self.checkpoint_dir = Path(
            checkpoint_dir or settings.research_checkpoint_dir
        )
        self.max_concurrent = (
            max_concurrent
            if max_concurrent is not None
            else settings.research_max_concurrent_subtasks
        )

    async def research(
        self,
        query: str,
        task_id: str | None = None,
        resume: bool = False,
    ) -> ResearchResult:
        """
        Execute research task.

        Args:
            query: Research query
            task_id: Optional task ID for tracking
            resume: Whether to resume from checkpoint

        Returns:
            ResearchResult with final report
        """
        task_id = task_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        logger.info(f"Starting research task {task_id}: {query[:50]}...")

        # Try to resume from checkpoint
        progress = None
        if resume:
            progress = self._load_checkpoint(task_id)

        # Plan if not resuming
        if not progress:
            progress = await self._plan_research(task_id, query)
            self._save_checkpoint(progress)

        # Execute subtasks
        all_memories: list[ScoredMemory] = []

        try:
            progress.status = ResearchStatus.RESEARCHING
            self._save_checkpoint(progress)

            # Execute subtasks with concurrency limit
            pending_subtasks = [
                s for s in progress.subtasks if s.status == "pending"
            ]

            # Group by priority and execute in parallel within groups
            for priority in sorted(set(s.priority for s in pending_subtasks)):
                priority_tasks = [
                    s for s in pending_subtasks if s.priority == priority
                ]

                # Execute in batches
                for i in range(0, len(priority_tasks), self.max_concurrent):
                    batch = priority_tasks[i:i + self.max_concurrent]

                    results = await asyncio.gather(
                        *(self._execute_subtask(s, progress) for s in batch),
                        return_exceptions=True,
                    )

                    for subtask, result in zip(batch, results):
                        if isinstance(result, Exception):
                            subtask.status = "failed"
                            subtask.error = str(result)
                            logger.warning(f"Subtask failed: {result}")
                        else:
                            subtask.status = "complete"
                            subtask.result = result.final_answer if result else ""
                            subtask.memories_used = [
                                m.memory.id for m in (result.total_memories if result else [])
                            ]
                            if result:
                                all_memories.extend(result.total_memories)

                    progress.last_update = datetime.utcnow()
                    self._save_checkpoint(progress)

            # Synthesize results
            progress.status = ResearchStatus.SYNTHESIZING
            self._save_checkpoint(progress)

            final_report = await self._synthesize_results(query, progress.subtasks)

            progress.status = ResearchStatus.COMPLETE
            self._save_checkpoint(progress)

        except Exception as e:
            logger.error(f"Research task failed: {e}")
            progress.status = ResearchStatus.FAILED
            progress.error = str(e)
            self._save_checkpoint(progress)
            raise

        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Research complete: {task_id}, "
            f"{progress.completed_count}/{progress.total_count} subtasks, "
            f"{len(all_memories)} memories, {duration:.1f}s"
        )

        return ResearchResult(
            task_id=task_id,
            query=query,
            final_report=final_report,
            subtask_results=progress.subtasks,
            total_memories=all_memories,
            duration_seconds=duration,
            status=progress.status,
        )

    async def _plan_research(
        self,
        task_id: str,
        query: str,
    ) -> ResearchProgress:
        """Plan research by creating subtasks."""
        logger.info(f"Planning research: {query[:50]}...")

        try:
            prompt = PLANNING_PROMPT.format(query=query)
            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024,
                fallback=None,
            )

            subtasks = []
            if result and "subtasks" in result:
                for i, st in enumerate(result["subtasks"][:5]):  # Max 5 subtasks
                    subtasks.append(ResearchSubtask(
                        id=f"{task_id}_sub_{i+1}",
                        query=st.get("query", query),
                        priority=st.get("priority", i + 1),
                    ))

        except Exception as e:
            logger.warning(f"Planning failed: {e}")
            # Fallback: single subtask
            subtasks = [ResearchSubtask(
                id=f"{task_id}_sub_1",
                query=query,
                priority=1,
            )]

        return ResearchProgress(
            task_id=task_id,
            original_query=query,
            status=ResearchStatus.PLANNING,
            subtasks=subtasks,
        )

    async def _execute_subtask(
        self,
        subtask: ResearchSubtask,
        progress: ResearchProgress,
    ) -> IRCoTResult:
        """Execute a single subtask using IRCoT."""
        progress.current_subtask_id = subtask.id
        subtask.status = "running"

        logger.info(f"Executing subtask: {subtask.query[:50]}...")

        result = await self.ircot.reason(subtask.query)

        return result

    async def _synthesize_results(
        self,
        query: str,
        subtasks: list[ResearchSubtask],
    ) -> str:
        """Synthesize final report from subtask results."""
        logger.info("Synthesizing research results...")

        subtask_results = []
        for st in subtasks:
            if st.status == "complete" and st.result:
                subtask_results.append(f"## {st.query}\n{st.result}")
            elif st.status == "failed":
                subtask_results.append(f"## {st.query}\n[Не удалось исследовать: {st.error}]")

        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            subtask_results="\n\n".join(subtask_results),
        )

        report = await self.llm.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=4096,
        )

        return report.strip()

    def _save_checkpoint(self, progress: ResearchProgress) -> None:
        """Save progress to checkpoint file."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = self.checkpoint_dir / f"{progress.task_id}.json"

            with open(checkpoint_file, "w") as f:
                json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, task_id: str) -> ResearchProgress | None:
        """Load progress from checkpoint file."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{task_id}.json"
            if not checkpoint_file.exists():
                return None

            with open(checkpoint_file) as f:
                data = json.load(f)

            subtasks = [
                ResearchSubtask(
                    id=st["id"],
                    query=st["query"],
                    priority=st["priority"],
                    status=st["status"],
                )
                for st in data.get("subtasks", [])
            ]

            return ResearchProgress(
                task_id=data["task_id"],
                original_query=data["original_query"],
                status=ResearchStatus(data["status"]),
                subtasks=subtasks,
                current_subtask_id=data.get("current_subtask_id"),
                start_time=datetime.fromisoformat(data["start_time"]),
                last_update=datetime.fromisoformat(data["last_update"]),
                error=data.get("error"),
            )

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def get_progress(self, task_id: str) -> ResearchProgress | None:
        """Get progress for a task."""
        return self._load_checkpoint(task_id)


async def research(
    db: Neo4jClient,
    query: str,
    task_id: str | None = None,
    llm_client: LLMClient | None = None,
) -> ResearchResult:
    """Convenience function for research."""
    agent = ResearchAgent(db=db, llm_client=llm_client)
    return await agent.research(query, task_id)
