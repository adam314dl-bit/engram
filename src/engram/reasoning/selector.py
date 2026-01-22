"""LLM-based memory selection for two-phase retrieval.

This module implements the second phase of retrieval where an LLM
selects which memories from a large candidate pool are actually
relevant to the query.
"""

import logging
import re
from dataclasses import dataclass

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


# Russian prompt for memory selection
MEMORY_SELECTION_PROMPT = """Ты — система отбора релевантных знаний.

Вопрос пользователя: {query}

Вот список знаний из базы данных. Каждое знание имеет ID и краткое описание.
Выбери ID тех знаний, которые содержат информацию, нужную для ответа на вопрос.

Знания:
{memories_list}

Инструкции:
1. Прочитай вопрос пользователя внимательно
2. Определи, какая информация нужна для полного ответа
3. Выбери только те знания, которые ДЕЙСТВИТЕЛЬНО релевантны
4. Не выбирай знания с похожими словами, если они не помогут ответить на вопрос
5. Лучше выбрать меньше, но точнее

Выведи ТОЛЬКО список ID через запятую, без объяснений:
SELECTED: id1, id2, id3

Ответ:"""


@dataclass
class SelectionResult:
    """Result of LLM memory selection."""

    selected_ids: list[str]
    all_candidate_ids: list[str]
    selection_ratio: float  # selected / total


class MemorySelector:
    """
    LLM-based selection of relevant memories.

    Takes a large pool of candidate memories and uses LLM to select
    which ones actually contain information needed for the query.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_candidates_per_batch: int = 50,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.max_candidates_per_batch = max_candidates_per_batch

    async def select(
        self,
        query: str,
        candidates: list[ScoredMemory],
        max_candidates: int = 100,
    ) -> SelectionResult:
        """
        Select relevant memories from candidates using LLM.

        Args:
            query: User query text
            candidates: List of candidate memories with scores
            max_candidates: Maximum candidates to consider

        Returns:
            SelectionResult with selected memory IDs
        """
        if not candidates:
            return SelectionResult(
                selected_ids=[],
                all_candidate_ids=[],
                selection_ratio=0.0,
            )

        # Limit candidates
        candidates = candidates[:max_candidates]
        all_ids = [sm.memory.id for sm in candidates]

        # Format memories for LLM
        memories_list = self._format_memories(candidates)

        # Build prompt
        prompt = MEMORY_SELECTION_PROMPT.format(
            query=query,
            memories_list=memories_list,
        )

        # Call LLM
        try:
            response = await self.llm.generate(
                prompt,
                system_prompt="Ты отбираешь релевантные знания. Выводи только ID.",
                temperature=0.1,  # Low temperature for deterministic selection
                max_tokens=500,
            )
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, returning top candidates")
            # Fallback: return top 10 by score
            top_ids = [sm.memory.id for sm in candidates[:10]]
            return SelectionResult(
                selected_ids=top_ids,
                all_candidate_ids=all_ids,
                selection_ratio=len(top_ids) / len(all_ids) if all_ids else 0.0,
            )

        # Parse selected IDs
        selected_ids = self._parse_selected_ids(response, all_ids)

        logger.info(
            f"LLM selected {len(selected_ids)}/{len(candidates)} memories "
            f"({len(selected_ids)/len(candidates)*100:.1f}%)"
        )

        return SelectionResult(
            selected_ids=selected_ids,
            all_candidate_ids=all_ids,
            selection_ratio=len(selected_ids) / len(all_ids) if all_ids else 0.0,
        )

    def _format_memories(self, candidates: list[ScoredMemory]) -> str:
        """Format memories for LLM prompt."""
        lines = []
        for sm in candidates:
            memory = sm.memory
            # Use content which is now "Summary: ... | Keywords: ..."
            content = memory.content
            # Truncate if too long
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"[{memory.id}] {content}")
        return "\n".join(lines)

    def _parse_selected_ids(
        self,
        response: str,
        valid_ids: list[str],
    ) -> list[str]:
        """Parse selected memory IDs from LLM response."""
        valid_id_set = set(valid_ids)
        selected: list[str] = []

        # Look for SELECTED: line
        match = re.search(r"SELECTED:\s*(.+)", response, re.IGNORECASE)
        if match:
            ids_str = match.group(1)
        else:
            # Fallback: look for mem_ patterns anywhere in response
            ids_str = response

        # Extract all mem_xxx patterns (alphanumeric IDs)
        id_pattern = re.compile(r"mem_[a-zA-Z0-9]+")
        found_ids = id_pattern.findall(ids_str)

        # Validate and deduplicate
        seen = set()
        for mem_id in found_ids:
            if mem_id in valid_id_set and mem_id not in seen:
                selected.append(mem_id)
                seen.add(mem_id)

        return selected


async def select_memories(
    query: str,
    candidates: list[ScoredMemory],
    llm_client: LLMClient | None = None,
) -> SelectionResult:
    """Convenience function for memory selection."""
    selector = MemorySelector(llm_client=llm_client)
    return await selector.select(query, candidates)
