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


# Russian prompt for memory selection with confidence scoring
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
6. Оцени, достаточно ли выбранной информации для полного ответа (0-10):
   - 0-4: Информации недостаточно или она неполная
   - 5-7: Информации достаточно для базового ответа
   - 8-10: Информации достаточно для исчерпывающего ответа

Выведи ТОЛЬКО в формате:
SELECTED: id1, id2, id3
CONFIDENCE: 7

Ответ:"""

# Prompt for selecting from raw document chunks (Phase 2)
CHUNK_SELECTION_PROMPT = """Ты — система отбора релевантных фрагментов документов.

Вопрос пользователя: {query}

Вот фрагменты из исходных документов. Каждый фрагмент имеет ID и текст.
Выбери ID фрагментов, которые содержат информацию для ответа на вопрос.

Фрагменты:
{chunks_list}

Инструкции:
1. Прочитай вопрос пользователя внимательно
2. Выбери фрагменты с конкретной информацией для ответа
3. Не выбирай фрагменты только по ключевым словам без релевантного контекста
4. Оцени качество найденной информации (0-10)

Выведи ТОЛЬКО в формате:
SELECTED: id1, id2, id3
CONFIDENCE: 7

Ответ:"""


@dataclass
class SelectionResult:
    """Result of LLM memory selection."""

    selected_ids: list[str]
    all_candidate_ids: list[str]
    selection_ratio: float  # selected / total
    confidence: float  # 0-10 scale from LLM assessment


@dataclass
class ChunkSelectionResult:
    """Result of LLM chunk selection (Phase 2)."""

    selected_ids: list[str]
    all_candidate_ids: list[str]
    selection_ratio: float
    confidence: float
    # Map chunk_id -> (text, doc_id) for synthesis
    chunks: dict[str, tuple[str, str]]


class MemorySelector:
    """
    LLM-based selection of relevant memories.

    Takes a large pool of candidate memories and uses LLM to select
    which ones actually contain information needed for the query.

    Note: Small LLMs (like qwen3:8b) struggle with long prompts,
    so we process candidates in batches.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        batch_size: int = 8,  # Small batches for local LLMs
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.batch_size = batch_size

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

        # Process in batches (small LLMs struggle with long prompts)
        selected_ids: list[str] = []
        confidences: list[float] = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_ids = [sm.memory.id for sm in batch]

            # Format memories for LLM
            memories_list = self._format_memories(batch)

            # Build prompt
            prompt = MEMORY_SELECTION_PROMPT.format(
                query=query,
                memories_list=memories_list,
            )

            logger.debug(f"Selection batch {i//self.batch_size + 1}: {len(batch)} memories, {len(prompt)} chars")

            # Call LLM
            try:
                response = await self.llm.generate(
                    prompt,
                    system_prompt="Ты отбираешь релевантные знания. Выводи только ID.",
                    temperature=0.1,  # Low temperature for deterministic selection
                    max_tokens=500,
                )
                logger.debug(f"LLM batch response: {response[:200]}")
            except Exception as e:
                logger.warning(f"LLM selection failed for batch: {e}")
                continue

            # Parse selected IDs and confidence from this batch
            batch_selected = self._parse_selected_ids(response, batch_ids)
            selected_ids.extend(batch_selected)
            batch_confidence = self._parse_confidence(response)
            confidences.append(batch_confidence)

        # Aggregate confidence (average across batches)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 5.0

        # Fallback if nothing selected
        if not selected_ids:
            logger.warning("LLM selected nothing, falling back to top candidates by score")
            selected_ids = [sm.memory.id for sm in candidates[:5]]
            avg_confidence = 3.0  # Low confidence for fallback

        logger.info(
            f"LLM selected {len(selected_ids)}/{len(candidates)} memories "
            f"({len(selected_ids)/len(candidates)*100:.1f}%), confidence: {avg_confidence:.1f}"
        )

        return SelectionResult(
            selected_ids=selected_ids,
            all_candidate_ids=all_ids,
            selection_ratio=len(selected_ids) / len(all_ids) if all_ids else 0.0,
            confidence=avg_confidence,
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

        # Extract all UUID patterns (with or without mem_ prefix)
        # UUID format: 8-4-4-4-12 hex digits
        uuid_pattern = re.compile(r"(?:mem_)?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", re.IGNORECASE)
        found_ids = uuid_pattern.findall(ids_str)

        # Validate and deduplicate
        seen = set()
        for mem_id in found_ids:
            if mem_id in valid_id_set and mem_id not in seen:
                selected.append(mem_id)
                seen.add(mem_id)

        return selected

    def _parse_confidence(self, response: str) -> float:
        """Parse confidence score from LLM response."""
        # Look for CONFIDENCE: line
        match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            try:
                confidence = float(match.group(1))
                # Clamp to 0-10 range
                return max(0.0, min(10.0, confidence))
            except ValueError:
                pass
        # Default to middle confidence if not found
        return 5.0

    async def select_from_chunks(
        self,
        query: str,
        chunks: list[tuple[str, str, str, float]],  # (chunk_id, text, doc_id, score)
        max_candidates: int = 100,
    ) -> ChunkSelectionResult:
        """
        Select relevant chunks from raw document chunks using LLM.

        Args:
            query: User query text
            chunks: List of (chunk_id, text, doc_id, score) tuples
            max_candidates: Maximum candidates to consider

        Returns:
            ChunkSelectionResult with selected chunk IDs and confidence
        """
        if not chunks:
            return ChunkSelectionResult(
                selected_ids=[],
                all_candidate_ids=[],
                selection_ratio=0.0,
                confidence=0.0,
                chunks={},
            )

        # Limit candidates
        chunks = chunks[:max_candidates]
        all_ids = [c[0] for c in chunks]
        chunk_map = {c[0]: (c[1], c[2]) for c in chunks}  # id -> (text, doc_id)

        # Process in batches
        selected_ids: list[str] = []
        confidences: list[float] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_ids = [c[0] for c in batch]

            # Format chunks for LLM
            chunks_list = self._format_chunks(batch)

            prompt = CHUNK_SELECTION_PROMPT.format(
                query=query,
                chunks_list=chunks_list,
            )

            try:
                response = await self.llm.generate(
                    prompt,
                    system_prompt="Ты отбираешь релевантные фрагменты документов. Выводи только ID.",
                    temperature=0.1,
                    max_tokens=500,
                )
            except Exception as e:
                logger.warning(f"LLM chunk selection failed for batch: {e}")
                continue

            # Parse selected chunk IDs
            batch_selected = self._parse_chunk_ids(response, batch_ids)
            selected_ids.extend(batch_selected)
            batch_confidence = self._parse_confidence(response)
            confidences.append(batch_confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 5.0

        # Fallback if nothing selected
        if not selected_ids:
            logger.warning("LLM selected no chunks, falling back to top by score")
            selected_ids = [c[0] for c in chunks[:5]]
            avg_confidence = 3.0

        logger.info(
            f"LLM selected {len(selected_ids)}/{len(chunks)} chunks, "
            f"confidence: {avg_confidence:.1f}"
        )

        return ChunkSelectionResult(
            selected_ids=selected_ids,
            all_candidate_ids=all_ids,
            selection_ratio=len(selected_ids) / len(all_ids) if all_ids else 0.0,
            confidence=avg_confidence,
            chunks={cid: chunk_map[cid] for cid in selected_ids if cid in chunk_map},
        )

    def _format_chunks(
        self,
        chunks: list[tuple[str, str, str, float]],
    ) -> str:
        """Format chunks for LLM prompt."""
        lines = []
        for chunk_id, text, doc_id, score in chunks:
            # Truncate text if too long
            display_text = text[:300] + "..." if len(text) > 300 else text
            lines.append(f"[{chunk_id}] {display_text}")
        return "\n".join(lines)

    def _parse_chunk_ids(
        self,
        response: str,
        valid_ids: list[str],
    ) -> list[str]:
        """Parse selected chunk IDs from LLM response."""
        valid_id_set = set(valid_ids)
        selected: list[str] = []

        # Look for SELECTED: line
        match = re.search(r"SELECTED:\s*(.+)", response, re.IGNORECASE)
        if match:
            ids_str = match.group(1)
        else:
            ids_str = response

        # Extract chunk_ IDs
        chunk_pattern = re.compile(r"chunk_[a-f0-9]{12}", re.IGNORECASE)
        found_ids = chunk_pattern.findall(ids_str)

        # Validate and deduplicate
        seen = set()
        for chunk_id in found_ids:
            if chunk_id in valid_id_set and chunk_id not in seen:
                selected.append(chunk_id)
                seen.add(chunk_id)

        return selected


async def select_memories(
    query: str,
    candidates: list[ScoredMemory],
    llm_client: LLMClient | None = None,
) -> SelectionResult:
    """Convenience function for memory selection."""
    selector = MemorySelector(llm_client=llm_client)
    return await selector.select(query, candidates)
