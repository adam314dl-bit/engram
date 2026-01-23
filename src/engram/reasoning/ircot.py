"""IRCoT: Interleaved Retrieval Chain-of-Thought for multi-hop reasoning.

Handles complex queries requiring multiple retrieval passes by interleaving
retrieval and reasoning steps. Useful for queries like:
- "Compare X and Y" (needs info about both)
- "How does A affect B through C?" (causal chains)
- "What are all the prerequisites for X?" (transitive closure)

Based on: "Interleaving Retrieval with Chain-of-Thought Reasoning" (Trivedi et al., 2023)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.retrieval.hybrid_search import HybridSearch, ScoredMemory
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Type of reasoning step."""

    RETRIEVAL = "retrieval"  # Fetched new information
    REASONING = "reasoning"  # Drew conclusions
    DECOMPOSITION = "decomposition"  # Broke down query
    SYNTHESIS = "synthesis"  # Combined information


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_number: int
    step_type: StepType
    query: str  # Sub-query or reasoning prompt
    result: str  # Retrieved content or reasoning output
    memories_used: list[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class IRCoTResult:
    """Complete result of IRCoT reasoning."""

    final_answer: str
    reasoning_chain: list[ReasoningStep]
    total_memories: list[ScoredMemory]
    convergence_reason: str
    paragraph_count: int

    @property
    def step_count(self) -> int:
        """Number of reasoning steps."""
        return len(self.reasoning_chain)

    @property
    def retrieval_count(self) -> int:
        """Number of retrieval steps."""
        return sum(
            1 for s in self.reasoning_chain
            if s.step_type == StepType.RETRIEVAL
        )


DECOMPOSITION_PROMPT = """Разбей сложный вопрос на подвопросы для пошагового ответа.

Вопрос: {query}

Разбей на 2-4 подвопроса, которые нужно последовательно ответить.

Верни каждый подвопрос в формате:
SUBQUERY|текст подвопроса

Пример:
SUBQUERY|Что такое Kubernetes?
SUBQUERY|Что такое Docker Swarm?
SUBQUERY|В чём различия между ними?

Ответ:"""


REASONING_STEP_PROMPT = """Продолжи рассуждение на основе новой информации.

Исходный вопрос: {original_query}

Уже известно:
{known_facts}

Новая информация:
{new_info}

Что можно заключить из этой информации?
Если нужна дополнительная информация, укажи какая.

Рассуждение:"""


SYNTHESIS_PROMPT = """Синтезируй финальный ответ на основе собранной информации.

Вопрос: {query}

Собранные факты:
{facts}

Цепочка рассуждений:
{reasoning_chain}

Дай полный, связный ответ на вопрос, используя все собранные факты.
Не добавляй информацию, которой нет в фактах.

Ответ:"""


CONVERGENCE_CHECK_PROMPT = """Проверь, достаточно ли информации для ответа.

Вопрос: {query}

Собранные факты:
{facts}

Текущие выводы:
{conclusions}

Верни результат в формате:
CONVERGE|готов|недостающая_информация

Где готов: yes или no

Пример:
CONVERGE|yes|нет
CONVERGE|no|Нужна информация о ценах

Ответ:"""


class IRCoTReasoner:
    """
    IRCoT reasoner for multi-hop queries.

    Interleaves retrieval and reasoning steps until:
    1. Sufficient information is collected
    2. Max steps reached
    3. Max paragraphs reached
    4. No new information found
    """

    def __init__(
        self,
        db: Neo4jClient,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
        max_steps: int | None = None,
        max_paragraphs: int | None = None,
    ) -> None:
        self.db = db
        self.llm = llm_client or get_llm_client()
        self.embeddings = embedding_service or get_embedding_service()
        self.hybrid_search = HybridSearch(db=db)
        self.max_steps = (
            max_steps
            if max_steps is not None
            else settings.ircot_max_steps
        )
        self.max_paragraphs = (
            max_paragraphs
            if max_paragraphs is not None
            else settings.ircot_max_paragraphs
        )

    async def reason(
        self,
        query: str,
        initial_memories: list[ScoredMemory] | None = None,
    ) -> IRCoTResult:
        """
        Execute multi-hop reasoning.

        Args:
            query: Original complex query
            initial_memories: Optional pre-retrieved memories

        Returns:
            IRCoTResult with answer and reasoning chain
        """
        logger.info(f"IRCoT reasoning for: {query[:50]}...")

        reasoning_chain: list[ReasoningStep] = []
        all_memories: list[ScoredMemory] = list(initial_memories or [])
        known_facts: list[str] = []
        conclusions: list[str] = []
        used_memory_ids: set[str] = set()

        # Step 1: Decompose query
        sub_queries = await self._decompose_query(query)
        reasoning_chain.append(ReasoningStep(
            step_number=1,
            step_type=StepType.DECOMPOSITION,
            query=query,
            result=f"Decomposed into: {sub_queries}",
        ))

        # Process each sub-query
        current_query = query
        step_num = 2

        for sub_query in sub_queries:
            if step_num > self.max_steps:
                break
            if len(known_facts) >= self.max_paragraphs:
                break

            # Retrieval step
            new_memories = await self._retrieve_for_query(sub_query, used_memory_ids)
            all_memories.extend(new_memories)

            new_facts = []
            for m in new_memories:
                used_memory_ids.add(m.memory.id)
                new_facts.append(m.memory.content[:500])

            known_facts.extend(new_facts)

            reasoning_chain.append(ReasoningStep(
                step_number=step_num,
                step_type=StepType.RETRIEVAL,
                query=sub_query,
                result=f"Retrieved {len(new_memories)} documents",
                memories_used=[m.memory.id for m in new_memories],
            ))
            step_num += 1

            if step_num > self.max_steps:
                break

            # Reasoning step
            if new_facts:
                conclusion = await self._reason_step(
                    original_query=query,
                    known_facts=known_facts,
                    new_info=new_facts,
                )
                conclusions.append(conclusion)

                reasoning_chain.append(ReasoningStep(
                    step_number=step_num,
                    step_type=StepType.REASONING,
                    query=sub_query,
                    result=conclusion,
                ))
                step_num += 1

            # Check convergence
            if await self._check_convergence(query, known_facts, conclusions):
                logger.info(f"IRCoT converged after {step_num - 1} steps")
                break

        # Synthesis step
        final_answer = await self._synthesize(
            query=query,
            facts=known_facts,
            reasoning_chain=conclusions,
        )

        reasoning_chain.append(ReasoningStep(
            step_number=step_num,
            step_type=StepType.SYNTHESIS,
            query=query,
            result=final_answer,
        ))

        convergence_reason = self._determine_convergence_reason(
            step_num, len(known_facts), len(all_memories)
        )

        logger.info(
            f"IRCoT complete: {step_num} steps, {len(all_memories)} memories, "
            f"reason: {convergence_reason}"
        )

        return IRCoTResult(
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            total_memories=all_memories,
            convergence_reason=convergence_reason,
            paragraph_count=len(known_facts),
        )

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose complex query into sub-queries."""
        try:
            prompt = DECOMPOSITION_PROMPT.format(query=query)
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
            )

            sub_queries = self._parse_decomposition_response(response)
            if sub_queries:
                return sub_queries[:4]  # Max 4 sub-queries

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")

        # Fallback: return original query
        return [query]

    def _parse_decomposition_response(self, text: str) -> list[str]:
        """Parse pipe-delimited decomposition response."""
        sub_queries = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("SUBQUERY|"):
                parts = line.split("|", 1)
                if len(parts) >= 2:
                    sub_query = parts[1].strip()
                    if sub_query:
                        sub_queries.append(sub_query)
        return sub_queries

    async def _retrieve_for_query(
        self,
        query: str,
        exclude_ids: set[str],
    ) -> list[ScoredMemory]:
        """Retrieve memories for a sub-query."""
        query_embedding = await self.embeddings.embed(query)

        memories = await self.hybrid_search.search_memories(
            query=query,
            query_embedding=query_embedding,
            use_dynamic_k=True,
        )

        # Filter out already used memories
        new_memories = [
            m for m in memories
            if m.memory.id not in exclude_ids
        ]

        return new_memories[:5]  # Limit per step

    async def _reason_step(
        self,
        original_query: str,
        known_facts: list[str],
        new_info: list[str],
    ) -> str:
        """Perform a reasoning step."""
        prompt = REASONING_STEP_PROMPT.format(
            original_query=original_query,
            known_facts="\n".join(known_facts[-10:]),  # Last 10 facts
            new_info="\n".join(new_info),
        )

        conclusion = await self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=512,
        )

        return conclusion.strip()

    async def _check_convergence(
        self,
        query: str,
        facts: list[str],
        conclusions: list[str],
    ) -> bool:
        """Check if we have enough information to answer."""
        if len(facts) >= self.max_paragraphs:
            return True

        if not facts:
            return False

        try:
            prompt = CONVERGENCE_CHECK_PROMPT.format(
                query=query,
                facts="\n".join(facts[-10:]),
                conclusions="\n".join(conclusions[-3:]) if conclusions else "Нет выводов пока",
            )

            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
            )

            return self._parse_convergence_response(response)

        except Exception as e:
            logger.warning(f"Convergence check failed: {e}")
            return False

    def _parse_convergence_response(self, text: str) -> bool:
        """Parse pipe-delimited convergence response."""
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CONVERGE|"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    ready = parts[1].strip().lower()
                    return ready in ("yes", "да", "true", "1")
        return False

    async def _synthesize(
        self,
        query: str,
        facts: list[str],
        reasoning_chain: list[str],
    ) -> str:
        """Synthesize final answer from collected information."""
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            facts="\n".join(f"- {f[:300]}" for f in facts[:15]),
            reasoning_chain="\n".join(f"- {r[:200]}" for r in reasoning_chain),
        )

        answer = await self.llm.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=2048,
        )

        return answer.strip()

    def _determine_convergence_reason(
        self,
        steps: int,
        paragraphs: int,
        memories: int,
    ) -> str:
        """Determine why reasoning stopped."""
        if steps >= self.max_steps:
            return "max_steps_reached"
        if paragraphs >= self.max_paragraphs:
            return "max_paragraphs_reached"
        if memories == 0:
            return "no_information_found"
        return "convergence_detected"


async def ircot_reason(
    db: Neo4jClient,
    query: str,
    initial_memories: list[ScoredMemory] | None = None,
    llm_client: LLMClient | None = None,
) -> IRCoTResult:
    """Convenience function for IRCoT reasoning."""
    reasoner = IRCoTReasoner(db=db, llm_client=llm_client)
    return await reasoner.reason(query, initial_memories)
