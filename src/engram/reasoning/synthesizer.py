"""Response synthesizer for generating answers using retrieved context.

Handles:
1. Context formatting for LLM
2. Response generation
3. Behavior extraction (metacognitive reuse)
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# Russian month names for date formatting
RUSSIAN_MONTHS = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import EpisodicMemory, SemanticMemory
from engram.retrieval.hybrid_search import ScoredEpisode, ScoredMemory
from engram.retrieval.pipeline import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Behavior:
    """Extracted behavior pattern from response."""

    name: str  # "check_disk_usage", "explain_concept"
    instruction: str  # Reusable pattern description
    domain: str = "general"  # "docker", "kubernetes", etc.


@dataclass
class SynthesisResult:
    """Result of response synthesis."""

    answer: str
    behavior: Behavior
    memories_used: list[str] = field(default_factory=list)
    concepts_activated: list[str] = field(default_factory=list)
    confidence: float = 0.8

    # For episode creation
    query: str = ""
    importance: float = 5.0


def format_memories_context(memories: list[ScoredMemory], max_memories: int = 10) -> str:
    """Format memories for LLM context."""
    if not memories:
        return "Нет релевантных знаний."

    lines = []
    for i, sm in enumerate(memories[:max_memories], 1):
        memory = sm.memory
        memory_type = memory.memory_type
        confidence = f" (уверенность: {memory.confidence:.0%})" if memory.confidence < 0.8 else ""

        if memory_type == "fact":
            lines.append(f"{i}. [Факт]{confidence}: {memory.content}")
        elif memory_type == "procedure":
            lines.append(f"{i}. [Процедура]{confidence}: {memory.content}")
        elif memory_type == "relationship":
            lines.append(f"{i}. [Связь]{confidence}: {memory.content}")
        else:
            lines.append(f"{i}. {memory.content}")

    return "\n".join(lines)


def format_episodes_context(episodes: list[ScoredEpisode], max_episodes: int = 3) -> str:
    """Format past episodes as reasoning templates."""
    if not episodes:
        return "Нет похожих прошлых рассуждений."

    lines = []
    for i, se in enumerate(episodes[:max_episodes], 1):
        episode = se.episode
        success_rate = episode.success_rate
        success_marker = "✓" if success_rate > 0.5 else "○"

        lines.append(
            f"{i}. [{success_marker}] {episode.behavior_name}: {episode.behavior_instruction}"
        )
        if episode.answer_summary:
            lines.append(f"   Результат: {episode.answer_summary[:100]}...")

    return "\n".join(lines)


def extract_behavior(response: str, query: str) -> Behavior:
    """
    Extract reusable behavior pattern from response.

    Looks for СТРАТЕГИЯ: line in response, falls back to inference.
    """
    # Try to extract explicit strategy line
    strategy_match = re.search(
        r"СТРАТЕГИЯ:\s*\[?([^\]—\n]+?)\]?\s*[—-]\s*(.+?)(?:\n|$)",
        response,
        re.IGNORECASE | re.MULTILINE,
    )

    if strategy_match:
        name = strategy_match.group(1).strip().lower().replace(" ", "_")
        instruction = strategy_match.group(2).strip()
        return Behavior(
            name=name,
            instruction=instruction,
            domain=infer_domain(query, response),
        )

    # Fallback: infer behavior from query pattern
    behavior_name, instruction = infer_behavior(query, response)
    return Behavior(
        name=behavior_name,
        instruction=instruction,
        domain=infer_domain(query, response),
    )


def infer_behavior(query: str, response: str) -> tuple[str, str]:
    """Infer behavior pattern when not explicitly stated."""
    query_lower = query.lower()

    # Pattern matching for common query types
    if any(word in query_lower for word in ["что такое", "what is", "explain", "объясни"]):
        return "explain_concept", "Объяснить концепцию, дать определение и примеры использования"

    if any(word in query_lower for word in ["как ", "how to", "how do"]):
        return "provide_instructions", "Предоставить пошаговые инструкции для выполнения задачи"

    if any(word in query_lower for word in ["ошибка", "error", "не работает", "проблема", "failed"]):
        return "troubleshoot", "Диагностировать проблему и предложить решение"

    if any(word in query_lower for word in ["почему", "why", "зачем"]):
        return "explain_reasoning", "Объяснить причины и логику"

    if any(word in query_lower for word in ["сравни", "compare", "разница", "difference"]):
        return "compare_options", "Сравнить варианты, выделить различия"

    if any(word in query_lower for word in ["лучший", "best", "рекомендуй", "recommend"]):
        return "recommend", "Порекомендовать оптимальный вариант с обоснованием"

    # Default
    return "general_response", "Ответить на вопрос используя доступные знания"


def infer_domain(query: str, response: str) -> str:
    """Infer domain from query and response content."""
    text = (query + " " + response).lower()

    domain_keywords = {
        "docker": ["docker", "контейнер", "container", "образ", "image", "dockerfile"],
        "kubernetes": ["kubernetes", "k8s", "pod", "kubectl", "deployment", "service"],
        "git": ["git", "commit", "branch", "merge", "репозитор"],
        "linux": ["linux", "ubuntu", "bash", "shell", "terminal", "apt"],
        "python": ["python", "pip", "venv", "django", "flask"],
        "database": ["database", "sql", "postgres", "mysql", "mongodb", "база данных"],
        "networking": ["network", "tcp", "http", "dns", "ip", "порт", "сеть"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in text for kw in keywords):
            return domain

    return "general"


class ResponseSynthesizer:
    """
    Synthesizes responses using retrieved context.

    Workflow:
    1. Format context (memories + episodes) for LLM
    2. Generate response with strategy extraction prompt
    3. Extract behavior pattern for episodic memory
    4. Return synthesis result
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_context_memories: int = 10,
        max_context_episodes: int = 3,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.max_context_memories = max_context_memories
        self.max_context_episodes = max_context_episodes

    async def synthesize(
        self,
        query: str,
        retrieval: RetrievalResult,
        temperature: float = 0.5,
    ) -> SynthesisResult:
        """
        Generate response using retrieved context.

        Args:
            query: User query
            retrieval: Retrieved memories, episodes, concepts
            temperature: LLM temperature for response

        Returns:
            SynthesisResult with answer, behavior, and metadata
        """
        # Format context
        memories_context = format_memories_context(
            retrieval.memories, self.max_context_memories
        )
        episodes_context = format_episodes_context(
            retrieval.episodes, self.max_context_episodes
        )

        # Build prompt
        system_prompt, user_prompt = self._build_prompt(query, memories_context, episodes_context)

        # Generate response
        logger.debug(f"Synthesizing response for: {query[:50]}...")
        response = await self.llm.generate(user_prompt, system_prompt=system_prompt, temperature=temperature)

        # Extract behavior pattern
        behavior = extract_behavior(response, query)

        # Calculate importance based on query complexity and context
        importance = self._estimate_importance(query, retrieval)

        # Clean answer (remove strategy line if present)
        answer = self._clean_answer(response)

        return SynthesisResult(
            answer=answer,
            behavior=behavior,
            memories_used=[sm.memory.id for sm in retrieval.memories],
            concepts_activated=list(retrieval.activated_concepts.keys()),
            confidence=self._estimate_confidence(retrieval),
            query=query,
            importance=importance,
        )

    def _build_prompt(
        self,
        query: str,
        memories_context: str,
        episodes_context: str,
    ) -> tuple[str, str]:
        """Build the synthesis prompt. Returns (system_prompt, user_prompt)."""
        # Format current date in Russian
        now = datetime.now()
        current_date = f"{now.day} {RUSSIAN_MONTHS[now.month]} {now.year} года"

        system_prompt = f"""Ты — ассистент, отвечающий на вопросы на основе предоставленного контекста.
Сегодня: {current_date}. Используй эту дату для вопросов о времени, сроках, актуальности и любых временных расчётов."""

        user_prompt = f"""Вопрос пользователя: {query}

Релевантные знания:
{memories_context}

Прошлые похожие рассуждения (используй как шаблон если применимо):
{episodes_context}

Инструкции:
1. Дай полезный ответ на основе предоставленных знаний.
2. Если это вопрос "что такое X" — объясни концепцию.
3. Если это проблема — предложи решение пошагово.
4. Используй текущую дату для вопросов о времени и актуальности информации.
5. Упомяни если уверенность в информации низкая.
6. Не выдумывай информацию, которой нет в контексте.

В конце ответа добавь строку в формате:
СТРАТЕГИЯ: [название_поведения] — [краткое описание твоего подхода к ответу]

Ответ:"""
        return system_prompt, user_prompt

    def _clean_answer(self, response: str) -> str:
        """Remove strategy line from response for clean answer."""
        # Remove СТРАТЕГИЯ line
        cleaned = re.sub(
            r"\n*СТРАТЕГИЯ:\s*\[?[^\]—\n]+?\]?\s*[—-]\s*.+?$",
            "",
            response,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return cleaned.strip()

    def _estimate_importance(self, query: str, retrieval: RetrievalResult) -> float:
        """Estimate importance of this interaction (1-10 scale)."""
        importance = 5.0

        # More concepts = more complex query
        if len(retrieval.query_concepts) > 3:
            importance += 1.0

        # More activated concepts = broader topic
        if len(retrieval.activated_concepts) > 5:
            importance += 1.0

        # Procedural content is often more important
        procedural_count = sum(
            1 for sm in retrieval.memories if sm.memory.memory_type == "procedure"
        )
        if procedural_count > 2:
            importance += 1.0

        # Question length suggests complexity
        if len(query) > 100:
            importance += 0.5

        return min(10.0, importance)

    def _estimate_confidence(self, retrieval: RetrievalResult) -> float:
        """Estimate confidence in the response."""
        if not retrieval.memories:
            return 0.3

        # Average memory confidence
        avg_confidence = sum(
            sm.memory.confidence for sm in retrieval.memories
        ) / len(retrieval.memories)

        # Boost if multiple sources agree
        source_count = len(retrieval.retrieval_sources)
        if source_count >= 3:
            avg_confidence = min(1.0, avg_confidence + 0.1)

        return avg_confidence


async def synthesize_response(
    query: str,
    retrieval: RetrievalResult,
    llm_client: LLMClient | None = None,
) -> SynthesisResult:
    """Convenience function for response synthesis."""
    synthesizer = ResponseSynthesizer(llm_client=llm_client)
    return await synthesizer.synthesize(query, retrieval)
