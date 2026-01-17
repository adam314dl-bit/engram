"""Threshold-triggered reflection for generating higher-level insights.

When accumulated importance of recent episodes exceeds a threshold (150),
the system reflects on patterns and generates meta-knowledge.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.models import EpisodicMemory, SemanticMemory
from engram.retrieval.embeddings import EmbeddingService, get_embedding_service
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Reflection thresholds
IMPORTANCE_THRESHOLD = 150.0
REFLECTION_HOURS = 24  # Look back 24 hours


@dataclass
class Reflection:
    """A single reflection/insight."""

    content: str
    concepts: list[str]  # Related concept names
    reflection_type: str  # "pattern", "approach", "gap"


@dataclass
class ReflectionResult:
    """Result of reflection process."""

    triggered: bool
    importance_sum: float
    episode_count: int
    reflections: list[Reflection] = field(default_factory=list)
    memories_created: list[SemanticMemory] = field(default_factory=list)


class Reflector:
    """
    Generates higher-level insights through reflection.

    Reflection is triggered when accumulated importance exceeds threshold.
    It analyzes recent episodes to identify:
    - Recurring themes/problems
    - Effective approaches
    - Knowledge gaps
    """

    def __init__(
        self,
        db: Neo4jClient,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
        importance_threshold: float = IMPORTANCE_THRESHOLD,
    ) -> None:
        self.db = db
        self.llm = llm_client or get_llm_client()
        self.embeddings = embedding_service or get_embedding_service()
        self.importance_threshold = importance_threshold

    async def maybe_reflect(self) -> ReflectionResult:
        """
        Check if reflection should be triggered and do it if needed.

        Returns:
            ReflectionResult with details
        """
        # Get recent episodes
        recent_episodes = await self.db.get_recent_episodes(
            hours=REFLECTION_HOURS,
            limit=100,
        )

        if not recent_episodes:
            return ReflectionResult(
                triggered=False,
                importance_sum=0,
                episode_count=0,
            )

        # Calculate accumulated importance
        importance_sum = sum(e.importance for e in recent_episodes)

        if importance_sum < self.importance_threshold:
            logger.debug(
                f"Reflection not triggered: importance {importance_sum:.1f} "
                f"< threshold {self.importance_threshold}"
            )
            return ReflectionResult(
                triggered=False,
                importance_sum=importance_sum,
                episode_count=len(recent_episodes),
            )

        # Trigger reflection
        logger.info(
            f"Triggering reflection: importance {importance_sum:.1f} "
            f"from {len(recent_episodes)} episodes"
        )

        reflections = await self._generate_reflections(recent_episodes)
        memories = await self._store_reflections(reflections)

        return ReflectionResult(
            triggered=True,
            importance_sum=importance_sum,
            episode_count=len(recent_episodes),
            reflections=reflections,
            memories_created=memories,
        )

    async def _generate_reflections(
        self,
        episodes: list[EpisodicMemory],
    ) -> list[Reflection]:
        """Generate reflections from recent episodes."""
        # Format episodes for analysis
        episodes_text = self._format_episodes_for_reflection(episodes)

        # Get domain distribution
        domains = {}
        for ep in episodes:
            domains[ep.domain] = domains.get(ep.domain, 0) + 1
        domain_summary = ", ".join(f"{d}: {c}" for d, c in sorted(domains.items(), key=lambda x: -x[1]))

        # Get behavior distribution
        behaviors = {}
        for ep in episodes:
            behaviors[ep.behavior_name] = behaviors.get(ep.behavior_name, 0) + 1
        behavior_summary = ", ".join(f"{b}: {c}" for b, c in sorted(behaviors.items(), key=lambda x: -x[1])[:5])

        prompt = f"""Проанализируй недавние взаимодействия и извлеки высокоуровневые инсайты:

Статистика:
- Всего эпизодов: {len(episodes)}
- Домены: {domain_summary}
- Поведения: {behavior_summary}

Примеры эпизодов:
{episodes_text}

Сгенерируй 2-3 обобщения, отвечая на вопросы:
1. ПАТТЕРН: Какие темы/проблемы повторяются чаще всего?
2. ПОДХОД: Какие подходы/стратегии работают лучше всего?
3. ПРОБЕЛ: Какие пробелы в знаниях выявились (вопросы без хороших ответов)?

Для каждого инсайта укажи связанные концепции в квадратных скобках.

Формат ответа:
ПАТТЕРН: [концепт1, концепт2] Описание паттерна
ПОДХОД: [концепт1] Описание эффективного подхода
ПРОБЕЛ: [концепт1, концепт2] Описание пробела в знаниях"""

        response = await self.llm.generate(prompt, temperature=0.5)
        return self._parse_reflections(response)

    def _parse_reflections(self, response: str) -> list[Reflection]:
        """Parse LLM response into Reflection objects."""
        reflections = []

        # Pattern for extracting reflections
        pattern = r"(ПАТТЕРН|ПОДХОД|ПРОБЕЛ):\s*\[([^\]]+)\]\s*(.+?)(?=(?:ПАТТЕРН|ПОДХОД|ПРОБЕЛ):|$)"

        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        type_map = {
            "ПАТТЕРН": "pattern",
            "ПОДХОД": "approach",
            "ПРОБЕЛ": "gap",
        }

        for reflection_type, concepts_str, content in matches:
            concepts = [c.strip().lower() for c in concepts_str.split(",")]
            content = content.strip()

            if content:
                reflections.append(Reflection(
                    content=content,
                    concepts=concepts,
                    reflection_type=type_map.get(reflection_type.upper(), "pattern"),
                ))

        # Fallback: if no structured reflections found, create one from full response
        if not reflections and response.strip():
            reflections.append(Reflection(
                content=response.strip()[:500],
                concepts=[],
                reflection_type="pattern",
            ))

        return reflections

    async def _store_reflections(
        self,
        reflections: list[Reflection],
    ) -> list[SemanticMemory]:
        """Store reflections as high-importance semantic memories."""
        memories = []

        for reflection in reflections:
            # Generate embedding
            embedding = await self.embeddings.embed(reflection.content)

            # Find concept IDs for mentioned concepts
            concept_ids = []
            for concept_name in reflection.concepts:
                concept = await self.db.get_concept_by_name(concept_name)
                if concept:
                    concept_ids.append(concept.id)

            # Create memory
            memory = SemanticMemory(
                id=f"memory-reflection-{uuid.uuid4()}",
                content=f"[Рефлексия/{reflection.reflection_type}] {reflection.content}",
                concept_ids=concept_ids,
                memory_type="fact",  # Meta-knowledge
                importance=9.0,  # High importance for reflections
                confidence=0.7,  # Moderate confidence (inferred knowledge)
                strength=2.0,
                embedding=embedding,
            )

            await self.db.save_semantic_memory(memory)

            # Link to concepts
            for concept_id in concept_ids:
                try:
                    await self.db.link_memory_to_concept(memory.id, concept_id)
                except Exception as e:
                    logger.warning(f"Failed to link reflection to concept: {e}")

            memories.append(memory)

            logger.info(
                f"Stored reflection ({reflection.reflection_type}): "
                f"{reflection.content[:60]}..."
            )

        return memories

    def _format_episodes_for_reflection(
        self,
        episodes: list[EpisodicMemory],
        max_count: int = 10,
    ) -> str:
        """Format episodes for reflection prompt."""
        # Sort by importance and select diverse examples
        sorted_eps = sorted(episodes, key=lambda e: e.importance, reverse=True)

        lines = []
        for ep in sorted_eps[:max_count]:
            success_marker = "✓" if ep.success_count > ep.failure_count else "✗"
            lines.append(
                f"[{success_marker}] {ep.domain}/{ep.behavior_name}: {ep.query[:80]}"
            )

        return "\n".join(lines)


async def maybe_reflect(db: Neo4jClient) -> ReflectionResult:
    """Convenience function for reflection check."""
    reflector = Reflector(db=db)
    return await reflector.maybe_reflect()
