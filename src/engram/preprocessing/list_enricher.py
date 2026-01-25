"""List enricher using LLM to generate searchable descriptions.

v4.1: Creates search-optimized descriptions for lists.
Mirrors table_enricher.py structure for consistency.

Each list becomes exactly 1 memory unit (atomic).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.models import SemanticMemory
from engram.preprocessing.content_context import ContentContext, ContentType
from engram.preprocessing.list_parser import ListType, ParsedList

logger = logging.getLogger(__name__)


# Russian prompt that explains WHY we need good descriptions
LIST_ENRICHMENT_PROMPT = """Ты эксперт по документации. Твоя задача — создать описание списка для поисковой системы.

ЗАЧЕМ ЭТО НУЖНО:
1. BM25 (поиск по словам) — нужны точные слова, синонимы, транслитерация (frontend/фронтенд)
2. Векторный поиск (embeddings) — нужен семантический смысл, чтобы найти по похожим запросам
3. Reranker — нужно описание, которое отвечает на потенциальные вопросы пользователя

КОНТЕКСТ ДОКУМЕНТА:
{context}

СПИСОК:
{list_markdown}

ИНСТРУКЦИИ:
1. ПРИНАДЛЕЖНОСТЬ: К чему относится список? (команда, проект, процесс, технология)
2. ТИП: Что это за список? (состав команды, этапы процесса, требования, характеристики)
3. ОПИСАНИЕ: 2-3 предложения, которые:
   - Содержат ключевые слова для BM25 (включая синонимы и транслитерацию)
   - Передают семантику для векторного поиска
   - Отвечают на вопросы: "кто входит в...", "какие шаги...", "что включает..."
4. ЗАПРОСЫ_ДЛЯ_ПОИСКА: 3-5 примеров запросов, по которым должен находиться этот список
5. ФАКТЫ: 2-4 ключевых факта из списка (имена, числа, конкретика)

ФОРМАТ ОТВЕТА:
ПРИНАДЛЕЖНОСТЬ|<к чему относится>
ТИП|<тип списка: состав_команды, этапы_процесса, требования, характеристики, перечисление>
ОПИСАНИЕ|<описание для поиска>
ЗАПРОС|<пример запроса 1>
ЗАПРОС|<пример запроса 2>
ЗАПРОС|<пример запроса 3>
ФАКТ|<факт 1>
ФАКТ|<факт 2>

ПРИМЕР:
ПРИНАДЛЕЖНОСТЬ|Команда Frontend
ТИП|состав_команды
ОПИСАНИЕ|Состав команды фронтенда (frontend team): разработчики, тестировщики и дизайнеры. В команде 5 человек: Иван Петров (тимлид), Мария Сидорова (React), Алексей Козлов (Vue), Анна Новикова (QA), Дмитрий Волков (UI/UX дизайнер).
ЗАПРОС|кто в команде фронтенда
ЗАПРОС|состав frontend team
ЗАПРОС|разработчики фронтенд
ЗАПРОС|тимлид frontend
ФАКТ|Иван Петров — тимлид команды Frontend
ФАКТ|В команде 5 человек
ФАКТ|Мария Сидорова — React разработчик"""


@dataclass
class EnrichedList:
    """Enriched list with LLM-generated metadata."""

    list: ParsedList
    belonging_context: str = ""  # К чему относится (команда, проект, etc.)
    list_type_semantic: str = ""  # Семантический тип (состав_команды, этапы, etc.)
    description: str = ""  # Search-optimized description
    search_queries: list[str] = field(default_factory=list)  # Example search queries
    key_facts: list[str] = field(default_factory=list)  # Extracted key facts

    @property
    def list_type_name(self) -> str:
        """Human-readable list type name."""
        return self.list.list_type.value


class ListEnricher:
    """Enrich lists with LLM-generated descriptions optimized for search."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or get_llm_client()

    async def enrich(
        self,
        parsed_list: ParsedList,
        context: ContentContext | None = None,
    ) -> EnrichedList:
        """
        Enrich a list with search-optimized description.

        Args:
            parsed_list: Parsed list to enrich
            context: Optional content context for hierarchy info

        Returns:
            EnrichedList with LLM-generated metadata
        """
        # Format context for prompt
        if context:
            context_str = context.format_for_prompt(ContentType.LIST)
        else:
            context_str = self._build_basic_context(parsed_list)

        prompt = LIST_ENRICHMENT_PROMPT.format(
            context=context_str,
            list_markdown=parsed_list.to_markdown(),
        )

        try:
            result = await self.llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=800,
            )
            return self._parse_enrichment_result(parsed_list, result)
        except Exception as e:
            logger.warning(f"Failed to enrich list: {e}")
            return self._fallback_enrichment(parsed_list, context)

    def _build_basic_context(self, parsed_list: ParsedList) -> str:
        """Build basic context from parsed list metadata."""
        parts = []
        if parsed_list.title:
            parts.append(f"Заголовок: {parsed_list.title}")
        if parsed_list.context:
            parts.append(f"Текст перед списком: {parsed_list.context}")
        return "\n".join(parts) if parts else "Контекст отсутствует."

    def _parse_enrichment_result(
        self, parsed_list: ParsedList, result: str
    ) -> EnrichedList:
        """Parse LLM enrichment result."""
        belonging = ""
        list_type_semantic = ""
        description = ""
        search_queries: list[str] = []
        facts: list[str] = []

        for line in result.split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = line.split("|", 1)
            if len(parts) != 2:
                continue

            key = parts[0].strip().upper()
            value = parts[1].strip()

            if key == "ПРИНАДЛЕЖНОСТЬ":
                belonging = value
            elif key == "ТИП":
                list_type_semantic = value
            elif key == "ОПИСАНИЕ":
                description = value
            elif key == "ЗАПРОС":
                if value:
                    search_queries.append(value)
            elif key == "ФАКТ":
                if value:
                    facts.append(value)

        # Fallback if no description
        if not description:
            description = self._generate_fallback_description(parsed_list)

        return EnrichedList(
            list=parsed_list,
            belonging_context=belonging,
            list_type_semantic=list_type_semantic,
            description=description,
            search_queries=search_queries,
            key_facts=facts,
        )

    def _fallback_enrichment(
        self, parsed_list: ParsedList, context: ContentContext | None
    ) -> EnrichedList:
        """Generate fallback enrichment without LLM."""
        description = self._generate_fallback_description(parsed_list)

        # Try to determine belonging from context
        belonging = ""
        if context:
            belonging = context.get_best_context_hint()
        elif parsed_list.title:
            belonging = parsed_list.title

        # Map list type to semantic type
        type_map = {
            ListType.NUMBERED: "этапы_процесса",
            ListType.CHECKLIST: "задачи",
            ListType.DEFINITION: "определения",
            ListType.BULLET: "перечисление",
        }
        list_type_semantic = type_map.get(parsed_list.list_type, "перечисление")

        # Extract basic facts (first 3 items)
        facts = parsed_list.items[:3]

        return EnrichedList(
            list=parsed_list,
            belonging_context=belonging,
            list_type_semantic=list_type_semantic,
            description=description,
            search_queries=[],
            key_facts=facts,
        )

    def _generate_fallback_description(self, parsed_list: ParsedList) -> str:
        """Generate a basic description without LLM."""
        parts = []

        # Add title context
        if parsed_list.title:
            parts.append(f"{parsed_list.title}.")

        # Add type description
        type_descriptions = {
            ListType.NUMBERED: "Нумерованный список",
            ListType.CHECKLIST: "Чек-лист задач",
            ListType.DEFINITION: "Список определений",
            ListType.BULLET: "Список",
        }
        parts.append(f"{type_descriptions.get(parsed_list.list_type, 'Список')}.")
        parts.append(f"Содержит {parsed_list.item_count} пунктов.")

        # Add first few items
        if parsed_list.items:
            sample = parsed_list.items[:3]
            parts.append("Включает: " + "; ".join(sample))

        return " ".join(parts)


def create_memory_from_list(
    enriched: EnrichedList,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> SemanticMemory:
    """
    Create a semantic memory from an enriched list.

    Creates exactly 1 memory per list (atomic unit).
    v4.5: Uses search_content for embedding, content for LLM generation.

    Args:
        enriched: Enriched list
        doc_id: Source document ID
        concept_ids: Concept IDs to link memory to

    Returns:
        SemanticMemory for the list
    """
    # v4.5: Build search_content for embedding (description + search queries)
    search_parts = []
    if enriched.belonging_context:
        search_parts.append(enriched.belonging_context)
    if enriched.description:
        search_parts.append(enriched.description)
    if enriched.search_queries:
        search_parts.append(" | ".join(enriched.search_queries))
    if enriched.key_facts:
        search_parts.append(" | ".join(enriched.key_facts[:3]))

    search_content = " | ".join(search_parts) if search_parts else None

    # Build content: raw data for LLM generation
    content_parts = []

    # Add belonging context as header
    if enriched.belonging_context:
        content_parts.append(f"# {enriched.belonging_context}")

    # Add description
    if enriched.description:
        content_parts.append(enriched.description)

    # Add raw markdown for generation context
    content_parts.append("")
    content_parts.append(enriched.list.raw_markdown)

    content = "\n".join(content_parts)

    # Build metadata
    metadata = {
        "source_type": "list",
        "list_type": enriched.list.list_type.value,
        "list_type_semantic": enriched.list_type_semantic,
        "belonging_context": enriched.belonging_context,
        "item_count": enriched.list.item_count,
        "search_queries": enriched.search_queries,
        "key_facts": enriched.key_facts,
    }

    if enriched.list.title:
        metadata["list_title"] = enriched.list.title

    return SemanticMemory(
        id=generate_id(),
        content=content,
        search_content=search_content,
        concept_ids=concept_ids or [],
        source_doc_ids=[doc_id] if doc_id else [],
        memory_type="fact",
        importance=5.5,  # Lists are moderately important
        metadata=metadata,
    )


def create_fact_memories_from_list(
    enriched: EnrichedList,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> list[SemanticMemory]:
    """
    Create additional fact memories from list key facts.

    Args:
        enriched: Enriched list
        doc_id: Source document ID
        concept_ids: Concept IDs to link memories to

    Returns:
        List of SemanticMemory objects for key facts
    """
    memories: list[SemanticMemory] = []

    for fact in enriched.key_facts:
        if len(fact) < 10:  # Skip very short facts
            continue

        memories.append(SemanticMemory(
            id=generate_id(),
            content=fact,
            concept_ids=concept_ids or [],
            source_doc_ids=[doc_id] if doc_id else [],
            memory_type="fact",
            importance=5.0,
            metadata={
                "source_type": "list_fact",
                "source_list": enriched.belonging_context,
            },
        ))

    return memories
