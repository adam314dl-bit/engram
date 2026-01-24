"""Table enricher using LLM to generate searchable descriptions.

v4.1: Search-optimized prompts explaining WHY we need descriptions.
      Uses ContentContext for rich hierarchy information.
      1 table = 1 memory (atomic unit).

v3.3: Multi-vector strategy - stores both searchable summary and raw table.
Summary memory is embedded for search, raw table is fetched at generation time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.models import SemanticMemory
from engram.preprocessing.content_context import ContentContext, ContentType
from engram.preprocessing.table_parser import ParsedTable

if TYPE_CHECKING:
    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class TableWithSummary:
    """Multi-vector table representation for v3.3.

    Links a searchable summary to the raw table for retrieval:
    1. Summary memory is embedded and searched
    2. Raw table memory is fetched at generation time via doc_id link
    """

    doc_id: str  # Links summary ↔ raw table
    summary_ru: str  # Russian summary for embedding/search
    raw_markdown: str  # Full table markdown for generation
    surrounding_context: str  # 2-3 paragraphs around the table
    title: str | None = None
    key_facts: list[str] = field(default_factory=list)


# v4.1: Search-optimized prompt explaining WHY we need descriptions
TABLE_ENRICHMENT_PROMPT = """Ты эксперт по документации. Твоя задача — создать описание таблицы для поисковой системы.

ЗАЧЕМ ЭТО НУЖНО:
1. BM25 (поиск по словам) — нужны точные слова, синонимы, транслитерация (frontend/фронтенд)
2. Векторный поиск (embeddings) — нужен семантический смысл, чтобы найти по похожим запросам
3. Reranker — нужно описание, которое отвечает на потенциальные вопросы пользователя

КОНТЕКСТ ДОКУМЕНТА:
{context}

ТАБЛИЦА:
{table_markdown}

ИНСТРУКЦИИ:
1. ПРИНАДЛЕЖНОСТЬ: К чему относится таблица? (команда, проект, продукт, процесс)
2. ТИП: Что это за таблица? (сравнение, контакты, характеристики, расписание)
3. СТОЛБЦЫ: Краткое описание каждого столбца
4. ОПИСАНИЕ: 2-3 предложения, которые:
   - Содержат ключевые слова для BM25 (включая синонимы и транслитерацию)
   - Передают семантику для векторного поиска
   - Отвечают на вопросы: "email команды...", "сравнение продуктов...", "контакты..."
5. ЗАПРОСЫ_ДЛЯ_ПОИСКА: 3-5 примеров запросов, по которым должна находиться эта таблица
6. ФАКТЫ: 3-5 ключевых фактов из таблицы (имена, числа, конкретные данные)

ФОРМАТ ОТВЕТА:
ПРИНАДЛЕЖНОСТЬ|<к чему относится>
ТИП|<тип таблицы: сравнение, контакты, характеристики, матрица_функций>
СТОЛБЦЫ|<описание столбцов через запятую>
ОПИСАНИЕ|<описание для поиска>
ЗАПРОС|<пример запроса 1>
ЗАПРОС|<пример запроса 2>
ЗАПРОС|<пример запроса 3>
ФАКТ|<факт 1>
ФАКТ|<факт 2>

ПРИМЕР:
ПРИНАДЛЕЖНОСТЬ|Команда Frontend
ТИП|контакты
СТОЛБЦЫ|Имя сотрудника, Роль в команде, Email для связи, Telegram username
ОПИСАНИЕ|Контакты команды фронтенда (frontend team): email и telegram разработчиков. Таблица содержит контактные данные 5 человек: Иван Петров (тимлид, ivan@company.ru), Мария Сидорова (React), Алексей Козлов (Vue).
ЗАПРОС|email фронтенд команды
ЗАПРОС|контакты frontend разработчиков
ЗАПРОС|телеграм тимлида
ЗАПРОС|как связаться с командой фронтенда
ФАКТ|Иван Петров — тимлид, email: ivan@company.ru
ФАКТ|В команде Frontend 5 человек
ФАКТ|Мария Сидорова — React разработчик"""


# Legacy prompts (kept for backward compatibility)
TABLE_DESCRIPTION_PROMPT = """Ты эксперт по анализу данных. Проанализируй следующую таблицу и создай её краткое описание.

{table_context}

Таблица:
{table_markdown}

Инструкции:
1. Опиши что содержит таблица (1-2 предложения)
2. Перечисли 3-5 ключевых фактов из таблицы в формате FACT|<факт>

Пример вывода:
DESCRIPTION|Таблица сравнения возможностей продуктов A, B и C в области автоматизации.
FACT|Продукт A поддерживает интеграцию с Kubernetes
FACT|Только продукт B имеет встроенный мониторинг
FACT|Все продукты поддерживают REST API"""

COMPARISON_TABLE_PROMPT = """Ты эксперт по анализу данных. Проанализируй следующую сравнительную таблицу.

{table_context}

Таблица:
{table_markdown}

Инструкции:
1. Опиши что сравнивается в таблице (1 предложение)
2. Для каждого столбца (кроме первого) выведи ключевые характеристики в формате:
   COLUMN|<название>|<да: список возможностей через запятую>|<нет: список отсутствующих возможностей>
3. Выдели 3-5 ключевых отличий между сравниваемыми объектами в формате FACT|<отличие>

Пример вывода:
DESCRIPTION|Сравнение трёх облачных провайдеров по функциональности.
COLUMN|AWS|Kubernetes, Lambda, S3|нативный GitOps
COLUMN|GCP|Kubernetes, Cloud Functions|Lambda-совместимость
FACT|AWS единственный поддерживает Lambda
FACT|Все провайдеры поддерживают Kubernetes"""


TABLE_SUMMARY_PROMPT = """Создай краткое описание таблицы на русском языке для поиска.

{table_context}

Таблица:
{table_markdown}

Инструкции:
1. Если текст перед таблицей относится к ней — объедини его смысл с описанием таблицы в одно связное описание.
   Не копируй контекст дословно, а суммаризируй вместе с данными таблицы.
   Например: заголовок "Февраль" + таблица с именами и датами → "Дни рождения сотрудников в феврале: Иван Петров (05.02), ..."
2. Опиши что содержит таблица и какие ключевые данные представлены
3. Упомяни для чего таблица может быть полезна

Ответ: 2-3 предложения на русском языке, без форматирования."""


def extract_surrounding_context(
    full_text: str,
    table_markdown: str,
    context_paragraphs: int = 2,
) -> str:
    """
    Extract surrounding context (paragraphs) around a table.

    Args:
        full_text: Full document text
        table_markdown: Table markdown to find
        context_paragraphs: Number of paragraphs before/after

    Returns:
        Surrounding context string
    """
    # Try to find table in text
    # Tables might be slightly different due to normalization
    table_lines = table_markdown.strip().split("\n")
    if not table_lines:
        return ""

    # Find first line of table in text
    first_line = table_lines[0].strip()
    if first_line not in full_text:
        # Try finding pipe-delimited content
        first_line = first_line.replace("|", "").strip()

    # Split text into paragraphs
    paragraphs = re.split(r"\n\n+", full_text)

    # Find paragraph containing table
    table_idx = -1
    for i, para in enumerate(paragraphs):
        if first_line[:30] in para or ("|" in para and "---" in para):
            table_idx = i
            break

    if table_idx == -1:
        return ""

    # Get surrounding paragraphs
    start_idx = max(0, table_idx - context_paragraphs)
    end_idx = min(len(paragraphs), table_idx + context_paragraphs + 1)

    context_parts = []
    for i in range(start_idx, end_idx):
        if i != table_idx:  # Skip the table itself
            para = paragraphs[i].strip()
            if para and not para.startswith("|"):  # Skip table content
                context_parts.append(para)

    return "\n\n".join(context_parts)


@dataclass
class EnrichedTable:
    """Enriched table with LLM-generated metadata.

    v4.1: Added belonging_context, table_type_semantic, column_descriptions, search_queries.
    """

    table: ParsedTable
    description: str = ""
    key_facts: list[str] = field(default_factory=list)
    column_summaries: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # column_summaries: {"AWS": {"да": ["K8s", "Lambda"], "нет": ["GitOps"]}}

    # v4.1 new fields
    belonging_context: str = ""  # К чему относится (команда, проект, etc.)
    table_type_semantic: str = ""  # Семантический тип (сравнение, контакты, etc.)
    column_descriptions: str = ""  # Описание столбцов
    search_queries: list[str] = field(default_factory=list)  # Example search queries

    @property
    def table_type(self) -> str:
        """Determine table type based on content."""
        if self.table.is_comparison_table:
            return "comparison"
        return "data"


class TableEnricher:
    """Enrich tables with LLM-generated descriptions and facts."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or get_llm_client()

    async def enrich_v41(
        self,
        table: ParsedTable,
        context: ContentContext | None = None,
    ) -> EnrichedTable:
        """
        Enrich table with v4.1 search-optimized descriptions.

        Args:
            table: Parsed table to enrich
            context: Content context with hierarchy information

        Returns:
            EnrichedTable with search-optimized metadata
        """
        # Format context for prompt
        if context:
            context_str = context.format_for_prompt(ContentType.TABLE)
        else:
            context_str = self._build_basic_context(table)

        prompt = TABLE_ENRICHMENT_PROMPT.format(
            context=context_str,
            table_markdown=table.to_markdown(),
        )

        try:
            result = await self.llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=1000,
            )
            return self._parse_v41_result(table, result)
        except Exception as e:
            logger.warning(f"Failed to enrich table with v4.1: {e}")
            return self._fallback_enrichment(table)

    def _build_basic_context(self, table: ParsedTable) -> str:
        """Build basic context from table metadata."""
        parts = []
        if table.title:
            parts.append(f"Заголовок: {table.title}")
        if table.context:
            parts.append(f"Текст перед таблицей: {table.context}")
        return "\n".join(parts) if parts else "Контекст отсутствует."

    def _parse_v41_result(self, table: ParsedTable, result: str) -> EnrichedTable:
        """Parse v4.1 enrichment result."""
        belonging = ""
        table_type_semantic = ""
        column_descriptions = ""
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
                table_type_semantic = value
            elif key == "СТОЛБЦЫ":
                column_descriptions = value
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
            description = self._generate_fallback_description(table)

        return EnrichedTable(
            table=table,
            description=description,
            key_facts=facts,
            belonging_context=belonging,
            table_type_semantic=table_type_semantic,
            column_descriptions=column_descriptions,
            search_queries=search_queries,
        )

    async def enrich_multi_vector(
        self,
        table: ParsedTable,
        full_document_text: str | None = None,
    ) -> TableWithSummary:
        """
        Enrich table with multi-vector strategy (v3.3).

        Creates both a searchable summary and preserves raw table.

        Args:
            table: Parsed table
            full_document_text: Full document for context extraction

        Returns:
            TableWithSummary with linked summary and raw table
        """
        doc_id = generate_id()

        # Get surrounding context if document provided
        surrounding_context = ""
        if full_document_text:
            surrounding_context = extract_surrounding_context(
                full_document_text,
                table.to_markdown(),
            )

        # Build context for prompt
        context_parts = []
        if table.title:
            context_parts.append(f"Заголовок: {table.title}")
        if table.context:
            context_parts.append(f"Контекст: {table.context}")
        if surrounding_context:
            context_parts.append(f"Окружающий текст: {surrounding_context[:500]}")
        table_context = "\n".join(context_parts) if context_parts else "Нет контекста."

        # Generate summary
        prompt = TABLE_SUMMARY_PROMPT.format(
            table_context=table_context,
            table_markdown=table.to_markdown(),
        )

        try:
            summary_ru = await self.llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=500,
            )
            summary_ru = summary_ru.strip()
        except Exception as e:
            logger.warning(f"Failed to generate table summary: {e}")
            summary_ru = self._generate_fallback_description(table)

        # Also get key facts via standard enrichment
        enriched = await self.enrich(table)

        return TableWithSummary(
            doc_id=doc_id,
            summary_ru=summary_ru,
            raw_markdown=table.to_markdown(),
            surrounding_context=surrounding_context,
            title=table.title,
            key_facts=enriched.key_facts,
        )

    async def enrich(self, table: ParsedTable) -> EnrichedTable:
        """
        Enrich a table with description and key facts.

        Args:
            table: Parsed table to enrich

        Returns:
            EnrichedTable with LLM-generated metadata
        """
        # Build context string
        context_parts = []
        if table.title:
            context_parts.append(f"Заголовок: {table.title}")
        if table.context:
            context_parts.append(f"Контекст: {table.context}")
        table_context = "\n".join(context_parts) if context_parts else "Нет дополнительного контекста."

        # Choose prompt based on table type
        if table.is_comparison_table:
            return await self._enrich_comparison_table(table, table_context)
        else:
            return await self._enrich_data_table(table, table_context)

    async def _enrich_data_table(
        self, table: ParsedTable, context: str
    ) -> EnrichedTable:
        """Enrich a regular data table."""
        prompt = TABLE_DESCRIPTION_PROMPT.format(
            table_context=context,
            table_markdown=table.to_markdown(),
        )

        try:
            result = await self.llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            logger.warning(f"Failed to enrich table: {e}")
            return self._fallback_enrichment(table)

        return self._parse_enrichment_result(table, result)

    async def _enrich_comparison_table(
        self, table: ParsedTable, context: str
    ) -> EnrichedTable:
        """Enrich a comparison/feature matrix table."""
        prompt = COMPARISON_TABLE_PROMPT.format(
            table_context=context,
            table_markdown=table.to_markdown(),
        )

        try:
            result = await self.llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=1500,
            )
        except Exception as e:
            logger.warning(f"Failed to enrich comparison table: {e}")
            return self._fallback_enrichment(table)

        return self._parse_comparison_result(table, result)

    def _parse_enrichment_result(self, table: ParsedTable, result: str) -> EnrichedTable:
        """Parse LLM enrichment result."""
        description = ""
        facts: list[str] = []

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("DESCRIPTION|"):
                description = line.split("|", 1)[1].strip()
            elif line.startswith("FACT|"):
                fact = line.split("|", 1)[1].strip()
                if fact:
                    facts.append(fact)

        # Fallback if no description extracted
        if not description:
            description = self._generate_fallback_description(table)

        return EnrichedTable(
            table=table,
            description=description,
            key_facts=facts,
        )

    def _parse_comparison_result(self, table: ParsedTable, result: str) -> EnrichedTable:
        """Parse comparison table enrichment result."""
        description = ""
        facts: list[str] = []
        column_summaries: dict[str, dict[str, list[str]]] = {}

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("DESCRIPTION|"):
                description = line.split("|", 1)[1].strip()
            elif line.startswith("FACT|"):
                fact = line.split("|", 1)[1].strip()
                if fact:
                    facts.append(fact)
            elif line.startswith("COLUMN|"):
                parts = line.split("|")
                if len(parts) >= 4:
                    col_name = parts[1].strip()
                    yes_features = [f.strip() for f in parts[2].split(",") if f.strip()]
                    no_features = [f.strip() for f in parts[3].split(",") if f.strip()]
                    column_summaries[col_name] = {
                        "да": yes_features,
                        "нет": no_features,
                    }

        if not description:
            description = self._generate_fallback_description(table)

        return EnrichedTable(
            table=table,
            description=description,
            key_facts=facts,
            column_summaries=column_summaries,
        )

    def _fallback_enrichment(self, table: ParsedTable) -> EnrichedTable:
        """Generate fallback enrichment without LLM."""
        description = self._generate_fallback_description(table)
        facts = self._extract_basic_facts(table)

        return EnrichedTable(
            table=table,
            description=description,
            key_facts=facts,
        )

    def _generate_fallback_description(self, table: ParsedTable) -> str:
        """Generate a basic description without LLM."""
        parts = []

        if table.title:
            parts.append(f"Таблица: {table.title}.")
        else:
            parts.append("Таблица данных.")

        parts.append(f"Содержит {table.row_count} строк и {table.column_count} столбцов.")

        if table.headers:
            parts.append(f"Столбцы: {', '.join(table.headers)}.")

        return " ".join(parts)

    def _extract_basic_facts(self, table: ParsedTable) -> list[str]:
        """Extract basic facts from table without LLM."""
        facts: list[str] = []

        # For comparison tables, extract yes/no features
        if table.is_comparison_table and table.rows:
            for row in table.rows[:5]:  # Limit to first 5 rows
                if len(row) > 1:
                    feature = row[0]
                    for i, cell in enumerate(row[1:], 1):
                        if i < len(table.headers):
                            header = table.headers[i]
                            if cell.lower() in ("да", "yes", "+"):
                                facts.append(f"{header} поддерживает {feature}")
                            elif cell.lower() in ("нет", "no", "-"):
                                facts.append(f"{header} не поддерживает {feature}")

        return facts[:10]  # Limit facts


def create_memory_from_table(
    enriched: EnrichedTable,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> SemanticMemory:
    """
    Create a single semantic memory from an enriched table (v4.1).

    Creates exactly 1 memory per table (atomic unit).
    Uses description for embedding, raw_markdown for generation.

    Args:
        enriched: Enriched table
        doc_id: Source document ID
        concept_ids: Concept IDs to link memory to

    Returns:
        SemanticMemory for the table
    """
    # Build content: description + raw for generation
    content_parts = []

    # Add belonging context as header
    if enriched.belonging_context:
        content_parts.append(f"# {enriched.belonging_context}")

    # Add description
    if enriched.description:
        content_parts.append(enriched.description)

    # Add raw markdown for generation context
    content_parts.append("")
    content_parts.append(enriched.table.raw_text or enriched.table.to_markdown())

    content = "\n".join(content_parts)

    # Build metadata
    metadata = {
        "source_type": "table",
        "table_type": enriched.table_type,
        "table_type_semantic": enriched.table_type_semantic,
        "belonging_context": enriched.belonging_context,
        "column_descriptions": enriched.column_descriptions,
        "row_count": enriched.table.row_count,
        "column_count": enriched.table.column_count,
        "search_queries": enriched.search_queries,
        "key_facts": enriched.key_facts,
    }

    if enriched.table.title:
        metadata["table_title"] = enriched.table.title

    return SemanticMemory(
        id=generate_id(),
        content=content,
        concept_ids=concept_ids or [],
        source_doc_ids=[doc_id] if doc_id else [],
        memory_type="fact",
        importance=6.0,  # Tables are usually important
        metadata=metadata,
    )


def create_fact_memories_from_table(
    enriched: EnrichedTable,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> list[SemanticMemory]:
    """
    Create additional fact memories from table key facts.

    Args:
        enriched: Enriched table
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
                "source_type": "table_fact",
                "source_table": enriched.belonging_context or enriched.table.title,
            },
        ))

    return memories


def create_memories_from_table(
    enriched: EnrichedTable,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> list[SemanticMemory]:
    """
    Create semantic memories from an enriched table (legacy v3.x).

    Creates:
    1. One memory for the table description
    2. One memory for each key fact
    3. One memory for each row (if data table) or column summary (if comparison)

    Args:
        enriched: Enriched table
        doc_id: Source document ID
        concept_ids: Concept IDs to link memories to

    Returns:
        List of SemanticMemory objects
    """
    memories: list[SemanticMemory] = []
    table = enriched.table

    # Memory for table description
    if enriched.description:
        desc_content = enriched.description
        if table.title:
            desc_content = f"{table.title}: {desc_content}"

        memories.append(SemanticMemory(
            id=generate_id(),
            content=desc_content,
            concept_ids=concept_ids or [],
            source_doc_ids=[doc_id] if doc_id else [],
            memory_type="fact",
            importance=6.0,  # Tables are usually important
        ))

    # Memory for each key fact
    for fact in enriched.key_facts:
        memories.append(SemanticMemory(
            id=generate_id(),
            content=fact,
            concept_ids=concept_ids or [],
            source_doc_ids=[doc_id] if doc_id else [],
            memory_type="fact",
            importance=5.0,
        ))

    # For comparison tables, create column-based memories
    if enriched.table_type == "comparison" and enriched.column_summaries:
        for col_name, features in enriched.column_summaries.items():
            yes_list = features.get("да", [])
            no_list = features.get("нет", [])

            if yes_list:
                content = f"{col_name} поддерживает: {', '.join(yes_list)}"
                memories.append(SemanticMemory(
                    id=generate_id(),
                    content=content,
                    concept_ids=concept_ids or [],
                    source_doc_ids=[doc_id] if doc_id else [],
                    memory_type="fact",
                    importance=5.5,
                ))

            if no_list:
                content = f"{col_name} не поддерживает: {', '.join(no_list)}"
                memories.append(SemanticMemory(
                    id=generate_id(),
                    content=content,
                    concept_ids=concept_ids or [],
                    source_doc_ids=[doc_id] if doc_id else [],
                    memory_type="fact",
                    importance=5.0,
                ))
    else:
        # For data tables, create row-based memories
        row_texts = table.to_row_texts()
        for row_text in row_texts[:20]:  # Limit rows
            if len(row_text) > 20:  # Skip very short rows
                memories.append(SemanticMemory(
                    id=generate_id(),
                    content=row_text,
                    concept_ids=concept_ids or [],
                    source_doc_ids=[doc_id] if doc_id else [],
                    memory_type="fact",
                    importance=4.5,
                ))

    return memories


def create_multi_vector_memories(
    table_summary: TableWithSummary,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> tuple[SemanticMemory, SemanticMemory]:
    """
    Create multi-vector memories from a table (v3.3).

    Creates two linked memories:
    1. Summary memory: embedded, searchable
    2. Raw table memory: not indexed, fetched at generation time

    Args:
        table_summary: TableWithSummary with summary and raw table
        doc_id: Source document ID
        concept_ids: Concept IDs to link memories to

    Returns:
        Tuple of (summary_memory, raw_table_memory)
    """
    # Summary memory - this is embedded and searched
    summary_content = table_summary.summary_ru
    if table_summary.title:
        summary_content = f"{table_summary.title}: {summary_content}"

    # Add key facts to summary for better search
    if table_summary.key_facts:
        facts_text = " | ".join(table_summary.key_facts[:3])
        summary_content = f"{summary_content}\nКлючевые факты: {facts_text}"

    summary_memory = SemanticMemory(
        id=generate_id(),
        content=summary_content,
        concept_ids=concept_ids or [],
        source_doc_ids=[doc_id] if doc_id else [],
        memory_type="table_summary",
        importance=6.0,
        metadata={
            "table_doc_id": table_summary.doc_id,
            "has_raw_table": True,
            "source_type": "table",
        },
    )

    # Raw table memory - not indexed, fetched at generation time
    raw_content = table_summary.raw_markdown
    if table_summary.surrounding_context:
        raw_content = f"{table_summary.surrounding_context}\n\n{raw_content}"

    raw_memory = SemanticMemory(
        id=generate_id(),
        content=raw_content,
        concept_ids=concept_ids or [],
        source_doc_ids=[doc_id] if doc_id else [],
        memory_type="table_raw",
        importance=5.0,
        metadata={
            "table_doc_id": table_summary.doc_id,
            "for_generation_only": True,  # Don't index this
            "linked_summary_id": summary_memory.id,
            "source_type": "table",
        },
    )

    # Update summary with link to raw
    summary_memory.metadata["linked_raw_id"] = raw_memory.id

    return summary_memory, raw_memory


async def fetch_raw_tables_for_summaries(
    db: "Neo4jClient",  # type: ignore
    summary_ids: list[str],
) -> dict[str, SemanticMemory]:
    """
    Fetch raw table memories linked to summary memories.

    Used at generation time to get full table content.

    Args:
        db: Neo4j client
        summary_ids: IDs of summary memories

    Returns:
        Dict mapping summary_id to raw table memory
    """
    result: dict[str, SemanticMemory] = {}

    for summary_id in summary_ids:
        # Get summary to find linked raw ID
        summary = await db.get_semantic_memory(summary_id)
        if not summary or not summary.metadata:
            continue

        raw_id = summary.metadata.get("linked_raw_id")
        if not raw_id:
            continue

        raw_memory = await db.get_semantic_memory(raw_id)
        if raw_memory:
            result[summary_id] = raw_memory

    return result
