"""Table enricher using LLM to generate searchable descriptions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.ingestion.parser import generate_id
from engram.models import SemanticMemory
from engram.preprocessing.table_parser import ParsedTable

logger = logging.getLogger(__name__)


# Russian prompts for table enrichment
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


@dataclass
class EnrichedTable:
    """Enriched table with LLM-generated metadata."""

    table: ParsedTable
    description: str = ""
    key_facts: list[str] = field(default_factory=list)
    column_summaries: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # column_summaries: {"AWS": {"да": ["K8s", "Lambda"], "нет": ["GitOps"]}}

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


def create_memories_from_table(
    enriched: EnrichedTable,
    doc_id: str | None = None,
    concept_ids: list[str] | None = None,
) -> list[SemanticMemory]:
    """
    Create semantic memories from an enriched table.

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
