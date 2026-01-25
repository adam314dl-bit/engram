"""KB summary generation and storage for query enrichment.

Generates a summary of the knowledge base contents to help with:
- Query understanding (is this in scope?)
- Query rewriting (use domain-specific terms)
- Out-of-scope detection

The summary is computed once after ingestion and stored in Neo4j.

v4.4: LLM-enhanced summary generation with domain description,
capabilities, limitations, and sample questions.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from engram.config import settings
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class KBSummary:
    """Knowledge base summary for query enrichment."""

    # Domain coverage
    domains: list[str] = field(default_factory=list)  # e.g., ["DevOps", "HR", "Finance"]

    # Entity types present
    entity_types: dict[str, list[str]] = field(default_factory=dict)  # e.g., {"people": ["Иванов", "Петров"]}

    # Information types
    info_types: list[str] = field(default_factory=list)  # e.g., ["procedures", "contacts", "policies"]

    # Out-of-scope topics (explicitly not covered)
    out_of_scope: list[str] = field(default_factory=list)

    # Key terms and their frequencies
    key_terms: dict[str, int] = field(default_factory=dict)  # term -> count

    # Statistics
    statistics: dict[str, Any] = field(default_factory=dict)

    # v4.4 LLM-enhanced fields
    domain_description: str | None = None  # "Корпоративная база знаний о..."
    capabilities: list[str] = field(default_factory=list)  # What can be answered
    limitations: list[str] = field(default_factory=list)  # What's NOT covered
    sample_questions: list[str] = field(default_factory=list)  # Example queries

    def to_prompt_text(self, max_tokens: int = 500) -> str:
        """Format summary for inclusion in query enrichment prompts.

        Args:
            max_tokens: Approximate max tokens for the summary

        Returns:
            Formatted text for LLM prompt
        """
        parts = []

        # v4.4 LLM-enhanced fields (prioritized at top)
        if self.domain_description:
            parts.append(f"ДОМЕН: {self.domain_description}")

        if self.capabilities:
            cap_lines = ["ВОЗМОЖНОСТИ (какие вопросы можно задать):"]
            for cap in self.capabilities[:7]:
                cap_lines.append(f"  - {cap}")
            parts.append("\n".join(cap_lines))

        if self.limitations:
            lim_lines = ["ОГРАНИЧЕНИЯ (чего НЕТ в базе):"]
            for lim in self.limitations[:5]:
                lim_lines.append(f"  - {lim}")
            parts.append("\n".join(lim_lines))

        if self.sample_questions:
            q_lines = ["ПРИМЕРЫ ХОРОШИХ ВОПРОСОВ:"]
            for q in self.sample_questions[:5]:
                q_lines.append(f"  - {q}")
            parts.append("\n".join(q_lines))

        # Entity types (top 3 per type)
        if self.entity_types:
            entity_lines = ["Типы сущностей:"]
            for etype, entities in list(self.entity_types.items())[:5]:
                sample = entities[:3]
                if len(entities) > 3:
                    sample_str = f"{', '.join(sample)} и др."
                else:
                    sample_str = ', '.join(sample)
                entity_lines.append(f"  - {etype}: {sample_str}")
            if len(entity_lines) > 1:
                parts.append("\n".join(entity_lines))

        # Statistics
        if self.statistics:
            stats_parts = []
            if "concept_count" in self.statistics:
                stats_parts.append(f"{self.statistics['concept_count']} концептов")
            if "memory_count" in self.statistics:
                stats_parts.append(f"{self.statistics['memory_count']} фактов")
            if "document_count" in self.statistics:
                stats_parts.append(f"{self.statistics['document_count']} документов")
            if stats_parts:
                parts.append(f"Статистика: {', '.join(stats_parts)}")

        result = "\n".join(parts)

        # Truncate if too long (rough estimate: 1 token ~ 4 chars for Russian)
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "..."

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "domains": self.domains,
            "entity_types": json.dumps(self.entity_types, ensure_ascii=False),
            "info_types": self.info_types,
            "out_of_scope": self.out_of_scope,
            "key_terms": json.dumps(self.key_terms, ensure_ascii=False),
            "statistics": json.dumps(self.statistics, ensure_ascii=False),
            # v4.4 LLM-enhanced fields
            "domain_description": self.domain_description or "",
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "sample_questions": self.sample_questions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KBSummary":
        """Deserialize from dictionary."""
        entity_types = data.get("entity_types", "{}")
        if isinstance(entity_types, str):
            entity_types = json.loads(entity_types)

        key_terms = data.get("key_terms", "{}")
        if isinstance(key_terms, str):
            key_terms = json.loads(key_terms)

        statistics = data.get("statistics", "{}")
        if isinstance(statistics, str):
            statistics = json.loads(statistics)

        # Handle domain_description (empty string -> None)
        domain_description = data.get("domain_description")
        if domain_description == "":
            domain_description = None

        return cls(
            domains=data.get("domains", []),
            entity_types=entity_types,
            info_types=data.get("info_types", []),
            out_of_scope=data.get("out_of_scope", []),
            key_terms=key_terms,
            statistics=statistics,
            # v4.4 LLM-enhanced fields
            domain_description=domain_description,
            capabilities=data.get("capabilities", []),
            limitations=data.get("limitations", []),
            sample_questions=data.get("sample_questions", []),
        )


class KBSummaryGenerator:
    """Generates KB summary from Neo4j graph data."""

    def __init__(self, db: Neo4jClient) -> None:
        self.db = db

    async def generate(self) -> KBSummary:
        """Generate summary from current graph state.

        Returns:
            KBSummary with domains, entities, terms, and statistics
        """
        logger.info("Generating KB summary...")

        # Gather data in parallel-ish fashion
        statistics = await self._get_statistics()
        domains = await self._extract_domains()
        entity_types = await self._extract_entity_types()
        info_types = await self._extract_info_types()
        key_terms = await self._extract_key_terms()

        summary = KBSummary(
            domains=domains,
            entity_types=entity_types,
            info_types=info_types,
            out_of_scope=[],  # Could be manually configured later
            key_terms=key_terms,
            statistics=statistics,
        )

        logger.info(
            f"KB summary generated: {len(domains)} domains, "
            f"{len(entity_types)} entity types, {len(key_terms)} key terms"
        )

        return summary

    async def _get_statistics(self) -> dict[str, Any]:
        """Get basic statistics from the graph."""
        query = """
        MATCH (c:Concept) WITH count(c) as concepts
        MATCH (m:SemanticMemory) WITH concepts, count(m) as memories
        MATCH (d:Document) WITH concepts, memories, count(d) as documents
        RETURN concepts, memories, documents
        """
        result = await self.db.execute_query(query)
        if result:
            return {
                "concept_count": result[0]["concepts"],
                "memory_count": result[0]["memories"],
                "document_count": result[0]["documents"],
            }
        return {}

    async def _extract_domains(self) -> list[str]:
        """Extract top domains/topics from concept types."""
        query = """
        MATCH (c:Concept)
        WHERE c.type IS NOT NULL AND c.type <> ''
        RETURN c.type as domain, count(*) as cnt
        ORDER BY cnt DESC
        LIMIT 20
        """
        result = await self.db.execute_query(query)
        return [r["domain"] for r in result if r["domain"]]

    async def _extract_entity_types(self) -> dict[str, list[str]]:
        """Extract entity types and sample entities."""
        entity_types: dict[str, list[str]] = {}

        # Get person entities
        person_query = """
        MATCH (c:Concept)
        WHERE c.type IN ['person', 'персона', 'сотрудник', 'человек']
        RETURN c.name as name
        ORDER BY c.activation_count DESC
        LIMIT 20
        """
        persons = await self.db.execute_query(person_query)
        if persons:
            entity_types["люди"] = [p["name"] for p in persons]

        # Get product/service entities
        product_query = """
        MATCH (c:Concept)
        WHERE c.type IN ['product', 'service', 'продукт', 'сервис', 'система', 'tool']
        RETURN c.name as name
        ORDER BY c.activation_count DESC
        LIMIT 20
        """
        products = await self.db.execute_query(product_query)
        if products:
            entity_types["продукты/сервисы"] = [p["name"] for p in products]

        # Get process/procedure entities
        process_query = """
        MATCH (c:Concept)
        WHERE c.type IN ['process', 'procedure', 'процесс', 'процедура']
        RETURN c.name as name
        ORDER BY c.activation_count DESC
        LIMIT 20
        """
        processes = await self.db.execute_query(process_query)
        if processes:
            entity_types["процессы"] = [p["name"] for p in processes]

        # Get team/department entities
        team_query = """
        MATCH (c:Concept)
        WHERE c.type IN ['team', 'department', 'команда', 'отдел', 'подразделение']
        RETURN c.name as name
        ORDER BY c.activation_count DESC
        LIMIT 20
        """
        teams = await self.db.execute_query(team_query)
        if teams:
            entity_types["команды/отделы"] = [t["name"] for t in teams]

        return entity_types

    async def _extract_info_types(self) -> list[str]:
        """Extract information types from memory types."""
        query = """
        MATCH (m:SemanticMemory)
        WHERE m.memory_type IS NOT NULL
        RETURN m.memory_type as info_type, count(*) as cnt
        ORDER BY cnt DESC
        LIMIT 15
        """
        result = await self.db.execute_query(query)
        return [r["info_type"] for r in result if r["info_type"]]

    async def _extract_key_terms(self) -> dict[str, int]:
        """Extract key terms from concept names by frequency."""
        query = """
        MATCH (c:Concept)
        RETURN c.name as term, coalesce(c.activation_count, 0) as cnt
        ORDER BY cnt DESC
        LIMIT 100
        """
        result = await self.db.execute_query(query)
        return {r["term"]: r["cnt"] for r in result if r["term"]}


class LLMKBSummaryEnhancer:
    """Enhances KB summary using LLM for better query understanding.

    v4.4: Uses enrichment LLM (qwen3:4b) to generate:
    - Domain description ("This KB is about...")
    - Capabilities (what questions can be answered)
    - Limitations (what's NOT covered)
    - Sample questions (HyDE-style examples)
    """

    # LLM prompt in Russian for better quality with Russian content
    ENHANCEMENT_PROMPT = """Проанализируй выборку знаний из базы и создай её описание.

=== ВЫБОРКА ЗНАНИЙ ===
{sampled_memories}

=== СТАТИСТИКА ===
{statistics}

Определи:
1. ДОМЕН: Кратко опиши, о чём эта база знаний (1-2 предложения)
2. ВОЗМОЖНОСТИ: Какие типы вопросов можно задать? (5-7 пунктов)
3. ОГРАНИЧЕНИЯ: Чего точно НЕТ в базе? (2-3 пункта, основываясь на анализе контента)
4. ПРИМЕРЫ_ВОПРОСОВ: Сгенерируй {max_questions} конкретных вопросов, на которые можно ответить

Формат ответа (каждый пункт на новой строке):
DOMAIN|описание домена
CAPABILITY|тип вопросов
CAPABILITY|другой тип вопросов
LIMITATION|чего нет
QUESTION|пример вопроса
QUESTION|другой пример вопроса"""

    def __init__(self, db: Neo4jClient) -> None:
        self.db = db
        self.sample_size = settings.kb_summary_sample_size
        self.max_questions = settings.kb_summary_max_questions

    async def enhance(self, summary: KBSummary) -> KBSummary:
        """Enhance KB summary with LLM-generated fields.

        Args:
            summary: Base KBSummary from Neo4j queries

        Returns:
            Enhanced KBSummary with domain_description, capabilities,
            limitations, and sample_questions
        """
        logger.info("Enhancing KB summary with LLM...")

        try:
            # Sample diverse memories for LLM context
            sampled_memories = await self._sample_memories()

            if not sampled_memories:
                logger.warning("No memories to sample, skipping LLM enhancement")
                return summary

            # Format statistics for prompt
            stats_text = self._format_statistics(summary.statistics)

            # Format sampled memories
            memories_text = "\n".join(
                f"- {m['content'][:200]}..." if len(m.get('content', '')) > 200
                else f"- {m.get('content', '')}"
                for m in sampled_memories
            )

            # Build prompt
            prompt = self.ENHANCEMENT_PROMPT.format(
                sampled_memories=memories_text,
                statistics=stats_text,
                max_questions=self.max_questions,
            )

            # Call enrichment LLM
            from engram.ingestion.llm_client import get_enrichment_llm_client
            llm = get_enrichment_llm_client()

            response = await llm.generate(
                prompt=prompt,
                temperature=0.3,  # Lower for more consistent output
                max_tokens=1024,
            )

            # Parse response
            enhanced = self._parse_response(response, summary)
            logger.info(
                f"KB summary enhanced: domain={bool(enhanced.domain_description)}, "
                f"capabilities={len(enhanced.capabilities)}, "
                f"limitations={len(enhanced.limitations)}, "
                f"sample_questions={len(enhanced.sample_questions)}"
            )
            return enhanced

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}, using base summary")
            return summary

    async def _sample_memories(self) -> list[dict[str, Any]]:
        """Sample diverse memories for LLM context.

        Sampling strategy (total = sample_size):
        - 25% most accessed (popular)
        - 25% highest importance
        - 25% from diverse concept types
        - 25% random for coverage
        """
        sample_per_category = self.sample_size // 4
        all_memories: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # 1. Most accessed memories
        popular_query = """
        MATCH (m:SemanticMemory)
        WHERE m.content IS NOT NULL AND m.content <> ''
        RETURN m.id as id, m.content as content
        ORDER BY coalesce(m.access_count, 0) DESC
        LIMIT $limit
        """
        popular = await self.db.execute_query(popular_query, limit=sample_per_category)
        for m in popular:
            if m["id"] not in seen_ids:
                all_memories.append(m)
                seen_ids.add(m["id"])

        # 2. Highest importance memories
        importance_query = """
        MATCH (m:SemanticMemory)
        WHERE m.content IS NOT NULL AND m.content <> ''
        AND m.id NOT IN $seen
        RETURN m.id as id, m.content as content
        ORDER BY coalesce(m.importance, 0) DESC
        LIMIT $limit
        """
        important = await self.db.execute_query(
            importance_query, limit=sample_per_category, seen=list(seen_ids)
        )
        for m in important:
            if m["id"] not in seen_ids:
                all_memories.append(m)
                seen_ids.add(m["id"])

        # 3. Diverse concept types
        diverse_query = """
        MATCH (c:Concept)-[:RELATES_TO]-(m:SemanticMemory)
        WHERE m.content IS NOT NULL AND m.content <> ''
        AND c.type IS NOT NULL
        AND m.id NOT IN $seen
        WITH c.type as concept_type, collect(DISTINCT {id: m.id, content: m.content})[0..2] as memories
        UNWIND memories as m
        RETURN m.id as id, m.content as content
        LIMIT $limit
        """
        diverse = await self.db.execute_query(
            diverse_query, limit=sample_per_category, seen=list(seen_ids)
        )
        for m in diverse:
            if m["id"] not in seen_ids:
                all_memories.append(m)
                seen_ids.add(m["id"])

        # 4. Random sampling for coverage
        random_query = """
        MATCH (m:SemanticMemory)
        WHERE m.content IS NOT NULL AND m.content <> ''
        AND m.id NOT IN $seen
        RETURN m.id as id, m.content as content
        ORDER BY rand()
        LIMIT $limit
        """
        random_memories = await self.db.execute_query(
            random_query, limit=sample_per_category, seen=list(seen_ids)
        )
        for m in random_memories:
            if m["id"] not in seen_ids:
                all_memories.append(m)
                seen_ids.add(m["id"])

        logger.debug(f"Sampled {len(all_memories)} diverse memories for LLM enhancement")

        # Shuffle to avoid ordering bias
        random.shuffle(all_memories)
        return all_memories[:self.sample_size]

    def _format_statistics(self, stats: dict[str, Any]) -> str:
        """Format statistics for LLM prompt."""
        parts = []
        if stats.get("concept_count"):
            parts.append(f"Концептов: {stats['concept_count']}")
        if stats.get("memory_count"):
            parts.append(f"Фактов: {stats['memory_count']}")
        if stats.get("document_count"):
            parts.append(f"Документов: {stats['document_count']}")
        return "\n".join(parts) if parts else "Статистика недоступна"

    def _parse_response(self, response: str, base_summary: KBSummary) -> KBSummary:
        """Parse LLM response into enhanced summary fields.

        Expected format:
        DOMAIN|description
        CAPABILITY|capability1
        CAPABILITY|capability2
        LIMITATION|limitation1
        QUESTION|question1
        """
        domain_description: str | None = None
        capabilities: list[str] = []
        limitations: list[str] = []
        sample_questions: list[str] = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = line.split("|", 1)
            if len(parts) != 2:
                continue

            tag, value = parts[0].strip().upper(), parts[1].strip()
            if not value:
                continue

            if tag == "DOMAIN":
                domain_description = value
            elif tag == "CAPABILITY":
                capabilities.append(value)
            elif tag == "LIMITATION":
                limitations.append(value)
            elif tag == "QUESTION":
                sample_questions.append(value)

        # Create enhanced summary with all original fields
        return KBSummary(
            domains=base_summary.domains,
            entity_types=base_summary.entity_types,
            info_types=base_summary.info_types,
            out_of_scope=base_summary.out_of_scope,
            key_terms=base_summary.key_terms,
            statistics=base_summary.statistics,
            # Enhanced fields
            domain_description=domain_description,
            capabilities=capabilities[:7],  # Limit to max 7
            limitations=limitations[:5],  # Limit to max 5
            sample_questions=sample_questions[:self.max_questions],
        )


class KBSummaryStore:
    """Stores and retrieves KB summary from Neo4j."""

    NODE_LABEL = "KBSummary"

    def __init__(self, db: Neo4jClient) -> None:
        self.db = db

    async def save(self, summary: KBSummary) -> None:
        """Save KB summary to Neo4j.

        Replaces any existing summary.
        """
        # Delete existing summary
        await self.db.execute_query(f"MATCH (s:{self.NODE_LABEL}) DELETE s")

        # Create new summary node
        data = summary.to_dict()
        query = f"""
        CREATE (s:{self.NODE_LABEL} {{
            domains: $domains,
            entity_types: $entity_types,
            info_types: $info_types,
            out_of_scope: $out_of_scope,
            key_terms: $key_terms,
            statistics: $statistics,
            domain_description: $domain_description,
            capabilities: $capabilities,
            limitations: $limitations,
            sample_questions: $sample_questions,
            created_at: datetime()
        }})
        """
        await self.db.execute_query(
            query,
            domains=data["domains"],
            entity_types=data["entity_types"],
            info_types=data["info_types"],
            out_of_scope=data["out_of_scope"],
            key_terms=data["key_terms"],
            statistics=data["statistics"],
            domain_description=data["domain_description"],
            capabilities=data["capabilities"],
            limitations=data["limitations"],
            sample_questions=data["sample_questions"],
        )
        logger.info("KB summary saved to Neo4j")

    async def load(self) -> KBSummary | None:
        """Load KB summary from Neo4j.

        Returns:
            KBSummary if exists, None otherwise
        """
        query = f"MATCH (s:{self.NODE_LABEL}) RETURN s LIMIT 1"
        result = await self.db.execute_query(query)

        if not result:
            logger.warning("No KB summary found in Neo4j")
            return None

        data = dict(result[0]["s"])
        summary = KBSummary.from_dict(data)
        logger.debug("KB summary loaded from Neo4j")
        return summary

    async def exists(self) -> bool:
        """Check if KB summary exists."""
        query = f"MATCH (s:{self.NODE_LABEL}) RETURN count(s) as cnt"
        result = await self.db.execute_query(query)
        return result and result[0]["cnt"] > 0


async def generate_kb_summary(
    db: Neo4jClient,
    use_llm: bool | None = None,
) -> KBSummary:
    """Convenience function to generate and save KB summary.

    Args:
        db: Neo4j client
        use_llm: Whether to use LLM enhancement. If None, uses config setting.

    Returns:
        Generated (and optionally enhanced) KBSummary
    """
    generator = KBSummaryGenerator(db)
    store = KBSummaryStore(db)

    # Generate base summary from Neo4j
    summary = await generator.generate()

    # Optionally enhance with LLM
    should_use_llm = use_llm if use_llm is not None else settings.kb_summary_use_llm
    if should_use_llm:
        enhancer = LLMKBSummaryEnhancer(db)
        summary = await enhancer.enhance(summary)

    await store.save(summary)

    return summary


async def get_kb_summary(db: Neo4jClient) -> KBSummary | None:
    """Convenience function to get KB summary."""
    store = KBSummaryStore(db)
    return await store.load()
