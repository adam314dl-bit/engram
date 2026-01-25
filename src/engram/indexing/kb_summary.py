"""KB summary generation and storage for query enrichment.

Generates a summary of the knowledge base contents to help with:
- Query understanding (is this in scope?)
- Query rewriting (use domain-specific terms)
- Out-of-scope detection

The summary is computed once after ingestion and stored in Neo4j.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

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

    def to_prompt_text(self, max_tokens: int = 400) -> str:
        """Format summary for inclusion in query enrichment prompts.

        Args:
            max_tokens: Approximate max tokens for the summary

        Returns:
            Formatted text for LLM prompt
        """
        parts = []

        # Domains
        if self.domains:
            parts.append(f"Домены: {', '.join(self.domains[:10])}")

        # Entity types (top 3 per type)
        if self.entity_types:
            entity_lines = []
            for etype, entities in list(self.entity_types.items())[:5]:
                sample = entities[:3]
                if len(entities) > 3:
                    sample_str = f"{', '.join(sample)} и др."
                else:
                    sample_str = ', '.join(sample)
                entity_lines.append(f"  - {etype}: {sample_str}")
            if entity_lines:
                parts.append("Типы сущностей:\n" + "\n".join(entity_lines))

        # Info types
        if self.info_types:
            parts.append(f"Типы информации: {', '.join(self.info_types[:8])}")

        # Key terms (top 15)
        if self.key_terms:
            top_terms = sorted(self.key_terms.items(), key=lambda x: x[1], reverse=True)[:15]
            terms_str = ", ".join(t[0] for t in top_terms)
            parts.append(f"Ключевые термины: {terms_str}")

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

        return cls(
            domains=data.get("domains", []),
            entity_types=entity_types,
            info_types=data.get("info_types", []),
            out_of_scope=data.get("out_of_scope", []),
            key_terms=key_terms,
            statistics=statistics,
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


async def generate_kb_summary(db: Neo4jClient) -> KBSummary:
    """Convenience function to generate and save KB summary."""
    generator = KBSummaryGenerator(db)
    store = KBSummaryStore(db)

    summary = await generator.generate()
    await store.save(summary)

    return summary


async def get_kb_summary(db: Neo4jClient) -> KBSummary | None:
    """Convenience function to get KB summary."""
    store = KBSummaryStore(db)
    return await store.load()
