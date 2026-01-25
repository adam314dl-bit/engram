"""Query understanding and enrichment pipeline for v4.3.

Transforms vague user queries into retrieval-optimized versions using:
1. Query understanding (type, complexity, entity detection)
2. BM25 expansion (synonyms, transliteration, domain terms)
3. Semantic rewrite (intent clarification)
4. Optional HyDE (hypothetical document embedding)

The KB summary is used ONLY during query enrichment, NOT during answer generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from engram.config import settings
from engram.indexing.kb_summary import KBSummary, get_kb_summary
from engram.ingestion.llm_client import LLMClient, get_enrichment_llm_client
from engram.preprocessing.russian import (
    get_word_forms,
    lemmatize_word,
    tokenize,
)
from engram.preprocessing.transliteration import expand_query_transliteration
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type classification."""

    FACTUAL = "factual"  # Who, what, when, where
    PROCEDURAL = "procedural"  # How to, steps
    PERSON = "person"  # About a person/role
    COMPARISON = "comparison"  # Compare X vs Y
    TROUBLESHOOTING = "troubleshooting"  # Error, problem, fix
    NAVIGATION = "navigation"  # Where to find, location
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"  # Single fact lookup
    MULTI_HOP = "multi_hop"  # Requires combining facts
    AMBIGUOUS = "ambiguous"  # Unclear intent
    OUT_OF_SCOPE = "out_of_scope"  # Not in KB


@dataclass
class QueryUnderstanding:
    """Result of query analysis."""

    query_type: QueryType
    complexity: QueryComplexity
    entities: list[str] = field(default_factory=list)  # Detected entities
    needs_clarification: bool = False
    clarification_question: str | None = None
    confidence: float = 0.8

    @property
    def is_out_of_scope(self) -> bool:
        """Check if query is out of scope."""
        return self.complexity == QueryComplexity.OUT_OF_SCOPE


@dataclass
class EnrichedQuery:
    """Enriched query with multiple retrieval variants."""

    original: str
    bm25_expanded: str | None = None  # Expanded for BM25 (synonyms, lemmas)
    semantic_rewrite: str | None = None  # Clarified intent for vector search
    hyde_document: str | None = None  # Hypothetical document for HyDE

    # Metadata
    query_type: QueryType = QueryType.UNKNOWN
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    understanding: QueryUnderstanding | None = None

    def get_all_variants(self) -> list[str]:
        """Get all query variants (deduplicated)."""
        variants = [self.original]

        if self.bm25_expanded and self.bm25_expanded != self.original:
            variants.append(self.bm25_expanded)

        if self.semantic_rewrite and self.semantic_rewrite not in variants:
            variants.append(self.semantic_rewrite)

        if self.hyde_document and self.hyde_document not in variants:
            variants.append(self.hyde_document)

        # Limit to max variants
        return variants[:settings.query_enrichment_max_variants]

    def get_variant_labels(self) -> dict[str, str]:
        """Get labeled variants for debugging."""
        labels = {"original": self.original}

        if self.bm25_expanded:
            labels["bm25_expanded"] = self.bm25_expanded

        if self.semantic_rewrite:
            labels["semantic_rewrite"] = self.semantic_rewrite

        if self.hyde_document:
            labels["hyde"] = self.hyde_document

        return labels


# Query type patterns
QUERY_TYPE_PATTERNS: dict[QueryType, list[str]] = {
    QueryType.FACTUAL: [
        r"(кто|что|где|когда|какой|какая|какое|какие)\s+",
        r"(who|what|where|when|which)\s+",
    ],
    QueryType.PROCEDURAL: [
        r"(как|как\s+сделать|как\s+настроить|инструкция)",
        r"(how\s+to|how\s+do\s+i|steps?\s+to)",
    ],
    QueryType.PERSON: [
        r"(кто\s+такой|кто\s+такая|кто\s+отвечает|кто\s+занимается)",
        r"(контакт|телефон|email|почта)\s+",
    ],
    QueryType.COMPARISON: [
        r"(сравн|отлич|разниц|vs|versus|или\s+лучше)",
        r"(compare|difference|vs)",
    ],
    QueryType.TROUBLESHOOTING: [
        r"(ошибк|error|не\s+работает|проблем|failed|issue)",
        r"(fix|solve|debug|troubleshoot)",
    ],
    QueryType.NAVIGATION: [
        r"(где\s+найти|где\s+находится|куда|как\s+попасть)",
        r"(where\s+to\s+find|location\s+of)",
    ],
}


QUERY_UNDERSTANDING_PROMPT = """Проанализируй запрос пользователя для поиска в базе знаний.

База знаний содержит:
{kb_summary}

Запрос: {query}

Определи:
1. Тип запроса: factual (факт), procedural (инструкция), person (человек/контакт), comparison (сравнение), troubleshooting (проблема), navigation (где найти)
2. Сложность: simple (один факт), multi_hop (нужно объединить факты), ambiguous (неясно), out_of_scope (нет в базе)
3. Упомянутые сущности (люди, продукты, процессы)
4. Нужно ли уточнение

Верни в формате:
TYPE|тип|сложность|сущности через запятую|needs_clarification:да/нет|уточняющий вопрос

Пример:
TYPE|person|simple|Иванов,DevOps|needs_clarification:нет|

Ответ:"""


class QueryUnderstandingModule:
    """Analyzes query intent and extracts entities."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        kb_summary: KBSummary | None = None,
    ) -> None:
        self.llm = llm_client or get_enrichment_llm_client()
        self.kb_summary = kb_summary

    async def analyze(self, query: str) -> QueryUnderstanding:
        """Analyze query to understand intent and complexity.

        Args:
            query: User query

        Returns:
            QueryUnderstanding with type, complexity, entities
        """
        # First try fast pattern-based classification
        query_type = self._pattern_classify(query)

        # Use LLM for more detailed analysis
        try:
            kb_text = ""
            if self.kb_summary:
                kb_text = self.kb_summary.to_prompt_text(max_tokens=300)
            else:
                kb_text = "(информация о базе знаний недоступна)"

            prompt = QUERY_UNDERSTANDING_PROMPT.format(
                kb_summary=kb_text,
                query=query,
            )

            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
            )

            understanding = self._parse_understanding_response(response, query_type)
            logger.debug(
                f"Query understanding: type={understanding.query_type.value}, "
                f"complexity={understanding.complexity.value}"
            )
            return understanding

        except Exception as e:
            logger.warning(f"Query understanding failed: {e}")
            # Fallback to pattern-based result
            return QueryUnderstanding(
                query_type=query_type,
                complexity=QueryComplexity.SIMPLE,
                confidence=0.5,
            )

    def _pattern_classify(self, query: str) -> QueryType:
        """Fast pattern-based query type classification."""
        import re

        query_lower = query.lower()

        for qtype, patterns in QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return qtype

        return QueryType.UNKNOWN

    def _parse_understanding_response(
        self,
        text: str,
        fallback_type: QueryType,
    ) -> QueryUnderstanding:
        """Parse LLM understanding response."""
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("TYPE|"):
                parts = line.split("|")
                if len(parts) >= 5:
                    # Parse type
                    type_str = parts[1].strip().lower()
                    try:
                        query_type = QueryType(type_str)
                    except ValueError:
                        query_type = fallback_type

                    # Parse complexity
                    complexity_str = parts[2].strip().lower()
                    try:
                        complexity = QueryComplexity(complexity_str)
                    except ValueError:
                        complexity = QueryComplexity.SIMPLE

                    # Parse entities
                    entities_str = parts[3].strip()
                    entities = [e.strip() for e in entities_str.split(",") if e.strip()]

                    # Parse clarification
                    clarify_str = parts[4].strip().lower()
                    needs_clarification = "да" in clarify_str or "yes" in clarify_str

                    clarification_question = None
                    if len(parts) > 5 and needs_clarification:
                        clarification_question = parts[5].strip()

                    return QueryUnderstanding(
                        query_type=query_type,
                        complexity=complexity,
                        entities=entities,
                        needs_clarification=needs_clarification,
                        clarification_question=clarification_question,
                        confidence=0.8,
                    )

        return QueryUnderstanding(
            query_type=fallback_type,
            complexity=QueryComplexity.SIMPLE,
            confidence=0.5,
        )


class BM25Expander:
    """Expands query for better BM25 retrieval."""

    def __init__(self, kb_summary: KBSummary | None = None) -> None:
        self.kb_summary = kb_summary

    def expand(self, query: str, understanding: QueryUnderstanding | None = None) -> str:
        """Expand query with synonyms, lemmas, and domain terms.

        Args:
            query: Original query
            understanding: Optional query understanding

        Returns:
            Expanded query string
        """
        terms = []
        words = tokenize(query)

        for word in words:
            # Add original
            terms.append(word)

            # Add lemma
            lemma = lemmatize_word(word)
            if lemma != word:
                terms.append(lemma)

            # Add word forms (for important terms)
            if len(word) > 4:
                forms = get_word_forms(word)
                # Add top 2 additional forms
                for form in forms[:2]:
                    if form not in terms:
                        terms.append(form)

        # Add transliteration variants
        translit_variants = expand_query_transliteration(query)
        for variant in translit_variants:
            if variant != query:
                variant_words = tokenize(variant)
                for w in variant_words:
                    if w not in terms:
                        terms.append(w)

        # Add detected entities
        if understanding and understanding.entities:
            for entity in understanding.entities[:3]:
                entity_words = tokenize(entity.lower())
                for w in entity_words:
                    if w not in terms:
                        terms.append(w)

        # Add KB key terms that appear related
        if self.kb_summary and self.kb_summary.key_terms:
            query_lower = query.lower()
            for term in list(self.kb_summary.key_terms.keys())[:50]:
                term_lower = term.lower()
                # Add if term appears in query or shares a root
                if term_lower in query_lower or any(term_lower.startswith(w[:4]) for w in words if len(w) > 3):
                    if term_lower not in terms:
                        terms.append(term_lower)

        # Deduplicate and join
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return " ".join(unique_terms)


SEMANTIC_REWRITE_PROMPT = """Перепиши запрос пользователя для улучшения семантического поиска.

База знаний содержит:
{kb_summary}

Исходный запрос: {query}

Правила:
1. Сохрани смысл запроса
2. Добавь контекст, если запрос слишком краткий
3. Используй термины из базы знаний
4. Сделай запрос более конкретным

Перепишенный запрос (только текст, без пояснений):"""


class SemanticRewriter:
    """Rewrites query for better semantic/vector retrieval."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        kb_summary: KBSummary | None = None,
    ) -> None:
        self.llm = llm_client or get_enrichment_llm_client()
        self.kb_summary = kb_summary

    async def rewrite(self, query: str, understanding: QueryUnderstanding | None = None) -> str:
        """Rewrite query for semantic search.

        Args:
            query: Original query
            understanding: Optional query understanding

        Returns:
            Rewritten query
        """
        # Skip if query is already specific enough
        if len(query.split()) > 10:
            return query

        try:
            kb_text = ""
            if self.kb_summary:
                kb_text = self.kb_summary.to_prompt_text(max_tokens=250)
            else:
                kb_text = "(информация о базе знаний недоступна)"

            prompt = SEMANTIC_REWRITE_PROMPT.format(
                kb_summary=kb_text,
                query=query,
            )

            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=256,
            )

            rewritten = response.strip()

            # Validate rewrite
            if len(rewritten) < 5 or len(rewritten) > len(query) * 3:
                return query

            logger.debug(f"Semantic rewrite: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"Semantic rewrite failed: {e}")
            return query


HYDE_PROMPT = """Напиши гипотетический ответ на вопрос, как если бы ты нашёл информацию в базе знаний.

База знаний содержит:
{kb_summary}

Вопрос: {query}

Напиши краткий (2-3 предложения) ответ в стиле документации. Не выдумывай конкретные имена или числа.

Гипотетический ответ:"""


class HyDEGenerator:
    """Generates hypothetical documents for HyDE retrieval."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        kb_summary: KBSummary | None = None,
    ) -> None:
        self.llm = llm_client or get_enrichment_llm_client()
        self.kb_summary = kb_summary

    async def generate(self, query: str, understanding: QueryUnderstanding | None = None) -> str | None:
        """Generate hypothetical document for HyDE.

        Only generates for complex/multi-hop queries.

        Args:
            query: Original query
            understanding: Optional query understanding

        Returns:
            Hypothetical document or None if not applicable
        """
        # Only use HyDE for complex queries
        if understanding and understanding.complexity not in (
            QueryComplexity.MULTI_HOP,
            QueryComplexity.AMBIGUOUS,
        ):
            return None

        try:
            kb_text = ""
            if self.kb_summary:
                kb_text = self.kb_summary.to_prompt_text(max_tokens=200)
            else:
                kb_text = "(информация о базе знаний недоступна)"

            prompt = HYDE_PROMPT.format(
                kb_summary=kb_text,
                query=query,
            )

            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=256,
            )

            hyde_doc = response.strip()

            # Validate
            if len(hyde_doc) < 20:
                return None

            logger.debug(f"HyDE document: {hyde_doc[:100]}...")
            return hyde_doc

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return None


class QueryEnrichmentPipeline:
    """Orchestrates the full query enrichment pipeline.

    Runs BM25 expansion and semantic rewrite in parallel.
    """

    def __init__(
        self,
        db: Neo4jClient,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.db = db
        self.llm = llm_client or get_enrichment_llm_client()
        self.kb_summary: KBSummary | None = None

        # Components (initialized lazily)
        self._understanding: QueryUnderstandingModule | None = None
        self._bm25_expander: BM25Expander | None = None
        self._semantic_rewriter: SemanticRewriter | None = None
        self._hyde_generator: HyDEGenerator | None = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pipeline with KB summary."""
        if self._initialized:
            return

        # Load KB summary
        self.kb_summary = await get_kb_summary(self.db)
        if not self.kb_summary:
            logger.warning("No KB summary found, enrichment will be limited")

        # Initialize components
        self._understanding = QueryUnderstandingModule(
            llm_client=self.llm,
            kb_summary=self.kb_summary,
        )
        self._bm25_expander = BM25Expander(kb_summary=self.kb_summary)
        self._semantic_rewriter = SemanticRewriter(
            llm_client=self.llm,
            kb_summary=self.kb_summary,
        )
        self._hyde_generator = HyDEGenerator(
            llm_client=self.llm,
            kb_summary=self.kb_summary,
        )

        self._initialized = True
        logger.info("Query enrichment pipeline initialized")

    async def enrich(self, query: str) -> EnrichedQuery:
        """Enrich query with multiple retrieval variants.

        Args:
            query: Original user query

        Returns:
            EnrichedQuery with all variants
        """
        if not self._initialized:
            await self.initialize()

        # 1. Query understanding (serial, needed for other steps)
        understanding = await self._understanding.analyze(query)  # type: ignore

        # Handle out-of-scope or clarification needed
        if understanding.is_out_of_scope:
            return EnrichedQuery(
                original=query,
                query_type=understanding.query_type,
                complexity=understanding.complexity,
                understanding=understanding,
            )

        # 2. Run BM25 expansion and semantic rewrite in parallel
        bm25_task = asyncio.create_task(
            asyncio.to_thread(
                self._bm25_expander.expand, query, understanding  # type: ignore
            )
        )

        semantic_task = asyncio.create_task(
            self._semantic_rewriter.rewrite(query, understanding)  # type: ignore
        )

        # Optional: HyDE for complex queries
        hyde_task = None
        if settings.query_enrichment_use_hyde and understanding.complexity == QueryComplexity.MULTI_HOP:
            hyde_task = asyncio.create_task(
                self._hyde_generator.generate(query, understanding)  # type: ignore
            )

        # Wait for all tasks
        bm25_expanded = await bm25_task
        semantic_rewrite = await semantic_task

        hyde_document = None
        if hyde_task:
            hyde_document = await hyde_task

        logger.info(
            f"Query enriched: {len(query)} chars -> "
            f"BM25: {len(bm25_expanded)} chars, "
            f"semantic: {len(semantic_rewrite)} chars"
            + (f", HyDE: {len(hyde_document)} chars" if hyde_document else "")
        )

        return EnrichedQuery(
            original=query,
            bm25_expanded=bm25_expanded,
            semantic_rewrite=semantic_rewrite,
            hyde_document=hyde_document,
            query_type=understanding.query_type,
            complexity=understanding.complexity,
            understanding=understanding,
        )


async def enrich_query(
    db: Neo4jClient,
    query: str,
) -> EnrichedQuery:
    """Convenience function for query enrichment."""
    pipeline = QueryEnrichmentPipeline(db=db)
    await pipeline.initialize()
    return await pipeline.enrich(query)
