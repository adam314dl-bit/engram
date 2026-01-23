"""CRAG (Corrective RAG) document grading.

Grades retrieved documents as CORRECT, INCORRECT, or AMBIGUOUS before generation.
If all documents are irrelevant, triggers query rewrite for re-retrieval.

Based on: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.hybrid_search import ScoredMemory

logger = logging.getLogger(__name__)


class RetrievalQuality(str, Enum):
    """Overall quality of retrieved documents."""

    CORRECT = "correct"  # Majority relevant
    INCORRECT = "incorrect"  # All irrelevant
    AMBIGUOUS = "ambiguous"  # Mixed relevance


class DocumentRelevance(str, Enum):
    """Relevance of a single document."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


@dataclass
class DocumentGrade:
    """Grade for a single document."""

    memory_id: str
    relevance: DocumentRelevance
    reasoning: str


@dataclass
class CRAGResult:
    """Result of CRAG evaluation."""

    quality: RetrievalQuality
    grades: list[DocumentGrade]
    relevant_ids: list[str]
    irrelevant_ids: list[str]
    relevant_ratio: float
    rewritten_query: str | None = None  # If query rewrite was triggered
    corrective_actions: list[str] = field(default_factory=list)

    @property
    def should_proceed(self) -> bool:
        """Check if we have enough relevant docs to proceed."""
        return self.quality != RetrievalQuality.INCORRECT

    @property
    def needs_rewrite(self) -> bool:
        """Check if query rewrite is needed."""
        return self.quality == RetrievalQuality.INCORRECT


DOCUMENT_GRADING_PROMPT = """Оцени релевантность документа для ответа на вопрос.

Вопрос: {query}

Документ:
{document}

Релевантен ли документ для ответа на вопрос? Отвечай только "да" или "нет".

Ответ:"""


QUERY_REWRITE_PROMPT = """Перефразируй вопрос для улучшения поиска.

Оригинальный вопрос: {query}

Найденные документы оказались нерелевантными. Перефразируй вопрос,
чтобы найти более подходящую информацию. Используй синонимы,
уточни термины, или разбей на подвопросы.

Новый вопрос:"""


class CRAGEvaluator:
    """
    CRAG evaluator for document grading.

    Grades each document as relevant/irrelevant using binary yes/no classification.
    Triggers query rewrite if all documents are irrelevant.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        min_relevant_ratio: float | None = None,
        rewrite_on_failure: bool | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.min_relevant_ratio = (
            min_relevant_ratio
            if min_relevant_ratio is not None
            else settings.crag_min_relevant_ratio
        )
        self.rewrite_on_failure = (
            rewrite_on_failure
            if rewrite_on_failure is not None
            else settings.crag_rewrite_on_failure
        )

    async def evaluate(
        self,
        query: str,
        memories: list[ScoredMemory],
    ) -> CRAGResult:
        """
        Evaluate retrieved documents for relevance.

        Args:
            query: User query
            memories: Retrieved memories to evaluate

        Returns:
            CRAGResult with grades and corrective actions
        """
        if not memories:
            return CRAGResult(
                quality=RetrievalQuality.INCORRECT,
                grades=[],
                relevant_ids=[],
                irrelevant_ids=[],
                relevant_ratio=0.0,
                corrective_actions=["No documents retrieved"],
            )

        # Grade each document
        grades: list[DocumentGrade] = []
        relevant_ids: list[str] = []
        irrelevant_ids: list[str] = []

        for sm in memories:
            grade = await self._grade_document(query, sm)
            grades.append(grade)

            if grade.relevance == DocumentRelevance.RELEVANT:
                relevant_ids.append(grade.memory_id)
            else:
                irrelevant_ids.append(grade.memory_id)

        # Calculate relevance ratio
        relevant_ratio = len(relevant_ids) / len(memories) if memories else 0.0

        # Determine overall quality
        if relevant_ratio >= self.min_relevant_ratio:
            quality = RetrievalQuality.CORRECT
            corrective_actions = []
        elif relevant_ratio > 0:
            quality = RetrievalQuality.AMBIGUOUS
            corrective_actions = ["Some documents irrelevant, filtering results"]
        else:
            quality = RetrievalQuality.INCORRECT
            corrective_actions = ["All documents irrelevant"]

        logger.info(
            f"CRAG evaluation: {quality.value}, "
            f"{len(relevant_ids)}/{len(memories)} relevant ({relevant_ratio:.1%})"
        )

        # Try query rewrite if all documents are irrelevant
        rewritten_query = None
        if quality == RetrievalQuality.INCORRECT and self.rewrite_on_failure:
            rewritten_query = await self._rewrite_query(query)
            corrective_actions.append(f"Query rewritten: {rewritten_query}")

        return CRAGResult(
            quality=quality,
            grades=grades,
            relevant_ids=relevant_ids,
            irrelevant_ids=irrelevant_ids,
            relevant_ratio=relevant_ratio,
            rewritten_query=rewritten_query,
            corrective_actions=corrective_actions,
        )

    async def _grade_document(
        self,
        query: str,
        memory: ScoredMemory,
    ) -> DocumentGrade:
        """Grade a single document for relevance."""
        try:
            prompt = DOCUMENT_GRADING_PROMPT.format(
                query=query,
                document=memory.memory.content[:1000],  # Limit for speed
            )

            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.0,  # Deterministic
                max_tokens=16,
            )

            response_lower = response.lower().strip()

            if "да" in response_lower or "yes" in response_lower:
                relevance = DocumentRelevance.RELEVANT
                reasoning = "Document contains relevant information"
            else:
                relevance = DocumentRelevance.IRRELEVANT
                reasoning = "Document not relevant to query"

            return DocumentGrade(
                memory_id=memory.memory.id,
                relevance=relevance,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning(f"Document grading failed: {e}")
            # Default to relevant on error to avoid over-filtering
            return DocumentGrade(
                memory_id=memory.memory.id,
                relevance=DocumentRelevance.RELEVANT,
                reasoning=f"Grading failed, assuming relevant: {e}",
            )

    async def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval."""
        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            rewritten = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=256,
            )
            return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query  # Return original on failure

    def filter_relevant(
        self,
        memories: list[ScoredMemory],
        crag_result: CRAGResult,
    ) -> list[ScoredMemory]:
        """Filter memories to only include relevant ones."""
        relevant_set = set(crag_result.relevant_ids)
        return [m for m in memories if m.memory.id in relevant_set]


async def evaluate_retrieval(
    query: str,
    memories: list[ScoredMemory],
    llm_client: LLMClient | None = None,
) -> CRAGResult:
    """Convenience function for CRAG evaluation."""
    evaluator = CRAGEvaluator(llm_client=llm_client)
    return await evaluator.evaluate(query, memories)
