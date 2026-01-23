"""Intent classification for deciding whether to retrieve before generation.

Decides:
1. RETRIEVE - Query needs knowledge base lookup
2. NO_RETRIEVE - Query can be answered directly (greetings, general chat)
3. CLARIFY - Query is ambiguous and needs clarification

Uses pattern-based fast classification with LLM fallback for ambiguous cases.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from engram.config import settings
from engram.ingestion.llm_client import LLMClient, get_llm_client
from engram.retrieval.hybrid_search import classify_query_complexity

logger = logging.getLogger(__name__)


class RetrievalDecision(str, Enum):
    """Decision on whether to retrieve."""

    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    CLARIFY = "clarify"


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class IntentResult:
    """Result of intent classification."""

    decision: RetrievalDecision
    complexity: QueryComplexity
    confidence: float
    reasoning: str
    should_use_ircot: bool = False  # True for complex multi-hop queries

    @property
    def needs_retrieval(self) -> bool:
        """Check if retrieval is needed."""
        return self.decision == RetrievalDecision.RETRIEVE


# Patterns that don't need retrieval (greetings, chitchat, meta)
NO_RETRIEVE_PATTERNS = [
    # Greetings
    r"^(привет|здравствуй|добрый\s+(день|вечер|утро)|hi|hello|hey)\b",
    r"^(пока|до\s+свидания|bye|goodbye)\b",

    # Thanks
    r"^(спасибо|благодарю|thanks|thank\s+you)\b",

    # Meta questions about the assistant
    r"^(кто\s+ты|ты\s+кто|what\s+are\s+you|who\s+are\s+you)",
    r"^(что\s+ты\s+умеешь|что\s+ты\s+можешь)",

    # Simple confirmations
    r"^(да|нет|ок|okay|yes|no|понял|ясно|хорошо)$",

    # Empty or very short
    r"^.{0,3}$",
]

# Patterns that definitely need retrieval
RETRIEVE_PATTERNS = [
    # Factoid questions
    r"(что\s+такое|что\s+это|who\s+is|what\s+is)",
    r"(где\s+находится|where\s+is)",
    r"(когда\s+был|when\s+was|when\s+did)",
    r"(сколько|how\s+many|how\s+much)",

    # How-to questions
    r"(как\s+сделать|как\s+настроить|как\s+установить|how\s+to|how\s+do\s+i)",

    # Troubleshooting
    r"(ошибка|error|не\s+работает|проблема|failed|issue)",

    # Comparison and analysis
    r"(сравни|compare|разница|difference|отличия)",
    r"(преимущества|недостатки|плюсы|минусы|pros|cons)",

    # Explanation requests
    r"(объясни|explain|расскажи|describe|опиши)",

    # List requests
    r"(перечисли|list|покажи\s+все|show\s+all)",

    # Technical queries
    r"(docker|kubernetes|k8s|git|linux|python|api|database|sql)",
]

# Patterns that suggest query needs clarification
CLARIFY_PATTERNS = [
    r"^(это|that|it)\s*$",  # Ambiguous references
    r"^(тоже|также|and|also)\s",  # Continuations without context
    r"^\?\s*$",  # Just a question mark
]


def _pattern_classify(query: str) -> tuple[RetrievalDecision | None, str]:
    """
    Fast pattern-based classification.

    Returns:
        Tuple of (decision or None if ambiguous, reasoning)
    """
    query_lower = query.lower().strip()

    # Check no-retrieve patterns first
    for pattern in NO_RETRIEVE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return RetrievalDecision.NO_RETRIEVE, f"Matches no-retrieve pattern: {pattern}"

    # Check clarify patterns
    for pattern in CLARIFY_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return RetrievalDecision.CLARIFY, f"Matches clarify pattern: {pattern}"

    # Check retrieve patterns
    for pattern in RETRIEVE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return RetrievalDecision.RETRIEVE, f"Matches retrieve pattern: {pattern}"

    # Ambiguous - return None to trigger LLM fallback
    return None, "No pattern match"


INTENT_CLASSIFICATION_PROMPT = """Классифицируй запрос пользователя.

Запрос: {query}

Определи:
1. Нужен ли поиск в базе знаний для ответа?
2. Насколько сложный запрос?

Верни ответ в формате:
INTENT|решение|сложность|обоснование

Где:
- решение: retrieve, no_retrieve, clarify
- сложность: simple, moderate, complex
- обоснование: краткое объяснение

Правила:
- retrieve: запрос требует фактической информации из базы знаний
- no_retrieve: приветствия, благодарности, общий разговор
- clarify: запрос неоднозначен, нужно уточнение
- simple: простой фактоидный вопрос (кто, что, где, когда)
- moderate: стандартный вопрос, требующий объяснения
- complex: сравнение, анализ, многошаговое рассуждение

Пример:
INTENT|retrieve|complex|Вопрос требует информацию из базы знаний

Ответ:"""


class IntentClassifier:
    """
    Intent classifier for deciding whether to retrieve.

    Uses pattern-based fast classification (~1ms) with LLM fallback
    for ambiguous cases (~100-300ms).
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm_fallback: bool | None = None,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.use_llm_fallback = (
            use_llm_fallback
            if use_llm_fallback is not None
            else settings.intent_use_llm_fallback
        )

    async def classify(self, query: str) -> IntentResult:
        """
        Classify query intent.

        Args:
            query: User query text

        Returns:
            IntentResult with decision, complexity, and reasoning
        """
        # Get complexity from existing classifier
        complexity_str, _ = classify_query_complexity(query)
        complexity = QueryComplexity(complexity_str)

        # Try fast pattern-based classification
        decision, reasoning = _pattern_classify(query)

        if decision is not None:
            logger.debug(f"Pattern classification: {decision.value} ({reasoning})")
            return IntentResult(
                decision=decision,
                complexity=complexity,
                confidence=0.9,  # High confidence for pattern matches
                reasoning=reasoning,
                should_use_ircot=complexity == QueryComplexity.COMPLEX,
            )

        # Ambiguous - use LLM fallback if enabled
        if self.use_llm_fallback:
            return await self._llm_classify(query, complexity)

        # Default to retrieve for ambiguous cases
        logger.debug("Ambiguous query, defaulting to retrieve")
        return IntentResult(
            decision=RetrievalDecision.RETRIEVE,
            complexity=complexity,
            confidence=0.5,
            reasoning="Ambiguous query, defaulting to retrieve",
            should_use_ircot=complexity == QueryComplexity.COMPLEX,
        )

    async def _llm_classify(
        self,
        query: str,
        complexity: QueryComplexity,
    ) -> IntentResult:
        """Use LLM for ambiguous cases."""
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
            )

            decision_str, complexity_str, reasoning = self._parse_intent_response(
                response, complexity.value
            )

            decision = RetrievalDecision(decision_str)
            llm_complexity = QueryComplexity(complexity_str)

            logger.debug(f"LLM classification: {decision.value}, {llm_complexity.value}")

            return IntentResult(
                decision=decision,
                complexity=llm_complexity,
                confidence=0.8,
                reasoning=reasoning,
                should_use_ircot=llm_complexity == QueryComplexity.COMPLEX,
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        # Fallback to retrieve
        return IntentResult(
            decision=RetrievalDecision.RETRIEVE,
            complexity=complexity,
            confidence=0.5,
            reasoning="LLM classification failed, defaulting to retrieve",
            should_use_ircot=complexity == QueryComplexity.COMPLEX,
        )

    def _parse_intent_response(
        self,
        text: str,
        default_complexity: str,
    ) -> tuple[str, str, str]:
        """Parse pipe-delimited intent response."""
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("INTENT|"):
                parts = line.split("|")
                if len(parts) >= 4:
                    decision = parts[1].strip().lower()
                    complexity = parts[2].strip().lower()
                    reasoning = parts[3].strip()
                    # Validate decision
                    if decision not in ("retrieve", "no_retrieve", "clarify"):
                        decision = "retrieve"
                    # Validate complexity
                    if complexity not in ("simple", "moderate", "complex"):
                        complexity = default_complexity
                    return decision, complexity, reasoning
                elif len(parts) >= 2:
                    decision = parts[1].strip().lower()
                    if decision not in ("retrieve", "no_retrieve", "clarify"):
                        decision = "retrieve"
                    return decision, default_complexity, "Parsed from LLM"
        return "retrieve", default_complexity, "Fallback - no INTENT line found"


async def classify_intent(
    query: str,
    llm_client: LLMClient | None = None,
) -> IntentResult:
    """Convenience function for intent classification."""
    classifier = IntentClassifier(llm_client=llm_client)
    return await classifier.classify(query)
