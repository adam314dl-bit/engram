"""Person and role extraction for organizational queries.

Extracts people, their roles, and team affiliations from documents
to answer queries like "кто тимлид?" or "чем занимается Артур?".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import Natasha for Russian NER
try:
    from natasha import (
        Doc,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        NewsSyntaxParser,
        PER,
        Segmenter,
    )
    HAS_NATASHA = True
except ImportError:
    HAS_NATASHA = False
    logger.info("Natasha not available, using regex-based person extraction")


class PersonQueryType(str, Enum):
    """Types of person-related queries."""

    WHO_IS_ROLE = "who_is_role"  # "кто тимлид?"
    WHAT_DOES_PERSON_DO = "what_does_person_do"  # "чем занимается Артур?"
    PERSON_CONTACTS = "person_contacts"  # "контакты Артура"
    TEAM_MEMBERS = "team_members"  # "кто в команде X?"
    GENERAL = "general"  # Not a person query


@dataclass
class Person:
    """Extracted person with their attributes."""

    canonical_name: str  # Primary name form
    variants: list[str] = field(default_factory=list)  # Name variants
    role: str | None = None  # Job title/role
    team: str | None = None  # Team name
    responsibilities: list[str] = field(default_factory=list)
    contacts: dict[str, str] = field(default_factory=dict)  # email, telegram, etc.
    source_doc_ids: list[str] = field(default_factory=list)


@dataclass
class PersonExtractionResult:
    """Result of person extraction from a document."""

    persons: list[Person]
    person_role_pairs: list[tuple[str, str]]  # (name, role) pairs


# Russian patterns for person-role extraction
ROLE_PATTERNS = [
    # "Иван Петров — тимлид"
    r"([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s*[—–-]\s*(тимлид|лид|руководитель|менеджер|директор|архитектор|разработчик|аналитик|дизайнер|тестировщик|девопс|SRE|PM|PO|CTO|CEO|CFO|COO|HR|QA)[а-яё]*",
    # "тимлид: Иван Петров" or "тимлид — Иван Петров"
    r"(тимлид|лид|руководитель|менеджер|директор|архитектор|разработчик|аналитик|дизайнер|тестировщик|девопс|SRE|PM|PO|CTO|CEO|CFO|COO|HR|QA)[а-яё]*\s*[:—–-]\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)",
    # "Иван Петров (тимлид)"
    r"([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s*\((тимлид|лид|руководитель|менеджер|директор|архитектор|разработчик|аналитик|дизайнер|тестировщик|девопс|SRE|PM|PO|CTO|CEO|CFO|COO|HR|QA)[а-яё]*\)",
]

# Patterns for team membership
TEAM_PATTERNS = [
    # "команда X: Иван, Петр, Мария"
    r"команда\s+([а-яёА-ЯЁa-zA-Z0-9_-]+):\s*([А-ЯЁ][а-яё]+(?:,\s*[А-ЯЁ][а-яё]+)*)",
    # "в команде X работают: Иван, Петр"
    r"в\s+команде\s+([а-яёА-ЯЁa-zA-Z0-9_-]+)\s+работа[юе]т[а-я]*:\s*([А-ЯЁ][а-яё]+(?:,\s*[А-ЯЁ][а-яё]+)*)",
]

# Query classification patterns
QUERY_WHO_IS_ROLE = [
    r"кто\s+(тимлид|лид|руководитель|менеджер|директор|архитектор|PM|PO|CTO|CEO)",
    r"кто\s+является\s+(тимлид|лид|руководител|менеджер|директор)[а-яё]*",
    r"кто\s+занимает\s+должность",
    r"кто\s+отвечает\s+за",
]

QUERY_WHAT_DOES_PERSON = [
    r"(?:чем|что)\s+занимается\s+([А-ЯЁ][а-яё]+)",
    r"(?:какая|какие)\s+(?:роль|обязанности)\s+(?:у\s+)?([А-ЯЁ][а-яё]+)",
    r"([А-ЯЁ][а-яё]+)\s+отвечает\s+за",
]

QUERY_PERSON_CONTACTS = [
    r"контакт[ыа]?\s+([А-ЯЁ][а-яё]+)",
    r"(?:как\s+связаться|написать)\s+([А-ЯЁ][а-яё]+)",
    r"(?:email|почта|телеграм|telegram)\s+([А-ЯЁ][а-яё]+)",
]

QUERY_TEAM_MEMBERS = [
    r"кто\s+в\s+команде\s+([а-яёА-ЯЁa-zA-Z0-9_-]+)",
    r"состав\s+команды\s+([а-яёА-ЯЁa-zA-Z0-9_-]+)",
    r"члены\s+команды\s+([a-яёА-ЯЁa-zA-Z0-9_-]+)",
]


class RussianNameNormalizer:
    """Normalize Russian names to canonical form."""

    # Common name diminutives/variants
    NAME_VARIANTS: dict[str, list[str]] = {
        "Александр": ["Саша", "Саня", "Шура", "Алекс"],
        "Алексей": ["Лёша", "Алёша", "Лёха"],
        "Анастасия": ["Настя", "Ася"],
        "Андрей": ["Андрюша", "Андрюха"],
        "Анна": ["Аня", "Анюта", "Нюра"],
        "Артём": ["Артур", "Тёма"],
        "Борис": ["Боря"],
        "Владимир": ["Вова", "Володя", "Влад"],
        "Виктор": ["Витя"],
        "Виктория": ["Вика"],
        "Дмитрий": ["Дима", "Митя"],
        "Евгений": ["Женя"],
        "Евгения": ["Женя"],
        "Екатерина": ["Катя", "Катюша"],
        "Елена": ["Лена"],
        "Иван": ["Ваня"],
        "Игорь": ["Гоша"],
        "Илья": ["Илюша"],
        "Ирина": ["Ира"],
        "Максим": ["Макс"],
        "Мария": ["Маша"],
        "Михаил": ["Миша"],
        "Наталья": ["Наташа"],
        "Николай": ["Коля"],
        "Олег": ["Олежка"],
        "Ольга": ["Оля"],
        "Павел": ["Паша"],
        "Пётр": ["Петя"],
        "Роман": ["Рома"],
        "Светлана": ["Света"],
        "Сергей": ["Серёжа", "Серёга"],
        "Татьяна": ["Таня"],
        "Юлия": ["Юля"],
        "Юрий": ["Юра"],
    }

    # Reverse mapping: variant -> canonical
    _VARIANT_TO_CANONICAL: dict[str, str] = {}

    @classmethod
    def _build_reverse_mapping(cls) -> None:
        """Build reverse mapping on first use."""
        if not cls._VARIANT_TO_CANONICAL:
            for canonical, variants in cls.NAME_VARIANTS.items():
                for variant in variants:
                    cls._VARIANT_TO_CANONICAL[variant.lower()] = canonical
                cls._VARIANT_TO_CANONICAL[canonical.lower()] = canonical

    @classmethod
    def normalize(cls, name: str) -> str:
        """
        Normalize a name to its canonical form.

        Args:
            name: Name to normalize

        Returns:
            Canonical name form
        """
        cls._build_reverse_mapping()

        # Split name into parts
        parts = name.strip().split()
        normalized_parts = []

        for part in parts:
            # Check if it's a known variant
            canonical = cls._VARIANT_TO_CANONICAL.get(part.lower())
            if canonical:
                normalized_parts.append(canonical)
            else:
                # Keep original with proper capitalization
                normalized_parts.append(part.capitalize())

        return " ".join(normalized_parts)

    @classmethod
    def get_variants(cls, name: str) -> list[str]:
        """
        Get all known variants for a name.

        Args:
            name: Name to get variants for

        Returns:
            List of name variants
        """
        cls._build_reverse_mapping()

        # Normalize first
        canonical = cls.normalize(name)
        first_name = canonical.split()[0] if canonical else ""

        variants = [canonical]

        # Add diminutives
        if first_name in cls.NAME_VARIANTS:
            for variant in cls.NAME_VARIANTS[first_name]:
                full_variant = canonical.replace(first_name, variant, 1)
                if full_variant not in variants:
                    variants.append(full_variant)

        return variants


class PersonExtractor:
    """Extract person entities and their roles from text."""

    def __init__(self, use_natasha: bool = True) -> None:
        """
        Initialize person extractor.

        Args:
            use_natasha: Whether to use Natasha NER (if available)
        """
        self.use_natasha = use_natasha and HAS_NATASHA
        self._natasha_initialized = False
        self._segmenter = None
        self._morph_vocab = None
        self._emb = None
        self._morph_tagger = None
        self._syntax_parser = None
        self._ner_tagger = None

    def _init_natasha(self) -> None:
        """Lazy initialization of Natasha components."""
        if self._natasha_initialized or not self.use_natasha:
            return

        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)
        self._syntax_parser = NewsSyntaxParser(self._emb)
        self._ner_tagger = NewsNERTagger(self._emb)
        self._natasha_initialized = True

    def extract(self, text: str, doc_id: str | None = None) -> PersonExtractionResult:
        """
        Extract persons and their roles from text.

        Args:
            text: Document text
            doc_id: Source document ID

        Returns:
            PersonExtractionResult with extracted persons
        """
        persons: dict[str, Person] = {}  # canonical_name -> Person
        person_role_pairs: list[tuple[str, str]] = []

        # Extract using Natasha if available
        if self.use_natasha:
            self._init_natasha()
            natasha_persons = self._extract_with_natasha(text)
            for name in natasha_persons:
                canonical = RussianNameNormalizer.normalize(name)
                if canonical not in persons:
                    persons[canonical] = Person(
                        canonical_name=canonical,
                        variants=RussianNameNormalizer.get_variants(canonical),
                        source_doc_ids=[doc_id] if doc_id else [],
                    )

        # Extract using regex patterns
        regex_results = self._extract_with_regex(text)
        for name, role in regex_results:
            canonical = RussianNameNormalizer.normalize(name)
            person_role_pairs.append((canonical, role))

            if canonical not in persons:
                persons[canonical] = Person(
                    canonical_name=canonical,
                    variants=RussianNameNormalizer.get_variants(canonical),
                    role=role,
                    source_doc_ids=[doc_id] if doc_id else [],
                )
            else:
                # Update role if not set
                if not persons[canonical].role and role:
                    persons[canonical].role = role

        # Extract team memberships
        team_members = self._extract_team_members(text)
        for team_name, members in team_members.items():
            for member_name in members:
                canonical = RussianNameNormalizer.normalize(member_name)
                if canonical in persons:
                    persons[canonical].team = team_name
                else:
                    persons[canonical] = Person(
                        canonical_name=canonical,
                        variants=RussianNameNormalizer.get_variants(canonical),
                        team=team_name,
                        source_doc_ids=[doc_id] if doc_id else [],
                    )

        return PersonExtractionResult(
            persons=list(persons.values()),
            person_role_pairs=person_role_pairs,
        )

    def _extract_with_natasha(self, text: str) -> list[str]:
        """Extract person names using Natasha NER."""
        if not self.use_natasha or not self._segmenter:
            return []

        try:
            doc = Doc(text)
            doc.segment(self._segmenter)
            doc.tag_morph(self._morph_tagger)
            doc.tag_ner(self._ner_tagger)

            # Extract PER (person) entities
            persons = []
            for span in doc.spans:
                if span.type == PER:
                    # Normalize the span text
                    span.normalize(self._morph_vocab)
                    persons.append(span.normal or span.text)

            return persons
        except Exception as e:
            logger.warning(f"Natasha extraction failed: {e}")
            return []

    def _extract_with_regex(self, text: str) -> list[tuple[str, str]]:
        """Extract person-role pairs using regex patterns."""
        results: list[tuple[str, str]] = []

        for pattern in ROLE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    # Determine which group is name and which is role
                    # based on pattern structure
                    name, role = groups[0], groups[1]
                    # Swap if role appears before name in pattern
                    if re.match(r"(тимлид|лид|руководитель)", groups[0], re.IGNORECASE):
                        name, role = groups[1], groups[0]
                    results.append((name.strip(), role.strip().lower()))

        return results

    def _extract_team_members(self, text: str) -> dict[str, list[str]]:
        """Extract team membership information."""
        teams: dict[str, list[str]] = {}

        for pattern in TEAM_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches:
                team_name = match.group(1)
                members_str = match.group(2)
                members = [m.strip() for m in members_str.split(",")]
                if team_name not in teams:
                    teams[team_name] = []
                teams[team_name].extend(members)

        return teams


def classify_person_query(query: str) -> tuple[PersonQueryType, str | None]:
    """
    Classify a query to determine if it's person-related.

    Args:
        query: User query

    Returns:
        Tuple of (query_type, extracted_entity)
        - query_type: Type of person query
        - extracted_entity: Name/role/team extracted from query
    """
    query_lower = query.lower().strip()

    # Check "who is role" patterns
    for pattern in QUERY_WHO_IS_ROLE:
        match = re.search(pattern, query_lower)
        if match:
            role = match.group(1) if match.groups() else None
            return (PersonQueryType.WHO_IS_ROLE, role)

    # Check "what does person do" patterns
    for pattern in QUERY_WHAT_DOES_PERSON:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            name = match.group(1) if match.groups() else None
            return (PersonQueryType.WHAT_DOES_PERSON_DO, name)

    # Check person contacts patterns
    for pattern in QUERY_PERSON_CONTACTS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            name = match.group(1) if match.groups() else None
            return (PersonQueryType.PERSON_CONTACTS, name)

    # Check team members patterns
    for pattern in QUERY_TEAM_MEMBERS:
        match = re.search(pattern, query_lower)
        if match:
            team = match.group(1) if match.groups() else None
            return (PersonQueryType.TEAM_MEMBERS, team)

    return (PersonQueryType.GENERAL, None)


def extract_persons_from_text(
    text: str,
    doc_id: str | None = None,
) -> PersonExtractionResult:
    """
    Convenience function to extract persons from text.

    Args:
        text: Document text
        doc_id: Source document ID

    Returns:
        PersonExtractionResult
    """
    extractor = PersonExtractor()
    return extractor.extract(text, doc_id)
