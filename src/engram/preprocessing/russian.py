"""Russian NLP preprocessing with PyMorphy3 lemmatization."""

import logging
import re
from functools import lru_cache
from typing import Callable

import pymorphy3

from engram.config import settings

logger = logging.getLogger(__name__)

# Russian stopwords (common words that don't carry semantic meaning)
RUSSIAN_STOPWORDS: frozenset[str] = frozenset([
    # Pronouns
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "меня", "тебя", "его", "её", "нас", "вас", "их",
    "мне", "тебе", "ему", "ей", "нам", "вам", "им",
    "мной", "тобой", "им", "ей", "нами", "вами", "ими",
    "себя", "себе", "собой",
    "мой", "твой", "его", "её", "наш", "ваш", "их",
    "этот", "тот", "такой", "какой", "который", "чей",
    "весь", "всё", "все", "сам", "самый",
    "кто", "что", "никто", "ничто", "некто", "нечто",
    # Prepositions
    "в", "на", "с", "к", "у", "о", "об", "по", "из", "за", "от", "до",
    "под", "над", "при", "через", "после", "перед", "между", "без",
    "для", "про", "около", "вокруг", "среди", "вдоль", "против",
    # Conjunctions
    "и", "а", "но", "или", "да", "ни", "либо", "то",
    "что", "чтобы", "как", "когда", "если", "хотя", "пока", "потому",
    "поэтому", "так", "также", "тоже", "однако", "зато", "ведь",
    # Particles
    "не", "ни", "бы", "же", "ли", "ль", "вот", "вон", "даже",
    "лишь", "только", "уже", "ещё", "еще", "именно", "разве", "неужели",
    # Auxiliary verbs
    "быть", "есть", "был", "была", "было", "были", "будет", "будут",
    "буду", "будем", "будешь", "будете",
    "можно", "нужно", "надо", "нельзя", "стоит",
    # Common adverbs
    "очень", "там", "тут", "здесь", "где", "куда", "откуда", "когда",
    "теперь", "сейчас", "тогда", "потом", "раньше", "позже",
    "как", "так", "почему", "зачем", "сколько",
    # Other common words
    "это", "этого", "этому", "этим", "этом",
    "то", "того", "тому", "тем", "том",
])


@lru_cache(maxsize=1)
def get_morph_analyzer() -> pymorphy3.MorphAnalyzer:
    """Get or create cached PyMorphy3 analyzer (singleton)."""
    logger.info("Initializing PyMorphy3 analyzer...")
    return pymorphy3.MorphAnalyzer()


def lemmatize_word(word: str) -> str:
    """
    Lemmatize a single Russian word using PyMorphy3.

    Args:
        word: Word to lemmatize

    Returns:
        Lemmatized form (normal form) of the word
    """
    morph = get_morph_analyzer()
    parsed = morph.parse(word)
    if parsed:
        return parsed[0].normal_form
    return word


def lemmatize_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Lemmatize Russian text: tokenize, lemmatize each word, optionally remove stopwords.

    Args:
        text: Text to lemmatize
        remove_stopwords: Whether to remove stopwords

    Returns:
        Lemmatized text with words joined by spaces
    """
    words = tokenize(text)
    morph = get_morph_analyzer()

    result = []
    for word in words:
        # Skip stopwords if requested
        if remove_stopwords and word in RUSSIAN_STOPWORDS:
            continue

        # Lemmatize
        parsed = morph.parse(word)
        if parsed:
            lemma = parsed[0].normal_form
            # Skip stopwords after lemmatization too
            if remove_stopwords and lemma in RUSSIAN_STOPWORDS:
                continue
            result.append(lemma)
        else:
            result.append(word)

    return " ".join(result)


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into words (Russian and English).

    Handles Cyrillic and Latin alphabets, removes punctuation.

    Args:
        text: Text to tokenize

    Returns:
        List of lowercase tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Match words (Cyrillic or Latin letters, including hyphenated words)
    # \w includes underscores, so we use explicit character classes
    pattern = r"[а-яёa-z](?:[а-яёa-z\-]*[а-яёa-z])?"
    tokens = re.findall(pattern, text)

    # Remove single-character tokens except important ones
    tokens = [t for t in tokens if len(t) > 1 or t in {"я", "в", "с", "к", "у", "о", "а", "и"}]

    return tokens


def clean_text(text: str) -> str:
    """
    Clean text for processing: normalize whitespace, remove extra punctuation.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove multiple punctuation
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    return text.strip()


def prepare_bm25_query(query: str) -> str:
    """
    Prepare query for BM25 search based on settings.

    Applies lemmatization and stopword removal if configured.

    Args:
        query: Original query text

    Returns:
        Processed query for BM25
    """
    if settings.bm25_lemmatize:
        return lemmatize_text(query, remove_stopwords=settings.bm25_remove_stopwords)
    elif settings.bm25_remove_stopwords:
        words = tokenize(query)
        return " ".join(w for w in words if w not in RUSSIAN_STOPWORDS)
    else:
        return query


def prepare_bm25_document(text: str) -> str:
    """
    Prepare document text for BM25 indexing.

    Same processing as queries for consistency.

    Args:
        text: Document text

    Returns:
        Processed text for BM25 indexing
    """
    return prepare_bm25_query(text)


def get_word_forms(word: str) -> list[str]:
    """
    Get all word forms (inflections) for a given word.

    Useful for expanding search queries.

    Args:
        word: Base word

    Returns:
        List of word forms including the original
    """
    morph = get_morph_analyzer()
    parsed = morph.parse(word)

    if not parsed:
        return [word]

    # Get the primary parse
    primary = parsed[0]

    # Get all forms via lexeme
    forms = set()
    for form in primary.lexeme:
        forms.add(form.word)

    return list(forms)


def is_russian_text(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is primarily Russian.

    Args:
        text: Text to check
        threshold: Minimum ratio of Cyrillic characters

    Returns:
        True if text appears to be Russian
    """
    if not text:
        return False

    # Count Cyrillic and Latin characters
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")

    total = cyrillic + latin
    if total == 0:
        return False

    return cyrillic / total >= threshold


# Type alias for preprocessor function
TextPreprocessor = Callable[[str], str]


def create_preprocessor(
    lemmatize: bool = True,
    remove_stopwords: bool = True,
    clean: bool = True,
) -> TextPreprocessor:
    """
    Create a text preprocessor function with specified options.

    Args:
        lemmatize: Apply lemmatization
        remove_stopwords: Remove stopwords
        clean: Clean text (normalize whitespace, etc.)

    Returns:
        Preprocessor function
    """
    def preprocess(text: str) -> str:
        if clean:
            text = clean_text(text)
        if lemmatize:
            text = lemmatize_text(text, remove_stopwords=remove_stopwords)
        elif remove_stopwords:
            words = tokenize(text)
            text = " ".join(w for w in words if w not in RUSSIAN_STOPWORDS)
        return text

    return preprocess
