"""Russian/Latin transliteration for mixed-script content handling.

Provides:
- Transliteration between Cyrillic and Latin
- Dual-indexing support for mixed content
- Query expansion for transliterated variants
- Lookalike character fixing (А↔A, В↔B, etc.)
"""

from __future__ import annotations

import re
from functools import lru_cache

# Try to import cyrtranslit, fall back to manual mapping
try:
    import cyrtranslit
    HAS_CYRTRANSLIT = True
except ImportError:
    HAS_CYRTRANSLIT = False


# Manual transliteration mappings (Russian ↔ Latin)
# Based on GOST 16876-71 / ISO 9:1995 with common simplifications
CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    # Uppercase
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "Yo",
    "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Щ": "Shch",
    "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
}

LATIN_TO_CYRILLIC = {
    "a": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
    # Uppercase
    "A": "А", "B": "Б", "C": "С", "D": "Д", "E": "Е", "F": "Ф", "G": "Г",
    "H": "Х", "I": "И", "J": "Й", "K": "К", "L": "Л", "M": "М", "N": "Н",
    "O": "О", "P": "П", "Q": "К", "R": "Р", "S": "С", "T": "Т", "U": "У",
    "V": "В", "W": "В", "X": "КС", "Y": "Й", "Z": "З",
}

# Digraph mappings for Latin → Cyrillic
LATIN_DIGRAPHS_TO_CYRILLIC = {
    "sh": "ш", "ch": "ч", "zh": "ж", "ts": "ц", "ya": "я", "yu": "ю",
    "yo": "ё", "kh": "х", "shch": "щ",
    "Sh": "Ш", "Ch": "Ч", "Zh": "Ж", "Ts": "Ц", "Ya": "Я", "Yu": "Ю",
    "Yo": "Ё", "Kh": "Х", "Shch": "Щ",
}

# Cyrillic/Latin lookalike characters (visually similar)
# Map from lookalike to canonical form
LOOKALIKES: dict[str, str] = {
    # Cyrillic that looks like Latin
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "у": "y", "х": "x",
    # Latin that looks like Cyrillic
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "y": "у", "x": "х",
}


def has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r"[а-яА-ЯёЁ]", text))


def has_latin(text: str) -> bool:
    """Check if text contains Latin letters."""
    return bool(re.search(r"[a-zA-Z]", text))


def is_mixed_script(text: str) -> bool:
    """Check if text contains both Cyrillic and Latin characters."""
    return has_cyrillic(text) and has_latin(text)


@lru_cache(maxsize=10000)
def to_latin(text: str) -> str:
    """
    Transliterate Cyrillic text to Latin.

    Args:
        text: Text to transliterate

    Returns:
        Latin transliteration
    """
    if not has_cyrillic(text):
        return text

    if HAS_CYRTRANSLIT:
        return cyrtranslit.to_latin(text, "ru")

    # Manual transliteration
    result = []
    for char in text:
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        else:
            result.append(char)
    return "".join(result)


@lru_cache(maxsize=10000)
def to_cyrillic(text: str) -> str:
    """
    Transliterate Latin text to Cyrillic.

    Args:
        text: Text to transliterate

    Returns:
        Cyrillic transliteration
    """
    if not has_latin(text):
        return text

    if HAS_CYRTRANSLIT:
        return cyrtranslit.to_cyrillic(text, "ru")

    # Manual transliteration with digraph handling
    result = []
    i = 0
    text_lower = text.lower()

    while i < len(text):
        matched = False

        # Check for digraphs (longest first)
        for digraph_len in [4, 3, 2]:
            if i + digraph_len <= len(text):
                digraph = text[i:i + digraph_len]
                digraph_check = text_lower[i:i + digraph_len]

                # Check if it's a known digraph
                if digraph_check in LATIN_DIGRAPHS_TO_CYRILLIC:
                    cyrillic = LATIN_DIGRAPHS_TO_CYRILLIC[digraph_check]
                    # Preserve case
                    if digraph[0].isupper():
                        cyrillic = cyrillic.upper() if len(cyrillic) == 1 else cyrillic.capitalize()
                    result.append(cyrillic)
                    i += digraph_len
                    matched = True
                    break

        if not matched:
            char = text[i]
            if char in LATIN_TO_CYRILLIC:
                result.append(LATIN_TO_CYRILLIC[char])
            else:
                result.append(char)
            i += 1

    return "".join(result)


def normalize_for_index(text: str, target: str = "cyrillic") -> str:
    """
    Normalize text to target script for indexing.

    Args:
        text: Text to normalize
        target: Target script ("cyrillic" or "latin")

    Returns:
        Normalized text
    """
    if target == "cyrillic":
        return to_cyrillic(text)
    elif target == "latin":
        return to_latin(text)
    return text


def expand_query_transliteration(query: str) -> list[str]:
    """
    Expand query with transliteration variants.

    Returns original query plus transliterated variants for
    both Cyrillic and Latin forms.

    Args:
        query: Original query

    Returns:
        List of query variants (including original)
    """
    variants = [query]

    # Add Cyrillic variant
    cyrillic_variant = to_cyrillic(query)
    if cyrillic_variant != query:
        variants.append(cyrillic_variant)

    # Add Latin variant
    latin_variant = to_latin(query)
    if latin_variant != query and latin_variant not in variants:
        variants.append(latin_variant)

    # If query has mixed script, also try fixing lookalikes
    if is_mixed_script(query):
        fixed = fix_lookalike_typos(query)
        if fixed not in variants:
            variants.append(fixed)

    return variants


def fix_lookalike_typos(text: str) -> str:
    """
    Fix common lookalike character typos in mixed-script text.

    Detects whether the text is predominantly Cyrillic or Latin
    and normalizes lookalike characters to match.

    Args:
        text: Text potentially containing lookalike typos

    Returns:
        Text with lookalikes fixed
    """
    if not is_mixed_script(text):
        return text

    # Count Cyrillic vs Latin characters
    cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin_count = len(re.findall(r"[a-zA-Z]", text))

    # Determine dominant script
    if cyrillic_count > latin_count:
        # Text is mostly Cyrillic, convert Latin lookalikes to Cyrillic
        result = []
        for char in text:
            if char in "AaBCcEeHKkMOoPpTtXxYy" and char in LOOKALIKES:
                # Only convert if it's likely a typo (surrounded by Cyrillic)
                result.append(LOOKALIKES[char])
            else:
                result.append(char)
        return "".join(result)
    else:
        # Text is mostly Latin, convert Cyrillic lookalikes to Latin
        result = []
        for char in text:
            if char in "АаВСсЕеНКкМОоРрТтХхУу" and char in LOOKALIKES:
                result.append(LOOKALIKES[char])
            else:
                result.append(char)
        return "".join(result)


def create_dual_index_content(text: str) -> dict[str, str]:
    """
    Create dual-index content for a text.

    Returns both Cyrillic and Latin versions for indexing,
    allowing search in either script.

    Args:
        text: Original text

    Returns:
        Dict with "cyrillic" and "latin" keys
    """
    return {
        "cyrillic": to_cyrillic(text),
        "latin": to_latin(text),
        "original": text,
    }


def normalize_mixed_terms(text: str) -> str:
    """
    Normalize terms that commonly appear in mixed script.

    Handles cases like "login" / "логин", "email" / "имейл",
    technical terms that might appear in either script.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Common technical terms with their normalized forms
    term_mappings = {
        # Latin → keep as-is (technical terms)
        "login": "login",
        "email": "email",
        "api": "api",
        "url": "url",
        "http": "http",
        "https": "https",
        "json": "json",
        "xml": "xml",
        # Cyrillic transliterations → normalize to Latin
        "логин": "login",
        "имейл": "email",
        "юрл": "url",
    }

    result = text
    for term, normalized in term_mappings.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(normalized, result)

    return result


def expand_concept_for_bm25(name: str, aliases: list[str] | None = None) -> list[str]:
    """
    Expand a concept name to search variants for BM25.

    Generates variants from:
    - Original form
    - All aliases (from graph deduplication)

    Args:
        name: Concept name
        aliases: List of known aliases from graph

    Returns:
        List of unique search terms
    """
    variants: set[str] = {name.lower()}

    # Add aliases
    if aliases:
        for alias in aliases:
            variants.add(alias.lower())

    return list(variants)


def build_expanded_bm25_query(
    original_query: str,
    concept_names: list[str],
    concept_aliases: dict[str, list[str]] | None = None,
) -> str:
    """
    Build expanded BM25 query with concept variants.

    Combines original query with expanded concept terms.

    Args:
        original_query: Original user query
        concept_names: Extracted concept names
        concept_aliases: Map of concept name -> aliases from graph

    Returns:
        Expanded query string with OR clauses for concept variants
    """
    concept_aliases = concept_aliases or {}

    # Collect all expanded terms
    expanded_terms: set[str] = set()

    for name in concept_names:
        aliases = concept_aliases.get(name, [])
        variants = expand_concept_for_bm25(name, aliases)
        expanded_terms.update(variants)

    # If no expansion happened, return original
    if not expanded_terms:
        return original_query

    # Build query: original query + expanded concept terms
    # Neo4j fulltext uses Lucene syntax, OR is default between terms
    all_terms = list(expanded_terms)
    return " ".join(all_terms)
