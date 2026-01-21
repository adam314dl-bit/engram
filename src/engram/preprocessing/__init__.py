"""Preprocessing modules for Engram."""

from .normalizer import (
    BULLET_SYMBOLS,
    CHECKMARK_SYMBOLS,
    CROSS_SYMBOLS,
    normalize_bullet,
    normalize_bullets_in_text,
    normalize_full,
    normalize_table_cell,
    normalize_table_cells_in_text,
    normalize_unicode,
    normalize_whitespace,
)
from .russian import (
    RUSSIAN_STOPWORDS,
    clean_text,
    create_preprocessor,
    get_morph_analyzer,
    get_word_forms,
    is_russian_text,
    lemmatize_text,
    lemmatize_word,
    prepare_bm25_document,
    prepare_bm25_query,
    tokenize,
)
from .table_parser import (
    ParsedTable,
    TableBoundary,
    detect_tables,
    extract_tables,
    parse_table,
    remove_tables_from_text,
)
from .thinking_stripper import OutputParser, ThinkingStripper

# Note: table_enricher is imported lazily to avoid circular imports
# Use: from engram.preprocessing.table_enricher import TableEnricher

__all__ = [
    # Thinking stripper
    "ThinkingStripper",
    "OutputParser",
    # Russian NLP
    "RUSSIAN_STOPWORDS",
    "clean_text",
    "create_preprocessor",
    "get_morph_analyzer",
    "get_word_forms",
    "is_russian_text",
    "lemmatize_text",
    "lemmatize_word",
    "prepare_bm25_document",
    "prepare_bm25_query",
    "tokenize",
    # Normalizer
    "BULLET_SYMBOLS",
    "CHECKMARK_SYMBOLS",
    "CROSS_SYMBOLS",
    "normalize_bullet",
    "normalize_bullets_in_text",
    "normalize_full",
    "normalize_table_cell",
    "normalize_table_cells_in_text",
    "normalize_unicode",
    "normalize_whitespace",
    # Table parser
    "ParsedTable",
    "TableBoundary",
    "detect_tables",
    "extract_tables",
    "parse_table",
    "remove_tables_from_text",
]
