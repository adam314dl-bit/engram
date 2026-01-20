"""Prompts for Engram with anti-leakage support."""

from .base import (
    ANTI_LEAK_SUFFIX_RU,
    ANTI_LEAK_SUFFIX_EN,
    OUTPUT_MARKERS,
    MARKER_INSTRUCTION_RU,
    MARKER_INSTRUCTION_EN,
    EXTRACTION_SYSTEM_PROMPT_RU,
    EXTRACTION_SYSTEM_PROMPT_EN,
    get_anti_leak_suffix,
)

__all__ = [
    "ANTI_LEAK_SUFFIX_RU",
    "ANTI_LEAK_SUFFIX_EN",
    "OUTPUT_MARKERS",
    "MARKER_INSTRUCTION_RU",
    "MARKER_INSTRUCTION_EN",
    "EXTRACTION_SYSTEM_PROMPT_RU",
    "EXTRACTION_SYSTEM_PROMPT_EN",
    "get_anti_leak_suffix",
]
