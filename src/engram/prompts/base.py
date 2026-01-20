"""
Base prompts - FIXED version without aggressive anti-leakage.

The --reasoning-parser kimi_k2 flag handles separation automatically.
No need for prompt-level instructions that confuse the model.
"""

# Simple prompt ending (Russian)
PROMPT_ENDING_RU = """

Ответ:"""

# Simple prompt ending (English)
PROMPT_ENDING_EN = """

Answer:"""


def get_prompt_ending(language: str = "ru") -> str:
    """Get simple prompt ending."""
    return PROMPT_ENDING_RU if language == "ru" else PROMPT_ENDING_EN


# DEPRECATED: Keep for backwards compatibility but don't use
# These caused the model to put everything in <think> tags
ANTI_LEAK_SUFFIX_RU = PROMPT_ENDING_RU
ANTI_LEAK_SUFFIX_EN = PROMPT_ENDING_EN


def get_anti_leak_suffix(language: str = "ru") -> str:
    """DEPRECATED: Use get_prompt_ending instead."""
    return get_prompt_ending(language)


# System prompt - keep simple, no anti-reasoning instructions
EXTRACTION_SYSTEM_PROMPT_RU = """Ты — система извлечения информации.
Извлекай данные из документов точно и структурированно."""

EXTRACTION_SYSTEM_PROMPT_EN = """You are an information extraction system.
Extract data from documents accurately and in a structured way."""


# Marker-based output (optional, for critical extractions)
OUTPUT_MARKERS = {
    "start": "===РЕЗУЛЬТАТ===",
    "end": "===КОНЕЦ===",
}

MARKER_INSTRUCTION_RU = f"""
Формат ответа:
{OUTPUT_MARKERS['start']}
[твой ответ здесь]
{OUTPUT_MARKERS['end']}

Помести ТОЛЬКО финальный результат между маркерами."""

MARKER_INSTRUCTION_EN = f"""
Response format:
{OUTPUT_MARKERS['start']}
[your answer here]
{OUTPUT_MARKERS['end']}

Place ONLY the final result between markers."""
