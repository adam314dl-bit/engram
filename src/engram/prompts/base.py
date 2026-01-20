"""
Base prompts with anti-leakage instructions.
Designed for Kimi K2 Thinking with Russian content.
"""

# Anti-leakage suffix (Russian)
ANTI_LEAK_SUFFIX_RU = """

КРИТИЧЕСКИ ВАЖНО:
- Выдай ТОЛЬКО запрошенный результат
- НЕ показывай процесс рассуждения
- НЕ используй теги <think> или подобные
- Начни ответ СРАЗУ с результата"""

# Anti-leakage suffix (English)
ANTI_LEAK_SUFFIX_EN = """

CRITICAL:
- Output ONLY the requested result
- Do NOT show reasoning process
- Do NOT use <think> or similar tags
- Start response DIRECTLY with the result"""


def get_anti_leak_suffix(language: str = "ru") -> str:
    """Get anti-leakage suffix for prompt."""
    return ANTI_LEAK_SUFFIX_RU if language == "ru" else ANTI_LEAK_SUFFIX_EN


# System prompt for extraction tasks
EXTRACTION_SYSTEM_PROMPT_RU = """Ты — система извлечения информации.
Твоя задача — извлекать данные из документов точно и структурированно.

Правила:
1. Выдавай ТОЛЬКО извлечённые данные
2. Не объясняй свои рассуждения
3. Не используй теги размышлений
4. Отвечай в запрошенном формате"""

EXTRACTION_SYSTEM_PROMPT_EN = """You are an information extraction system.
Your task is to extract data from documents accurately and in a structured way.

Rules:
1. Output ONLY the extracted data
2. Do not explain your reasoning
3. Do not use thinking tags
4. Respond in the requested format"""


# Marker-based output format (for reliable extraction)
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
