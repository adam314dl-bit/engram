"""LLM prompts for knowledge extraction."""

# System prompt for JSON extraction tasks
JSON_SYSTEM_PROMPT = """You are a JSON extraction assistant.
IMPORTANT: Output ONLY valid JSON. No explanations, no markdown, no thinking out loud.
Start your response with {{ or [ and end with }} or ]."""

CONCEPT_EXTRACTION_PROMPT = """Extract concepts from this text and output as JSON.

Text:
{content}

Output format (JSON only, no explanations):
{{
  "concepts": [
    {{"name": "concept name lowercase", "type": "tool|resource|action|state|config|error|general", "description": "brief description or null"}}
  ],
  "relations": [
    {{"source": "concept1", "target": "concept2", "type": "uses|needs|causes|contains|is_a|related_to"}}
  ]
}}

JSON:"""


MEMORY_EXTRACTION_PROMPT = """Extract knowledge units from this document and output as JSON.

Document: {title}
Content:
{content}

Extract 5-15 knowledge units: facts, procedures, relationships.

Output format (JSON only, no explanations):
{{
  "memories": [
    {{"content": "Docker uses containers for isolation", "type": "fact", "concepts": ["docker", "container"], "importance": 8}},
    {{"content": "Run docker system prune to free disk space", "type": "procedure", "concepts": ["docker", "disk"], "importance": 7}}
  ]
}}

JSON:"""


BEHAVIOR_EXTRACTION_PROMPT = """Проанализируй ответ и извлеки стратегию рассуждения.

Вопрос пользователя: {query}
Ответ системы: {answer}

Опиши использованную стратегию в формате:
- name: краткое имя на английском в snake_case (например: check_disk_usage, explain_concept)
- instruction: одно предложение описывающее паттерн рассуждения
- domain: область применения (docker, kubernetes, general, и т.д.)

Выведи ТОЛЬКО валидный JSON:
{{
  "name": "check_disk_usage",
  "instruction": "Проверить использование диска командой df -h, найти крупнейших потребителей",
  "domain": "docker"
}}"""


CONSOLIDATION_PROMPT = """Эти похожие вопросы успешно решались одинаковым подходом:

{episodes}

Извлеки общий паттерн знания в формате:
"Когда [ситуация], [действие] потому что [причина]"

Будь кратким и общим, не привязывайся к конкретным деталям.
Выведи только текст паттерна, без JSON."""


REFLECTION_PROMPT = """Проанализируй недавние взаимодействия и извлеки высокоуровневые инсайты:

{episodes}

Сгенерируй 2-3 обобщения:
- Какие темы/проблемы повторяются?
- Какие подходы работают лучше всего?
- Какие пробелы в знаниях выявились?

Каждый инсайт должен быть связан с конкретными концептами.

Выведи JSON:
{{
  "reflections": [
    {{
      "content": "Пользователи часто сталкиваются с проблемами дискового пространства Docker",
      "concepts": ["docker", "disk space"],
      "type": "pattern"
    }}
  ]
}}"""


RESPONSE_SYNTHESIS_PROMPT = """Вопрос пользователя: {query}

Релевантные знания:
{context}

Прошлые похожие рассуждения (используй как шаблон если применимо):
{episodes}

Дай полезный ответ. Если это вопрос "что такое X" — объясни концепцию.
Если это проблема — предложи решение пошагово.
Упомяни если уверенность в информации низкая.

Также кратко опиши свою стратегию рассуждения одним предложением
в формате: СТРАТЕГИЯ: [название_поведения] — [краткое описание]"""


RE_REASONING_PROMPT = """Предыдущий подход не помог: {failed_behavior}

Вопрос пользователя: {query}

Попробуй альтернативный подход используя эти знания:
{alternative_memories}

Успешные альтернативные подходы в похожих ситуациях:
{alternative_episodes}

Предложи другое решение или задай уточняющий вопрос если нужна дополнительная информация.

СТРАТЕГИЯ: [название_поведения] — [краткое описание]"""
