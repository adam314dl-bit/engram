"""LLM prompts for knowledge extraction."""

# System prompt for structured extraction tasks
EXTRACTION_SYSTEM_PROMPT = """You extract information into a simple line format.
Output ONLY the requested lines. No explanations, no markdown, no extra text."""

CONCEPT_EXTRACTION_PROMPT = """Extract concepts from this text.

Text:
{content}

Output each concept on a new line in format:
CONCEPT|name|type|description

Types: tool, resource, action, state, config, error, general

Example:
CONCEPT|docker|tool|containerization platform
CONCEPT|container|resource|isolated environment
CONCEPT|dockerfile|config|build instructions

Also output relations:
RELATION|source|target|type

Relation types: uses, needs, causes, contains, is_a, related_to

Example:
RELATION|docker|container|uses
RELATION|dockerfile|image|creates

Output:"""


MEMORY_EXTRACTION_PROMPT = """Extract knowledge from this document.

Document: {title}
Content:
{content}

Output 5-15 knowledge items, one per line:
MEMORY|content|type|concepts|importance

Types: fact, procedure, relationship
Concepts: comma-separated list
Importance: 1-10

Example:
MEMORY|Docker uses containers for app isolation|fact|docker,container|8
MEMORY|Run docker system prune to free space|procedure|docker,disk|7
MEMORY|Kubernetes requires Docker or containerd|relationship|kubernetes,docker|9

Output:"""


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
