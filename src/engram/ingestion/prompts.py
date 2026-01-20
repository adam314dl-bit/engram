"""LLM prompts for knowledge extraction.

Enhanced with anti-leakage instructions for Kimi K2 Thinking model.
"""

from engram.prompts.base import ANTI_LEAK_SUFFIX_RU

# System prompt for structured extraction tasks
EXTRACTION_SYSTEM_PROMPT = """Ты извлекаешь информацию в простой строчный формат.
Выводи ТОЛЬКО запрошенные строки. Без объяснений, без markdown, без лишнего текста.
НЕ используй теги <think> или подобные. Начни ответ СРАЗУ с результата."""

CONCEPT_EXTRACTION_PROMPT = """Извлеки концепты из этого текста.

Текст:
{content}

Выведи каждый концепт на новой строке в формате:
CONCEPT|название|тип|описание

Типы: tool, resource, action, state, config, error, general

Пример:
CONCEPT|docker|tool|платформа контейнеризации
CONCEPT|контейнер|resource|изолированная среда выполнения
CONCEPT|dockerfile|config|инструкции для сборки образа

Также выведи связи:
RELATION|источник|цель|тип

Типы связей: uses, needs, causes, contains, is_a, related_to

Пример:
RELATION|docker|контейнер|uses
RELATION|dockerfile|образ|causes
""" + ANTI_LEAK_SUFFIX_RU + """

Вывод:"""


MEMORY_EXTRACTION_PROMPT = """Извлеки знания из этого документа.

Документ: {title}
Содержимое:
{content}

Выведи 5-15 единиц знаний, по одной на строку:
MEMORY|содержание|тип|концепты|важность

Типы: fact, procedure, relationship
Концепты: список через запятую
Важность: 1-10

Пример:
MEMORY|Docker использует контейнеры для изоляции приложений|fact|docker,контейнер|8
MEMORY|Команда docker system prune освобождает место на диске|procedure|docker,диск|7
MEMORY|Kubernetes требует Docker или containerd для работы|relationship|kubernetes,docker|9
""" + ANTI_LEAK_SUFFIX_RU + """

Вывод:"""


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
