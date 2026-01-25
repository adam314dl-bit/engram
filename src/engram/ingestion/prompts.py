"""LLM prompts for knowledge extraction.

FIXED version - removed aggressive anti-leakage that caused empty content.
The --reasoning-parser kimi_k2 flag handles thinking separation automatically.
"""

from engram.prompts.base import PROMPT_ENDING_RU

# System prompt for structured extraction tasks - SIMPLIFIED
EXTRACTION_SYSTEM_PROMPT = """Ты извлекаешь информацию в простой строчный формат.
Выводи ТОЛЬКО запрошенные строки. Без объяснений, без markdown, без лишнего текста."""

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
""" + PROMPT_ENDING_RU


MEMORY_EXTRACTION_PROMPT = """Извлеки знания из этого документа.

Документ: {title}
Содержимое:
{content}

Выведи 5-15 единиц знаний, по одной на строку:
MEMORY|search_summary|keywords|actual_content|type|concepts|importance

Формат (7 полей через |):
- search_summary: 1-2 предложения ОПИСЫВАЮЩИЕ что за информация содержится (для поиска)
- keywords: термины, сущности, названия для BM25 поиска (через запятую)
- actual_content: ТОЧНАЯ информация/факт из документа (цитата или близко к тексту, максимум 200 слов)
- type: fact, procedure, relationship
- concepts: связанные концепции через запятую
- importance: 1-10

ВАЖНО: actual_content должен содержать РЕАЛЬНЫЕ данные из документа:
- Для процедур: конкретные команды, шаги
- Для фактов: точные имена, числа, даты, параметры
- Для таблиц: ключевые строки/значения
- НЕ пересказывай — извлекай дословно или близко к оригиналу

Если документ содержит таблицы:
- search_summary: описание что таблица содержит (сравнение, список, матрица)
- keywords: названия колонок и ключевые значения
- actual_content: ключевые строки таблицы в читаемом формате

Если документ содержит списки:
- search_summary: описание что перечислено
- keywords: важные элементы списка
- actual_content: сам список или его ключевые пункты

Также выведи упомянутых людей (только настоящие имена людей, НЕ должности и НЕ общие слова):
PERSON|имя|роль|команда

Пример:
MEMORY|Описание работы Docker с контейнерами и изоляцией|docker, контейнер, изоляция, виртуализация|Docker контейнеры изолируют приложения друг от друга используя namespaces и cgroups ядра Linux. Каждый контейнер имеет свой изолированный процесс, сеть и файловую систему.|fact|docker,контейнер|8
MEMORY|Инструкция по очистке дискового пространства в Docker|docker system prune, очистка диска, docker cleanup|Для очистки неиспользуемых данных Docker используйте: docker system prune -a --volumes. Эта команда удаляет остановленные контейнеры, неиспользуемые сети, образы и тома.|procedure|docker,диск|7
MEMORY|Зависимость Kubernetes от container runtime|kubernetes, docker, containerd, runtime, cri|Kubernetes требует container runtime для запуска контейнеров. Поддерживаются: containerd (рекомендуется), CRI-O, Docker через dockershim (deprecated в 1.24).|relationship|kubernetes,docker|9
PERSON|Иван Петров|тимлид|backend
PERSON|Мария Сидорова|разработчик|frontend
""" + PROMPT_ENDING_RU


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


# v3.6: Unified extraction prompt - 1 LLM call per document
# v4.5: Updated to use dual-content format (search_summary|keywords|actual_content)
UNIFIED_EXTRACTION_PROMPT = """Извлеки структурированные знания из документа.

Документ: {title}
Содержимое:
{content}

Выведи в следующем порядке:

=== КОНЦЕПТЫ ===
CONCEPT|название|тип|описание

Типы: tool, resource, action, state, config, error, general
Извлеки 5-15 ключевых концептов.

=== СВЯЗИ ===
RELATION|источник|цель|тип|сила

Типы: is_a, contains, uses, needs, causes, related_to
Сила: 0.1-1.0 (сила связи)
Используй названия концептов из секции выше.

=== ЗНАНИЯ ===
MEMORY|search_summary|keywords|actual_content|type|concepts|importance

Формат (7 полей через |):
- search_summary: 1-2 предложения ОПИСЫВАЮЩИЕ что за информация (для поиска)
- keywords: термины для BM25 поиска (через запятую)
- actual_content: ТОЧНАЯ информация из документа (цитата, максимум 200 слов)
- type: fact, procedure, relationship
- concepts: связанные концепции через запятую
- importance: 1-10

ВАЖНО: actual_content должен содержать РЕАЛЬНЫЕ данные:
- Для процедур: конкретные команды, шаги
- Для фактов: точные имена, числа, даты
- НЕ пересказывай — извлекай близко к оригиналу

Если есть таблицы - извлеки ключевые строки в actual_content.
Если есть списки - извлеки важные пункты в actual_content.
Извлеки 5-15 единиц знаний.

=== ЛЮДИ ===
PERSON|имя|роль|команда

Только настоящие имена (не должности, не общие слова).

Пример вывода:
CONCEPT|docker|tool|платформа контейнеризации
CONCEPT|контейнер|resource|изолированная среда выполнения
RELATION|docker|контейнер|uses|0.9
RELATION|dockerfile|образ|causes|0.8
MEMORY|Описание работы Docker с контейнерами|docker, контейнер, изоляция|Docker контейнеры изолируют приложения используя namespaces и cgroups. Каждый контейнер имеет изолированный процесс и файловую систему.|fact|docker,контейнер|8
PERSON|Иван Петров|тимлид|backend
""" + PROMPT_ENDING_RU
