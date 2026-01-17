# Docker Compose Advanced

Продвинутые техники работы с Docker Compose.

## Переменные окружения

```yaml
services:
  app:
    image: myapp
    environment:
      - DB_HOST=${DB_HOST:-localhost}
      - DB_PORT=${DB_PORT:-5432}
    env_file:
      - .env
```

## Зависимости сервисов

```yaml
services:
  web:
    depends_on:
      db:
        condition: service_healthy
  db:
    healthcheck:
      test: ["CMD", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5
```

## Profiles

```yaml
services:
  web:
    profiles: ["frontend"]
  worker:
    profiles: ["backend"]
  debug:
    profiles: ["debug"]
```

Запуск с профилем:
```bash
docker compose --profile frontend up
```

## Расширение сервисов

```yaml
x-common: &common
  restart: always
  logging:
    driver: json-file

services:
  app:
    <<: *common
    image: myapp
```

## Секреты

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  db:
    secrets:
      - db_password
```
