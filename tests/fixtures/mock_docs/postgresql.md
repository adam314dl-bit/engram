# PostgreSQL

Основы работы с PostgreSQL.

## Подключение

```bash
psql -h localhost -U postgres -d mydb

# Или через URL
psql postgresql://user:password@localhost:5432/mydb
```

## Основные команды psql

```sql
\l          -- список баз
\c mydb     -- подключиться к базе
\dt         -- список таблиц
\d table    -- структура таблицы
\du         -- список пользователей
\q          -- выход
```

## DDL

```sql
-- Создание базы
CREATE DATABASE mydb;

-- Создание таблицы
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_name_gin ON users USING gin(name gin_trgm_ops);
```

## DML

```sql
-- Insert
INSERT INTO users (email, name) VALUES ('test@example.com', 'Test User');

-- Update
UPDATE users SET name = 'New Name' WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;

-- Select
SELECT * FROM users WHERE email LIKE '%@example.com' ORDER BY created_at DESC LIMIT 10;
```

## Транзакции

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- или ROLLBACK;
```

## Backup/Restore

```bash
# Backup
pg_dump mydb > backup.sql
pg_dump -Fc mydb > backup.dump  # сжатый формат

# Restore
psql mydb < backup.sql
pg_restore -d mydb backup.dump
```

## Производительность

```sql
-- План запроса
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- Статистика таблицы
ANALYZE users;

-- Очистка
VACUUM ANALYZE users;
```
