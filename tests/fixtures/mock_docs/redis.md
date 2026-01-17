# Redis

Основы работы с Redis.

## Подключение

```bash
redis-cli
redis-cli -h localhost -p 6379
```

## Типы данных

### Strings

```redis
SET key "value"
GET key
INCR counter
EXPIRE key 3600
TTL key
```

### Lists

```redis
LPUSH mylist "a"
RPUSH mylist "b"
LRANGE mylist 0 -1
LPOP mylist
```

### Sets

```redis
SADD myset "a" "b" "c"
SMEMBERS myset
SISMEMBER myset "a"
SUNION set1 set2
```

### Hashes

```redis
HSET user:1 name "John" age 30
HGET user:1 name
HGETALL user:1
HINCRBY user:1 age 1
```

### Sorted Sets

```redis
ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2"
ZRANGE leaderboard 0 -1 WITHSCORES
ZRANK leaderboard "player1"
```

## Pub/Sub

```redis
# Подписка
SUBSCRIBE channel

# Публикация
PUBLISH channel "message"
```

## Transactions

```redis
MULTI
SET key1 "value1"
SET key2 "value2"
EXEC
```

## Persistence

```redis
# RDB (snapshot)
SAVE
BGSAVE

# AOF (append-only file)
# В redis.conf: appendonly yes
```

## Кластер и Sentinel

```bash
# Sentinel для HA
redis-sentinel /path/to/sentinel.conf

# Кластер
redis-cli --cluster create node1:6379 node2:6379 node3:6379
```
