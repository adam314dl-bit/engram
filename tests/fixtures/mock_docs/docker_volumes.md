# Docker Volumes

Volumes — предпочтительный способ сохранения данных в Docker.

## Создание и использование

```bash
# Создать volume
docker volume create my-data

# Использовать volume
docker run -v my-data:/app/data nginx

# Просмотреть volumes
docker volume ls
```

## Типы монтирования

### Named Volumes
Управляются Docker, хранятся в /var/lib/docker/volumes.

```bash
docker run -v my-volume:/data nginx
```

### Bind Mounts
Монтируют директорию хоста в контейнер.

```bash
docker run -v /host/path:/container/path nginx
```

### tmpfs
Хранит данные только в памяти.

```bash
docker run --tmpfs /app/cache nginx
```

## Резервное копирование

Чтобы создать резервную копию volume:
```bash
docker run --rm -v my-volume:/data -v $(pwd):/backup alpine tar cvf /backup/backup.tar /data
```

## Очистка неиспользуемых volumes

```bash
docker volume prune
```

Осторожно: это удалит все volumes, которые не используются ни одним контейнером.
