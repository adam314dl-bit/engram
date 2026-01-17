# Docker Networking

Docker предоставляет несколько драйверов сети для разных сценариев использования.

## Типы сетей

### Bridge Network
Сеть по умолчанию для контейнеров на одном хосте.

```bash
docker network create my-bridge
docker run --network my-bridge nginx
```

Контейнеры в bridge сети могут общаться друг с другом по имени контейнера.

### Host Network
Контейнер использует сетевой стек хоста напрямую.

```bash
docker run --network host nginx
```

Используйте host network когда нужна максимальная производительность сети.

### Overlay Network
Для связи контейнеров между несколькими Docker хостами в Swarm.

```bash
docker network create --driver overlay my-overlay
```

### Macvlan
Присваивает контейнеру MAC-адрес, делая его видимым как физическое устройство.

## Диагностика сети

Чтобы проверить сетевые настройки контейнера:
```bash
docker inspect --format='{{json .NetworkSettings}}' container_name
```

Для тестирования связности между контейнерами:
```bash
docker exec container1 ping container2
```
