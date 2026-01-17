# ELK Stack

Централизованное логирование с Elasticsearch, Logstash, Kibana.

## Архитектура

```
App -> Filebeat -> Logstash -> Elasticsearch -> Kibana
```

## Docker Compose

```yaml
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Logstash конфигурация

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  grok {
    match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
  }
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{+YYYY.MM.dd}"
  }
}
```

## Filebeat

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /var/log/app/*.log

output.logstash:
  hosts: ["logstash:5044"]
```

## Kibana

1. Создать Index Pattern: `logs-*`
2. Discover — просмотр логов
3. Visualize — графики
4. Dashboard — дашборды

## Полезные запросы

```
# Поиск по тексту
message: "error"

# Фильтр по полю
level: "ERROR" AND service: "api"

# Диапазон времени
@timestamp:[now-1h TO now]
```
