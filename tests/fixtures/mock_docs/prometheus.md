# Prometheus

Мониторинг с Prometheus.

## Установка

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Конфигурация

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'app'
    static_configs:
      - targets: ['app:8080']
```

## PromQL

```promql
# Мгновенное значение
http_requests_total

# За последние 5 минут
http_requests_total[5m]

# Rate (запросов в секунду)
rate(http_requests_total[5m])

# Фильтрация
http_requests_total{method="GET", status="200"}

# Агрегация
sum(rate(http_requests_total[5m])) by (method)

# Среднее
avg(node_cpu_seconds_total)
```

## Alerting

```yaml
# alert.rules.yml
groups:
- name: example
  rules:
  - alert: HighRequestLatency
    expr: http_request_duration_seconds > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency on {{ $labels.instance }}"
```

## Service Discovery

```yaml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```
