# Grafana

Визуализация метрик с Grafana.

## Установка

```yaml
# docker-compose.yml
services:
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
```

## Data Sources

Grafana поддерживает множество источников данных:
- Prometheus
- InfluxDB
- Elasticsearch
- PostgreSQL
- MySQL
- Loki

Добавление через UI: Configuration > Data Sources > Add data source

## Dashboards

### Переменные

```
# Query variable
label_values(up, instance)

# Interval variable
$__interval
```

### Панели

Типы панелей:
- Time series — графики
- Stat — одно значение
- Gauge — шкала
- Table — таблица
- Logs — логи
- Heatmap — тепловая карта

### Queries

```promql
# Prometheus
rate(http_requests_total{instance="$instance"}[$__rate_interval])

# С переменными
sum(rate(http_requests_total{job="$job"}[5m])) by (method)
```

## Alerting

1. Create Alert Rule в панели
2. Настроить Contact Point (Email, Slack, PagerDuty)
3. Настроить Notification Policy

```yaml
# Provisioning alert
apiVersion: 1
groups:
- name: my-alerts
  rules:
  - alert: HighCPU
    condition: A > 80
```

## Provisioning

```yaml
# /etc/grafana/provisioning/dashboards/default.yaml
apiVersion: 1
providers:
- name: 'default'
  folder: ''
  type: file
  options:
    path: /var/lib/grafana/dashboards
```
