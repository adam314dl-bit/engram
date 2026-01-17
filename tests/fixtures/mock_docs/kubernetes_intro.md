# Введение в Kubernetes

Kubernetes (K8s) — это платформа оркестрации контейнеров. Она автоматизирует развертывание, масштабирование и управление контейнеризованными приложениями.

## Основные компоненты

### Pod

Pod — минимальная единица развертывания в Kubernetes. Pod может содержать один или несколько контейнеров.

### Deployment

Deployment управляет репликами Pod'ов и обеспечивает:
- Декларативное обновление
- Откат изменений
- Масштабирование

### Service

Service обеспечивает сетевой доступ к Pod'ам. Типы Service:
- ClusterIP — внутренний доступ
- NodePort — доступ через порт ноды
- LoadBalancer — внешний балансировщик

## Основные команды kubectl

```bash
kubectl get pods          # список pod'ов
kubectl apply -f file.yaml  # применить конфигурацию
kubectl logs pod-name     # логи pod'а
kubectl exec -it pod-name -- bash  # shell в pod
```

## Связь с Docker

Kubernetes использует Docker (или другой container runtime) для запуска контейнеров. Образы Docker разворачиваются в Pod'ах.
