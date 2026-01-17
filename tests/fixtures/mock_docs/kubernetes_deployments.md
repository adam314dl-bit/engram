# Kubernetes Deployments

Deployment управляет ReplicaSet и обеспечивает декларативные обновления.

## Создание Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
```

## Стратегии обновления

### Rolling Update (по умолчанию)

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Recreate

```yaml
spec:
  strategy:
    type: Recreate
```

## Откат

```bash
# История ревизий
kubectl rollout history deployment/nginx

# Откат к предыдущей версии
kubectl rollout undo deployment/nginx

# Откат к конкретной ревизии
kubectl rollout undo deployment/nginx --to-revision=2
```

## Масштабирование

```bash
kubectl scale deployment nginx --replicas=5
```

Или автоматически с HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nginx-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nginx
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
