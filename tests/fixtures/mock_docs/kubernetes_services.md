# Kubernetes Services

Service обеспечивает стабильный endpoint для доступа к Pod'ам.

## Типы сервисов

### ClusterIP (по умолчанию)

Внутренний IP в кластере:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: ClusterIP
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
```

### NodePort

Открывает порт на каждой ноде:

```yaml
spec:
  type: NodePort
  ports:
  - port: 80
    nodePort: 30080
```

### LoadBalancer

Создает внешний балансировщик (в облаке):

```yaml
spec:
  type: LoadBalancer
  ports:
  - port: 80
```

### ExternalName

DNS CNAME для внешнего сервиса:

```yaml
spec:
  type: ExternalName
  externalName: api.external.com
```

## Headless Service

Для StatefulSet и прямого доступа к Pod'ам:

```yaml
spec:
  clusterIP: None
  selector:
    app: myapp
```

## Service Discovery

Pods могут обращаться к сервису по DNS:
- `my-service` (в том же namespace)
- `my-service.namespace.svc.cluster.local` (полное имя)
