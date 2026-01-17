# Kubernetes ConfigMaps и Secrets

ConfigMaps хранят конфигурацию, Secrets — чувствительные данные.

## ConfigMap

### Создание

```bash
# Из литерала
kubectl create configmap my-config --from-literal=key=value

# Из файла
kubectl create configmap my-config --from-file=config.properties
```

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_URL: "postgres://db:5432"
  LOG_LEVEL: "info"
```

### Использование

```yaml
# Как переменные окружения
env:
- name: DATABASE_URL
  valueFrom:
    configMapKeyRef:
      name: app-config
      key: DATABASE_URL

# Как volume
volumes:
- name: config
  configMap:
    name: app-config
```

## Secrets

### Создание

```bash
kubectl create secret generic db-secret   --from-literal=password=mysecret
```

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  password: bXlzZWNyZXQ=  # base64 encoded
```

### Использование

```yaml
env:
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: db-secret
      key: password
```

## Важно

- ConfigMaps и Secrets имеют лимит 1MB
- Secrets кодируются в base64, но не шифруются
- Используйте external-secrets или sealed-secrets для продакшена
