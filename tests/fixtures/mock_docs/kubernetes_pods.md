# Kubernetes Pods

Pod — минимальная единица развертывания в Kubernetes.

## Структура Pod

Pod содержит один или несколько контейнеров, которые разделяют:
- Сетевое пространство имен (один IP)
- Volumes
- IPC namespace

## Создание Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  labels:
    app: myapp
spec:
  containers:
  - name: main
    image: nginx:1.21
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

## Init Containers

Init контейнеры выполняются до основных контейнеров:

```yaml
spec:
  initContainers:
  - name: init-db
    image: busybox
    command: ['sh', '-c', 'until nc -z db 5432; do sleep 2; done']
  containers:
  - name: app
    image: myapp
```

## Lifecycle Hooks

```yaml
lifecycle:
  postStart:
    exec:
      command: ["/bin/sh", "-c", "echo Started"]
  preStop:
    httpGet:
      path: /shutdown
      port: 8080
```

## Debugging Pods

```bash
# Логи
kubectl logs pod-name

# Выполнить команду
kubectl exec -it pod-name -- /bin/sh

# Описание pod
kubectl describe pod pod-name
```
