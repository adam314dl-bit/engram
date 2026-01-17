#!/usr/bin/env python3
"""Generate mock documentation for testing Engram."""

from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_docs"

MOCK_DOCS = {
    # Docker Advanced
    "docker_networking.md": """# Docker Networking

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
""",

    "docker_volumes.md": """# Docker Volumes

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
""",

    "docker_compose_advanced.md": """# Docker Compose Advanced

Продвинутые техники работы с Docker Compose.

## Переменные окружения

```yaml
services:
  app:
    image: myapp
    environment:
      - DB_HOST=${DB_HOST:-localhost}
      - DB_PORT=${DB_PORT:-5432}
    env_file:
      - .env
```

## Зависимости сервисов

```yaml
services:
  web:
    depends_on:
      db:
        condition: service_healthy
  db:
    healthcheck:
      test: ["CMD", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5
```

## Profiles

```yaml
services:
  web:
    profiles: ["frontend"]
  worker:
    profiles: ["backend"]
  debug:
    profiles: ["debug"]
```

Запуск с профилем:
```bash
docker compose --profile frontend up
```

## Расширение сервисов

```yaml
x-common: &common
  restart: always
  logging:
    driver: json-file

services:
  app:
    <<: *common
    image: myapp
```

## Секреты

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  db:
    secrets:
      - db_password
```
""",

    # Kubernetes
    "kubernetes_pods.md": """# Kubernetes Pods

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
""",

    "kubernetes_deployments.md": """# Kubernetes Deployments

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
""",

    "kubernetes_services.md": """# Kubernetes Services

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
""",

    "kubernetes_configmaps.md": """# Kubernetes ConfigMaps и Secrets

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
kubectl create secret generic db-secret \
  --from-literal=password=mysecret
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
""",

    "kubernetes_ingress.md": """# Kubernetes Ingress

Ingress управляет внешним доступом к сервисам в кластере.

## Ingress Controller

Перед использованием Ingress нужен контроллер:

```bash
# Nginx Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.0/deploy/static/provider/cloud/deploy.yaml
```

## Простой Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: 80
```

## TLS

```yaml
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: tls-secret
  rules:
  - host: myapp.example.com
```

## Аннотации

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
```

## Несколько путей

```yaml
spec:
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```
""",

    "kubernetes_rbac.md": """# Kubernetes RBAC

Role-Based Access Control управляет доступом к ресурсам кластера.

## Компоненты

- **Role/ClusterRole** — набор разрешений
- **RoleBinding/ClusterRoleBinding** — связывает Role с пользователем

## Role (namespace-scoped)

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
```

## ClusterRole (cluster-scoped)

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "watch", "list"]
```

## RoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: default
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

## ServiceAccount

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-service-account
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: my-sa-binding
subjects:
- kind: ServiceAccount
  name: my-service-account
  namespace: default
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

## Проверка доступа

```bash
kubectl auth can-i create pods --as=jane
kubectl auth can-i --list --as=system:serviceaccount:default:my-sa
```
""",

    # Linux Administration
    "linux_systemd.md": """# Systemd

Systemd — система инициализации и управления сервисами в Linux.

## Управление сервисами

```bash
# Статус
systemctl status nginx

# Запуск/остановка
systemctl start nginx
systemctl stop nginx
systemctl restart nginx
systemctl reload nginx

# Автозапуск
systemctl enable nginx
systemctl disable nginx
```

## Создание unit-файла

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/start.sh
ExecStop=/opt/myapp/bin/stop.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

После создания:
```bash
systemctl daemon-reload
systemctl enable myapp
systemctl start myapp
```

## Журналы (journald)

```bash
# Логи сервиса
journalctl -u nginx

# Последние 100 строк
journalctl -u nginx -n 100

# В реальном времени
journalctl -u nginx -f

# За последний час
journalctl -u nginx --since "1 hour ago"
```

## Типы сервисов

- **simple** — основной процесс указан в ExecStart
- **forking** — процесс создает дочерний и завершается
- **oneshot** — одноразовое выполнение
- **notify** — отправляет уведомление о готовности
""",

    "linux_networking.md": """# Linux Networking

Основы сетевой настройки в Linux.

## Просмотр сетевых интерфейсов

```bash
# Современный способ
ip addr show
ip link show

# Маршруты
ip route show

# Устаревший (но ещё используется)
ifconfig
route -n
```

## Настройка интерфейса

```bash
# Добавить IP
ip addr add 192.168.1.10/24 dev eth0

# Включить интерфейс
ip link set eth0 up

# Добавить маршрут
ip route add 10.0.0.0/8 via 192.168.1.1
```

## Netplan (Ubuntu)

```yaml
# /etc/netplan/01-netcfg.yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - 192.168.1.10/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

```bash
netplan apply
```

## Диагностика

```bash
# Проверка связи
ping -c 4 google.com

# Трассировка
traceroute google.com

# DNS
dig google.com
nslookup google.com

# Порты и соединения
ss -tulpn
netstat -tulpn

# Проверка порта
nc -zv localhost 80
```

## Firewall (iptables/nftables)

```bash
# Просмотр правил
iptables -L -n -v

# Открыть порт
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# UFW (упрощенный firewall)
ufw allow 80/tcp
ufw enable
```
""",

    "linux_disk.md": """# Linux Disk Management

Управление дисками и файловыми системами.

## Просмотр дисков

```bash
# Список блочных устройств
lsblk

# Информация о разделах
fdisk -l

# Использование дисков
df -h

# Использование директорий
du -sh /var/*
```

## Разметка диска

```bash
# Интерактивный режим
fdisk /dev/sdb

# Или gdisk для GPT
gdisk /dev/sdb
```

## Создание файловой системы

```bash
# ext4
mkfs.ext4 /dev/sdb1

# XFS
mkfs.xfs /dev/sdb1

# Swap
mkswap /dev/sdb2
swapon /dev/sdb2
```

## Монтирование

```bash
# Временное монтирование
mount /dev/sdb1 /mnt/data

# Постоянное монтирование в /etc/fstab
echo "/dev/sdb1 /mnt/data ext4 defaults 0 2" >> /etc/fstab
mount -a
```

## LVM

```bash
# Создание Physical Volume
pvcreate /dev/sdb

# Создание Volume Group
vgcreate myvg /dev/sdb

# Создание Logical Volume
lvcreate -L 10G -n mylv myvg

# Расширение
lvextend -L +5G /dev/myvg/mylv
resize2fs /dev/myvg/mylv
```

## RAID (mdadm)

```bash
# Создание RAID1
mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc

# Статус
cat /proc/mdstat
mdadm --detail /dev/md0
```
""",

    "linux_processes.md": """# Linux Process Management

Управление процессами в Linux.

## Просмотр процессов

```bash
# Все процессы
ps aux

# Дерево процессов
pstree

# Интерактивный мониторинг
top
htop  # более удобный

# По имени
pgrep nginx
pidof nginx
```

## Управление процессами

```bash
# Завершить процесс
kill PID
kill -9 PID  # принудительно (SIGKILL)
kill -15 PID  # корректно (SIGTERM)

# По имени
pkill nginx
killall nginx

# Приоритет (nice)
nice -n 10 command  # запуск с приоритетом
renice -n 5 -p PID  # изменить приоритет
```

## Фоновые процессы

```bash
# Запуск в фоне
command &

# Перевести в фон
Ctrl+Z
bg

# Вернуть на передний план
fg

# Список фоновых задач
jobs

# Отвязать от терминала
nohup command &
disown
```

## Мониторинг ресурсов

```bash
# Память
free -h

# CPU
mpstat 1

# I/O
iostat 1
iotop

# Сетевой трафик
iftop
nethogs
```

## cgroups

```bash
# Ограничение памяти для процесса
cgcreate -g memory:mygroup
echo 100M > /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
cgexec -g memory:mygroup command
```
""",

    # Git
    "git_basics.md": """# Git Basics

Основы работы с Git.

## Инициализация

```bash
# Новый репозиторий
git init

# Клонирование
git clone https://github.com/user/repo.git
```

## Базовый workflow

```bash
# Статус
git status

# Добавить файлы
git add file.txt
git add .  # все файлы

# Коммит
git commit -m "Описание изменений"

# Push
git push origin main
```

## Ветки

```bash
# Создать ветку
git branch feature

# Переключиться
git checkout feature
git switch feature  # новый синтаксис

# Создать и переключиться
git checkout -b feature
git switch -c feature

# Список веток
git branch -a

# Удалить ветку
git branch -d feature
```

## Слияние

```bash
# Merge
git checkout main
git merge feature

# Rebase
git checkout feature
git rebase main
```

## История

```bash
# Лог
git log
git log --oneline --graph

# Изменения в коммите
git show commit_hash

# Разница
git diff
git diff --staged
```

## Отмена изменений

```bash
# Отменить изменения в файле
git checkout -- file.txt
git restore file.txt

# Убрать из staging
git reset HEAD file.txt
git restore --staged file.txt

# Отменить коммит (сохранить изменения)
git reset --soft HEAD~1

# Отменить коммит (удалить изменения)
git reset --hard HEAD~1
```
""",

    "git_advanced.md": """# Git Advanced

Продвинутые техники Git.

## Interactive Rebase

```bash
# Изменить последние N коммитов
git rebase -i HEAD~3
```

Команды:
- `pick` — оставить коммит
- `reword` — изменить сообщение
- `squash` — объединить с предыдущим
- `fixup` — объединить без сообщения
- `drop` — удалить коммит

## Cherry-pick

```bash
# Применить конкретный коммит
git cherry-pick commit_hash
```

## Stash

```bash
# Сохранить изменения
git stash
git stash push -m "описание"

# Список
git stash list

# Применить
git stash pop
git stash apply stash@{0}

# Удалить
git stash drop stash@{0}
```

## Bisect

```bash
# Найти коммит с багом
git bisect start
git bisect bad  # текущий коммит плохой
git bisect good commit_hash  # этот был хороший

# После каждой проверки
git bisect good  # или
git bisect bad

# Завершить
git bisect reset
```

## Worktree

```bash
# Создать дополнительный worktree
git worktree add ../feature-branch feature

# Список
git worktree list

# Удалить
git worktree remove ../feature-branch
```

## Submodules

```bash
# Добавить
git submodule add https://github.com/user/lib libs/lib

# Клонировать с submodules
git clone --recursive repo

# Обновить
git submodule update --init --recursive
```

## Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
npm run lint
```
""",

    # CI/CD
    "github_actions.md": """# GitHub Actions

CI/CD с GitHub Actions.

## Базовый workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run tests
      run: npm test

    - name: Build
      run: npm run build
```

## Матрица стратегий

```yaml
jobs:
  test:
    strategy:
      matrix:
        node: [18, 20]
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
    - uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node }}
```

## Секреты

```yaml
steps:
- name: Deploy
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: ./deploy.sh
```

## Артефакты

```yaml
- name: Upload artifact
  uses: actions/upload-artifact@v4
  with:
    name: build
    path: dist/

- name: Download artifact
  uses: actions/download-artifact@v4
  with:
    name: build
```

## Кэширование

```yaml
- name: Cache node modules
  uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
```

## Условия

```yaml
- name: Deploy to prod
  if: github.ref == 'refs/heads/main'
  run: ./deploy-prod.sh
```
""",

    "gitlab_ci.md": """# GitLab CI/CD

CI/CD с GitLab.

## Базовый pipeline

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  NODE_VERSION: "20"

build:
  stage: build
  image: node:${NODE_VERSION}
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/

test:
  stage: test
  image: node:${NODE_VERSION}
  script:
    - npm ci
    - npm test
  coverage: '/Coverage: \\d+\\.\\d+%/'

deploy:
  stage: deploy
  script:
    - ./deploy.sh
  only:
    - main
```

## Переменные

```yaml
variables:
  DATABASE_URL: "postgres://..."

# Или в Settings > CI/CD > Variables
script:
  - echo $CI_PROJECT_NAME
  - echo $MY_SECRET_VAR
```

## Кэширование

```yaml
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - node_modules/
    - .npm/
```

## Сервисы

```yaml
test:
  services:
    - postgres:15
  variables:
    POSTGRES_DB: test
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
  script:
    - npm test
```

## Rules

```yaml
deploy-staging:
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      when: manual

deploy-prod:
  rules:
    - if: $CI_COMMIT_TAG
      when: manual
```

## Environments

```yaml
deploy-staging:
  environment:
    name: staging
    url: https://staging.example.com
  script:
    - ./deploy.sh staging
```
""",

    # Monitoring
    "prometheus.md": """# Prometheus

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
""",

    "grafana.md": """# Grafana

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
""",

    "logging_elk.md": """# ELK Stack

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
""",

    # Databases
    "postgresql.md": """# PostgreSQL

Основы работы с PostgreSQL.

## Подключение

```bash
psql -h localhost -U postgres -d mydb

# Или через URL
psql postgresql://user:password@localhost:5432/mydb
```

## Основные команды psql

```sql
\\l          -- список баз
\\c mydb     -- подключиться к базе
\\dt         -- список таблиц
\\d table    -- структура таблицы
\\du         -- список пользователей
\\q          -- выход
```

## DDL

```sql
-- Создание базы
CREATE DATABASE mydb;

-- Создание таблицы
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_name_gin ON users USING gin(name gin_trgm_ops);
```

## DML

```sql
-- Insert
INSERT INTO users (email, name) VALUES ('test@example.com', 'Test User');

-- Update
UPDATE users SET name = 'New Name' WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;

-- Select
SELECT * FROM users WHERE email LIKE '%@example.com' ORDER BY created_at DESC LIMIT 10;
```

## Транзакции

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- или ROLLBACK;
```

## Backup/Restore

```bash
# Backup
pg_dump mydb > backup.sql
pg_dump -Fc mydb > backup.dump  # сжатый формат

# Restore
psql mydb < backup.sql
pg_restore -d mydb backup.dump
```

## Производительность

```sql
-- План запроса
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- Статистика таблицы
ANALYZE users;

-- Очистка
VACUUM ANALYZE users;
```
""",

    "redis.md": """# Redis

Основы работы с Redis.

## Подключение

```bash
redis-cli
redis-cli -h localhost -p 6379
```

## Типы данных

### Strings

```redis
SET key "value"
GET key
INCR counter
EXPIRE key 3600
TTL key
```

### Lists

```redis
LPUSH mylist "a"
RPUSH mylist "b"
LRANGE mylist 0 -1
LPOP mylist
```

### Sets

```redis
SADD myset "a" "b" "c"
SMEMBERS myset
SISMEMBER myset "a"
SUNION set1 set2
```

### Hashes

```redis
HSET user:1 name "John" age 30
HGET user:1 name
HGETALL user:1
HINCRBY user:1 age 1
```

### Sorted Sets

```redis
ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2"
ZRANGE leaderboard 0 -1 WITHSCORES
ZRANK leaderboard "player1"
```

## Pub/Sub

```redis
# Подписка
SUBSCRIBE channel

# Публикация
PUBLISH channel "message"
```

## Transactions

```redis
MULTI
SET key1 "value1"
SET key2 "value2"
EXEC
```

## Persistence

```redis
# RDB (snapshot)
SAVE
BGSAVE

# AOF (append-only file)
# В redis.conf: appendonly yes
```

## Кластер и Sentinel

```bash
# Sentinel для HA
redis-sentinel /path/to/sentinel.conf

# Кластер
redis-cli --cluster create node1:6379 node2:6379 node3:6379
```
""",

    "mongodb.md": """# MongoDB

Основы работы с MongoDB.

## Подключение

```bash
mongosh
mongosh "mongodb://localhost:27017/mydb"
```

## Основные команды

```javascript
// Показать базы
show dbs

// Использовать базу
use mydb

// Показать коллекции
show collections
```

## CRUD

### Create

```javascript
// Один документ
db.users.insertOne({
  name: "John",
  email: "john@example.com",
  age: 30
})

// Несколько документов
db.users.insertMany([
  { name: "Alice", age: 25 },
  { name: "Bob", age: 35 }
])
```

### Read

```javascript
// Все документы
db.users.find()

// С условием
db.users.find({ age: { $gt: 25 } })

// Проекция
db.users.find({}, { name: 1, email: 1 })

// Сортировка и лимит
db.users.find().sort({ age: -1 }).limit(10)
```

### Update

```javascript
// Один документ
db.users.updateOne(
  { email: "john@example.com" },
  { $set: { age: 31 } }
)

// Несколько документов
db.users.updateMany(
  { age: { $lt: 30 } },
  { $inc: { age: 1 } }
)
```

### Delete

```javascript
db.users.deleteOne({ email: "john@example.com" })
db.users.deleteMany({ age: { $lt: 18 } })
```

## Индексы

```javascript
// Создание индекса
db.users.createIndex({ email: 1 })

// Составной индекс
db.users.createIndex({ name: 1, age: -1 })

// Текстовый индекс
db.users.createIndex({ bio: "text" })

// Просмотр индексов
db.users.getIndexes()
```

## Агрегация

```javascript
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: {
      _id: "$customerId",
      total: { $sum: "$amount" }
  }},
  { $sort: { total: -1 } }
])
```
""",

    # Security
    "security_basics.md": """# Security Basics

Основы безопасности в DevOps.

## Принципы

### Least Privilege
Давать только минимально необходимые права.

### Defense in Depth
Многоуровневая защита.

### Zero Trust
Не доверять ничему по умолчанию.

## SSH Security

```bash
# Генерация ключей
ssh-keygen -t ed25519 -C "user@example.com"

# Конфигурация sshd (/etc/ssh/sshd_config)
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

## Secrets Management

Никогда не храните секреты в:
- Коде
- Переменных окружения в docker-compose
- Git репозитории

Используйте:
- HashiCorp Vault
- AWS Secrets Manager
- Kubernetes Secrets (encrypted)

## HTTPS/TLS

```bash
# Let's Encrypt с certbot
certbot certonly --webroot -w /var/www/html -d example.com

# Проверка сертификата
openssl s_client -connect example.com:443
openssl x509 -in cert.pem -text
```

## Container Security

```dockerfile
# Не root пользователь
FROM alpine
RUN adduser -D appuser
USER appuser

# Минимальный образ
FROM scratch
COPY --from=builder /app /app
```

```bash
# Сканирование образов
trivy image myapp:latest
docker scout cves myapp:latest
```

## Network Security

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```
""",

    "ssl_tls.md": """# SSL/TLS

Работа с сертификатами.

## Создание самоподписанного сертификата

```bash
# Ключ и сертификат
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem

# Или с SAN
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -subj "/CN=example.com" \
  -addext "subjectAltName=DNS:example.com,DNS:*.example.com"
```

## Let's Encrypt

```bash
# Установка certbot
apt install certbot python3-certbot-nginx

# Получение сертификата
certbot --nginx -d example.com -d www.example.com

# Автообновление
certbot renew --dry-run
```

## Конфигурация Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
}
```

## Проверка

```bash
# Проверка сертификата
openssl x509 -in cert.pem -text -noout

# Проверка цепочки
openssl verify -CAfile ca.pem cert.pem

# Проверка соединения
openssl s_client -connect example.com:443 -servername example.com

# Дата истечения
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates
```

## mTLS

```bash
# Генерация CA
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt

# Генерация клиентского сертификата
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -out client.crt
```
""",

    # Terraform
    "terraform_basics.md": """# Terraform Basics

Инфраструктура как код с Terraform.

## Установка

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
mv terraform /usr/local/bin/
```

## Основные команды

```bash
terraform init      # Инициализация
terraform plan      # Просмотр изменений
terraform apply     # Применение
terraform destroy   # Удаление
terraform fmt       # Форматирование
terraform validate  # Валидация
```

## Базовая структура

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_instance" "web" {
  ami           = var.ami_id
  instance_type = var.instance_type

  tags = {
    Name = "web-server"
  }
}
```

## Переменные

```hcl
# variables.tf
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  type    = string
  default = "t3.micro"
}
```

```hcl
# terraform.tfvars
region        = "eu-west-1"
instance_type = "t3.small"
```

## Outputs

```hcl
# outputs.tf
output "instance_ip" {
  value       = aws_instance.web.public_ip
  description = "Public IP of the instance"
}
```

## Data Sources

```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}
```
""",

    "terraform_modules.md": """# Terraform Modules

Модули для переиспользования кода.

## Структура модуля

```
modules/
└── vpc/
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── README.md
```

## Создание модуля

```hcl
# modules/vpc/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true

  tags = {
    Name = var.name
  }
}

resource "aws_subnet" "public" {
  count             = length(var.public_subnets)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.public_subnets[count.index]
  availability_zone = var.azs[count.index]

  tags = {
    Name = "${var.name}-public-${count.index + 1}"
  }
}
```

```hcl
# modules/vpc/variables.tf
variable "name" {
  type = string
}

variable "cidr_block" {
  type    = string
  default = "10.0.0.0/16"
}

variable "public_subnets" {
  type    = list(string)
  default = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "azs" {
  type = list(string)
}
```

```hcl
# modules/vpc/outputs.tf
output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}
```

## Использование модуля

```hcl
# main.tf
module "vpc" {
  source = "./modules/vpc"

  name           = "production"
  cidr_block     = "10.0.0.0/16"
  public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  azs            = ["us-east-1a", "us-east-1b"]
}

output "vpc_id" {
  value = module.vpc.vpc_id
}
```

## Remote модули

```hcl
# Из Terraform Registry
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
}

# Из Git
module "vpc" {
  source = "git::https://github.com/user/repo.git//modules/vpc?ref=v1.0.0"
}
```

## Workspaces

```bash
terraform workspace new staging
terraform workspace new production
terraform workspace select staging
terraform workspace list
```
""",

    # Ansible
    "ansible_basics.md": """# Ansible Basics

Автоматизация с Ansible.

## Установка

```bash
pip install ansible
```

## Inventory

```ini
# inventory.ini
[web]
web1.example.com
web2.example.com

[db]
db1.example.com ansible_user=admin

[all:vars]
ansible_python_interpreter=/usr/bin/python3
```

## Ad-hoc команды

```bash
# Ping все хосты
ansible all -i inventory.ini -m ping

# Выполнить команду
ansible web -i inventory.ini -m shell -a "uptime"

# Установить пакет
ansible web -i inventory.ini -m apt -a "name=nginx state=present" --become
```

## Playbook

```yaml
# site.yml
---
- name: Configure web servers
  hosts: web
  become: yes

  vars:
    nginx_port: 80

  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
        update_cache: yes

    - name: Start nginx
      service:
        name: nginx
        state: started
        enabled: yes

    - name: Copy config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: Restart nginx

  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
```

```bash
# Запуск
ansible-playbook -i inventory.ini site.yml
```

## Переменные

```yaml
# group_vars/web.yml
nginx_port: 80
app_user: www-data

# host_vars/web1.example.com.yml
nginx_port: 8080
```

## Templates (Jinja2)

```jinja2
# templates/nginx.conf.j2
server {
    listen {{ nginx_port }};
    server_name {{ ansible_hostname }};
}
```
""",

    "ansible_roles.md": """# Ansible Roles

Организация кода с ролями.

## Структура роли

```
roles/
└── nginx/
    ├── tasks/
    │   └── main.yml
    ├── handlers/
    │   └── main.yml
    ├── templates/
    │   └── nginx.conf.j2
    ├── files/
    │   └── index.html
    ├── vars/
    │   └── main.yml
    ├── defaults/
    │   └── main.yml
    └── meta/
        └── main.yml
```

## Создание роли

```bash
ansible-galaxy init roles/nginx
```

## Пример роли

```yaml
# roles/nginx/tasks/main.yml
---
- name: Install nginx
  apt:
    name: nginx
    state: present
  tags: nginx

- name: Configure nginx
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: Restart nginx

- name: Ensure nginx is running
  service:
    name: nginx
    state: started
    enabled: yes
```

```yaml
# roles/nginx/handlers/main.yml
---
- name: Restart nginx
  service:
    name: nginx
    state: restarted
```

```yaml
# roles/nginx/defaults/main.yml
---
nginx_port: 80
nginx_worker_processes: auto
```

## Использование ролей

```yaml
# site.yml
---
- hosts: web
  become: yes
  roles:
    - nginx
    - { role: app, app_port: 8080 }
    - role: monitoring
      when: enable_monitoring
```

## Ansible Galaxy

```bash
# Установить роль
ansible-galaxy install geerlingguy.docker

# requirements.yml
roles:
  - name: geerlingguy.docker
    version: "6.1.0"

ansible-galaxy install -r requirements.yml
```

## Collections

```bash
# Установить collection
ansible-galaxy collection install community.docker

# Использование
- name: Run container
  community.docker.docker_container:
    name: nginx
    image: nginx:latest
```
""",

    # Nginx
    "nginx_config.md": """# Nginx Configuration

Конфигурация Nginx.

## Базовая структура

```nginx
# /etc/nginx/nginx.conf
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    sendfile on;
    keepalive_timeout 65;

    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

## Server Block

```nginx
# /etc/nginx/sites-available/example.com
server {
    listen 80;
    server_name example.com www.example.com;
    root /var/www/example.com;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location ~* \\.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## Reverse Proxy

```nginx
upstream backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Load Balancing

```nginx
upstream backend {
    least_conn;  # или ip_hash, round_robin
    server backend1.example.com weight=3;
    server backend2.example.com;
    server backend3.example.com backup;
}
```

## Rate Limiting

```nginx
limit_req_zone $binary_remote_addr zone=mylimit:10m rate=10r/s;

server {
    location /api/ {
        limit_req zone=mylimit burst=20 nodelay;
    }
}
```
""",

    # AWS
    "aws_ec2.md": """# AWS EC2

Работа с EC2 инстансами.

## AWS CLI

```bash
# Настройка
aws configure

# Список инстансов
aws ec2 describe-instances

# Запуск инстанса
aws ec2 run-instances \\
    --image-id ami-0123456789 \\
    --instance-type t3.micro \\
    --key-name my-key \\
    --security-group-ids sg-123456

# Остановка
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Терминация
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

## Security Groups

```bash
# Создание
aws ec2 create-security-group \\
    --group-name my-sg \\
    --description "My security group"

# Добавление правила
aws ec2 authorize-security-group-ingress \\
    --group-id sg-123456 \\
    --protocol tcp \\
    --port 22 \\
    --cidr 0.0.0.0/0
```

## Key Pairs

```bash
# Создание
aws ec2 create-key-pair --key-name my-key --query 'KeyMaterial' --output text > my-key.pem
chmod 400 my-key.pem

# Подключение
ssh -i my-key.pem ec2-user@<public-ip>
```

## AMI

```bash
# Создание AMI из инстанса
aws ec2 create-image \\
    --instance-id i-1234567890abcdef0 \\
    --name "My AMI"

# Список AMI
aws ec2 describe-images --owners self
```

## EBS

```bash
# Создание volume
aws ec2 create-volume \\
    --size 100 \\
    --volume-type gp3 \\
    --availability-zone us-east-1a

# Присоединение к инстансу
aws ec2 attach-volume \\
    --volume-id vol-123456 \\
    --instance-id i-123456 \\
    --device /dev/sdf
```

## User Data

```bash
#!/bin/bash
yum update -y
yum install -y httpd
systemctl start httpd
systemctl enable httpd
```
""",

    "aws_s3.md": """# AWS S3

Работа с S3.

## Основные команды

```bash
# Список бакетов
aws s3 ls

# Создание бакета
aws s3 mb s3://my-bucket

# Загрузка файла
aws s3 cp file.txt s3://my-bucket/
aws s3 cp file.txt s3://my-bucket/folder/

# Скачивание
aws s3 cp s3://my-bucket/file.txt .

# Синхронизация
aws s3 sync ./local-folder s3://my-bucket/remote-folder
aws s3 sync s3://my-bucket/remote-folder ./local-folder

# Удаление
aws s3 rm s3://my-bucket/file.txt
aws s3 rm s3://my-bucket --recursive  # все файлы

# Удаление бакета
aws s3 rb s3://my-bucket --force
```

## Bucket Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        }
    ]
}
```

```bash
aws s3api put-bucket-policy --bucket my-bucket --policy file://policy.json
```

## Versioning

```bash
# Включить версионирование
aws s3api put-bucket-versioning \\
    --bucket my-bucket \\
    --versioning-configuration Status=Enabled

# Список версий
aws s3api list-object-versions --bucket my-bucket
```

## Lifecycle Rules

```json
{
    "Rules": [
        {
            "ID": "Move to Glacier",
            "Status": "Enabled",
            "Filter": {"Prefix": "logs/"},
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": 365
            }
        }
    ]
}
```

## Pre-signed URLs

```bash
# Генерация URL для скачивания (действителен 1 час)
aws s3 presign s3://my-bucket/file.txt --expires-in 3600
```
""",
}


def generate_docs():
    """Generate mock documentation files."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in MOCK_DOCS.items():
        filepath = DOCS_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        print(f"Created: {filepath}")

    print(f"\nGenerated {len(MOCK_DOCS)} mock documents in {DOCS_DIR}")


if __name__ == "__main__":
    generate_docs()
