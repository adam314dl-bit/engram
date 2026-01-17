# Security Basics

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
