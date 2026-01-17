# SSL/TLS

Работа с сертификатами.

## Создание самоподписанного сертификата

```bash
# Ключ и сертификат
openssl req -x509 -nodes -days 365 -newkey rsa:2048   -keyout key.pem -out cert.pem

# Или с SAN
openssl req -x509 -nodes -days 365 -newkey rsa:2048   -keyout key.pem -out cert.pem   -subj "/CN=example.com"   -addext "subjectAltName=DNS:example.com,DNS:*.example.com"
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
