# Systemd

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
