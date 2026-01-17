# Linux Process Management

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
