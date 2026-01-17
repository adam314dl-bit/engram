# Linux Networking

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
