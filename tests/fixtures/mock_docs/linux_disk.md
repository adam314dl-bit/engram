# Linux Disk Management

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
