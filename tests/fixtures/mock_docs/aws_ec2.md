# AWS EC2

Работа с EC2 инстансами.

## AWS CLI

```bash
# Настройка
aws configure

# Список инстансов
aws ec2 describe-instances

# Запуск инстанса
aws ec2 run-instances \
    --image-id ami-0123456789 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-group-ids sg-123456

# Остановка
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Терминация
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

## Security Groups

```bash
# Создание
aws ec2 create-security-group \
    --group-name my-sg \
    --description "My security group"

# Добавление правила
aws ec2 authorize-security-group-ingress \
    --group-id sg-123456 \
    --protocol tcp \
    --port 22 \
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
aws ec2 create-image \
    --instance-id i-1234567890abcdef0 \
    --name "My AMI"

# Список AMI
aws ec2 describe-images --owners self
```

## EBS

```bash
# Создание volume
aws ec2 create-volume \
    --size 100 \
    --volume-type gp3 \
    --availability-zone us-east-1a

# Присоединение к инстансу
aws ec2 attach-volume \
    --volume-id vol-123456 \
    --instance-id i-123456 \
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
