# AWS S3

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
aws s3api put-bucket-versioning \
    --bucket my-bucket \
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
