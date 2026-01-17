# Terraform Basics

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
