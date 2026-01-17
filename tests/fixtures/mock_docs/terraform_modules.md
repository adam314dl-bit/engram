# Terraform Modules

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
