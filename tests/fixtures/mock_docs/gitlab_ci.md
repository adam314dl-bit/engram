# GitLab CI/CD

CI/CD с GitLab.

## Базовый pipeline

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  NODE_VERSION: "20"

build:
  stage: build
  image: node:${NODE_VERSION}
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/

test:
  stage: test
  image: node:${NODE_VERSION}
  script:
    - npm ci
    - npm test
  coverage: '/Coverage: \d+\.\d+%/'

deploy:
  stage: deploy
  script:
    - ./deploy.sh
  only:
    - main
```

## Переменные

```yaml
variables:
  DATABASE_URL: "postgres://..."

# Или в Settings > CI/CD > Variables
script:
  - echo $CI_PROJECT_NAME
  - echo $MY_SECRET_VAR
```

## Кэширование

```yaml
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - node_modules/
    - .npm/
```

## Сервисы

```yaml
test:
  services:
    - postgres:15
  variables:
    POSTGRES_DB: test
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
  script:
    - npm test
```

## Rules

```yaml
deploy-staging:
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      when: manual

deploy-prod:
  rules:
    - if: $CI_COMMIT_TAG
      when: manual
```

## Environments

```yaml
deploy-staging:
  environment:
    name: staging
    url: https://staging.example.com
  script:
    - ./deploy.sh staging
```
