# GitHub Actions

CI/CD с GitHub Actions.

## Базовый workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run tests
      run: npm test

    - name: Build
      run: npm run build
```

## Матрица стратегий

```yaml
jobs:
  test:
    strategy:
      matrix:
        node: [18, 20]
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
    - uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node }}
```

## Секреты

```yaml
steps:
- name: Deploy
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: ./deploy.sh
```

## Артефакты

```yaml
- name: Upload artifact
  uses: actions/upload-artifact@v4
  with:
    name: build
    path: dist/

- name: Download artifact
  uses: actions/download-artifact@v4
  with:
    name: build
```

## Кэширование

```yaml
- name: Cache node modules
  uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
```

## Условия

```yaml
- name: Deploy to prod
  if: github.ref == 'refs/heads/main'
  run: ./deploy-prod.sh
```
