# Git Basics

Основы работы с Git.

## Инициализация

```bash
# Новый репозиторий
git init

# Клонирование
git clone https://github.com/user/repo.git
```

## Базовый workflow

```bash
# Статус
git status

# Добавить файлы
git add file.txt
git add .  # все файлы

# Коммит
git commit -m "Описание изменений"

# Push
git push origin main
```

## Ветки

```bash
# Создать ветку
git branch feature

# Переключиться
git checkout feature
git switch feature  # новый синтаксис

# Создать и переключиться
git checkout -b feature
git switch -c feature

# Список веток
git branch -a

# Удалить ветку
git branch -d feature
```

## Слияние

```bash
# Merge
git checkout main
git merge feature

# Rebase
git checkout feature
git rebase main
```

## История

```bash
# Лог
git log
git log --oneline --graph

# Изменения в коммите
git show commit_hash

# Разница
git diff
git diff --staged
```

## Отмена изменений

```bash
# Отменить изменения в файле
git checkout -- file.txt
git restore file.txt

# Убрать из staging
git reset HEAD file.txt
git restore --staged file.txt

# Отменить коммит (сохранить изменения)
git reset --soft HEAD~1

# Отменить коммит (удалить изменения)
git reset --hard HEAD~1
```
