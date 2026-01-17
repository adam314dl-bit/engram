# Git Advanced

Продвинутые техники Git.

## Interactive Rebase

```bash
# Изменить последние N коммитов
git rebase -i HEAD~3
```

Команды:
- `pick` — оставить коммит
- `reword` — изменить сообщение
- `squash` — объединить с предыдущим
- `fixup` — объединить без сообщения
- `drop` — удалить коммит

## Cherry-pick

```bash
# Применить конкретный коммит
git cherry-pick commit_hash
```

## Stash

```bash
# Сохранить изменения
git stash
git stash push -m "описание"

# Список
git stash list

# Применить
git stash pop
git stash apply stash@{0}

# Удалить
git stash drop stash@{0}
```

## Bisect

```bash
# Найти коммит с багом
git bisect start
git bisect bad  # текущий коммит плохой
git bisect good commit_hash  # этот был хороший

# После каждой проверки
git bisect good  # или
git bisect bad

# Завершить
git bisect reset
```

## Worktree

```bash
# Создать дополнительный worktree
git worktree add ../feature-branch feature

# Список
git worktree list

# Удалить
git worktree remove ../feature-branch
```

## Submodules

```bash
# Добавить
git submodule add https://github.com/user/lib libs/lib

# Клонировать с submodules
git clone --recursive repo

# Обновить
git submodule update --init --recursive
```

## Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
npm run lint
```
