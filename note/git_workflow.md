# Git Workflow Checklist for Feature Development

## 1.  Clone the repository
```bash
git clone <repo_url>
cd <project_name>
```

## 2. Switch to the development branch

```bash
git switch dev

# If you don’t have a local dev branch yet:
git checkout -b dev origin/dev
```

## 3. Before you start: update your local code
```bash
git switch dev
git fetch origin --prune
git rebase origin/dev    # or: git pull --rebase --autostash

```

## 4. Start a new feature: create a branch

```bash
git switch dev
git fetch origin
git rebase origin/dev
git switch -c feature/xxx


```

## 5.During development
```bash
# Inspect changes after editing
git status
git diff
```


```bash
# Stage changes (prefer explicit files or -p)
git add -p
# or, if you must:
# git add .
```

```bash
# Commit
git commit -m "feat: describe the feature"
```

## 6. Push to remote
```bash
# First push: set upstream so later you can just `git push`
git push -u origin feature/xxx
```

## 7. Open a Pull Request (PR)
* Create a PR on GitHub / GitLab.

* Set the target branch to dev.

* After code review is approved, merge.

## 8. Sync with the latest dev (dev by team, feature/xxx by yourself)
```bash
# do it on your feature branch directly
git switch feature/xxx
git fetch origin --prune
git rebase origin/dev           # or: git merge origin/dev

```

## 9.Clean up stale branches
```bash
# Delete local branch
git branch -d feature/xxx

# Delete remote branch
git push origin --delete feature/xxx

# Prune local tracking refs that no longer exist on remote
git fetch --prune
```
