# Git Workflow Checklist for Feature Development

## 1.  Clone the repository
```
git clone <repo_url>
cd <project_name>
```

## 2. Switch to the development branch

```
git switch dev

# If you don’t have a local dev branch yet:
git checkout -b dev origin/dev
```

## 3. Before you start: update your local code
```
git switch dev
git fetch origin
git rebase origin/dev    # or: git pull --rebase

```

## 4. Start a new feature: create a branch

```
git switch dev
git fetch origin
git rebase origin/dev
git switch -c feature/xxx


```

## 5.During development
```
# Inspect changes after editing
git status
git diff
```


```
# Stage changes
git add .
```

```
# Commit
git commit -m "feat: describe the feature"
```

## 6. Push to remote
```
# First push: set upstream so later you can just `git push`
git push -u origin feature/xxx
```

## 7. Open a Pull Request (PR)
* Create a PR on GitHub / GitLab.

* Set the target branch to dev.

* After code review is approved, merge.

## 8. Sync with the latest dev
```
git switch dev
git fetch origin
git rebase origin/dev

git switch feature/xxx
# Bring your branch up to date with dev
git rebase dev            # preferred for linear history
# (Alternatively) git merge dev

```

## 9.Clean up stale branches
```
# Delete local branch
git branch -d feature/xxx

# Delete remote branch
git push origin --delete feature/xxx
```
