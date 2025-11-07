# Git Workflow Checklist 

## 1.  Clone the repository
```bash
git clone <repo_url>
cd <project_name>
```

## 2. Initialize local main

```bash
git fetch origin --prune
git switch -c main origin/main   # If main exists: git switch main && git pull
```

## 3. Before you start: update your local code
```bash
git switch main
git fetch origin --prune
git pull --rebase --autostash    # same effect as: git rebase origin/main


```

## 4. Start a feature branch (from latest main)

```bash
git switch -c feature/xxx origin/main

```

## 5.During development
```bash
# Inspect changes
git status
git diff

# Stage changes (prefer interactive / explicit)
git add -p           # or: git add <file1> <file2>

# Commit
git commit -m "feat: describe the feature"

```

## 6. First push (set upstream)
```bash
# First push: set upstream so later you can just `git push`
git push -u origin feature/xxx
```

## 7. Open a Pull Request (PR)
* Create a PR on GitHub / GitLab.

* Set the target branch to dev.

* After code review is approved, merge.

## 8. Keep your branch in sync with main (repeat during dev)

A) Merge (no history rewrite)
```bash
git switch feature/xxx
git fetch origin --prune
git merge origin/main
git push
```

B) Rebase (linear, cleaner PR)
```bash
git switch feature/xxx
git fetch origin --prune
git rebase origin/main
# if conflicts: fix -> git add <file> -> git rebase --continue
git push --force-with-lease
```



## 9 After merge: start next work
Want next PR to include only new changes:
```bash
git switch -c feature/xxx-next origin/main
```
Or continue on the same branch (first sync with main as in step 8).

## Clean up stale branches (when no longer needed)
```bash
git branch -d feature/xxx                 # local
git push origin --delete feature/xxx      # remote
git fetch --prune                         # prune tracking refs
```
