# Git 开发分支常用工作流程清单

## 1. 克隆项目（第一次）
```
git clone <repo_url>
cd <project_name>
```

## 2. 切换到开发分支

```
git switch dev

# 如果本地还没有 dev 分支
git checkout -b dev origin/dev
```

## 3. 开始工作前：更新代码
```
# 确保在 dev 分支
git switch dev 

# 拉取远程最新代码并合并
git pull origin dev
```

## 4. 开发新功能： 新建分支 

```
# 从dev 分支新建功能分支
git switch dev 
git pull origin dev
git checkout -b feature/xxx
```

## 5.开发过程
```
# 编辑代码后，先查看改动
git status
git diff
```


```
# 添加改动
git add .
```

```
# 提交
git commit -m "feat: 功能描述"
```

## 6.推送到远程
```
# 推送当前分支到远程
git push origin feature/xxx
```

## 7. 提交合并请求（PR）
1. 在 GitHub / GitLab 上创建 PR

2. 选择合并到 dev 分支

3. 代码评审 (Review) 通过后合并

## 8. 同步最新dev 分支
```
# 当你的功能分支开发一半，但 dev 有新更新时：
git switch dev
git pull origin dev
git switch feature/xxx
git merge dev   # 或 git rebase dev
```

## 9.清理无用分支
```
# 删除本地分支
git branch -d feature/xxx

# 删除远程分支
git push origin --delete feature/xxx
```
