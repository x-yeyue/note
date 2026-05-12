# 设置身份信息

## 设置SSH连接方式

通过公钥和私钥来验证身份。
`ssh-keygen -t rsa -C email.com`
执行该命令会生成公钥和私钥。
默认生成至该目录的.ssh文件夹中。
`id_rsa`为私钥，`id_rsa.pub`为公钥

`ssh -T git@gitee.com`: 验证本地和 gitee 的ssh链接是否正确。

`ssh -T git@github.com`: 验证本地和 github 的ssh链接是否正确。

# 操作

## Git分支管理策略

Git分支是一个独立的工作流，使用分支，开发者可以互相隔离的来进行开发，而不会影响到仓库和其他分支。

## Git工作原理

### 文件的四种状态

**Untracked未跟踪**
**Modifiled已修改**
**Staged已暂存**
**Committed已提交**

## 基本操作

### 本地仓库

- `git status`: 查看当前仓库状态
    当不知道下一步该如何做时可以执行该命令查看下仓库的状态，根据提示进行下一步。
- `git log`: 查看提交日志。

- `git add`: 将文件添加到暂存区。后面可以是多个文件名  ，可以以是`.` 将所有修改文件都添加。

- `git revert commitID`: 产生一个到该版本之前一个版本的新提交。(回退)

- `git reset commitID`: 回退到该版本。(删除该版本到目前版本之间的记录)

    可以通过 `git reflog` 尝试查看 `git` 操作记录来找回。

- `git commit -m “说明信息”`: 把修改提交到仓库。 `-m`后面跟本次提交的说明信息。应明确此次提交进行了那些变动，比如增加了哪些新功能，还是修复了某些bug。例如`git commit -m "x_yeyue完成开发V1"`。
- `git switch dev`: 切换到已有分支 `dev`。
    - `git switch -c dev`: 创建新分支 `dev` 并切换到该分支。

- `git branch`: 显示所有的本地分支。
    - `git branch dev`: 创建一个新分支 `dev`。
    - `git branch -d dev`: 删除分支 `dev`。当该分支未合并到当前分支时会报错，拒绝删除。
    - `git branch -D dev`: 不做任何检查强制删除分支 `dev`。有较高风险。

- `git merge --no-ff -m “说明信息” 开发分支`: 合并分支。`--no-ff`可以更好的保留历史记录。执行该命令会自动生成一个提交。

### 远程仓库

- `git remote`: 查看远程仓库。
    - `git remote -v`: 查看当前的远程仓库的详细信息。

- `git remote add name url`: 添加名为 `name` 的远程仓库。路径为 `utl`，推荐使用 `ssh`。

- `git push [-f] [--set-upstream] [远程仓库名 [本地分支名称][:远程分支名称]]`

    - `-f`: 强制覆盖。

    - `git push origin dev`: 将`dev`分支向远程仓库 `origin` 推送。
    - `git push orgin dev:dev01`: 将本地的 `dev` 分支想远程仓库 `origin` 的 `dev01` 分支推送。

- `git fetch [远程仓库名] [分支名]`: 抓取远程仓库的指定分支到本地，不会进行合并。

- `git pull [远程仓库名] [分支名]`: 拉取远程仓库的指定分支到本地并自动进行合并。等同 fetch+merge。

- `git remote rename old_name new_name`: 重命名远程仓库名称。

- `git remote set-url name new_url`: 更改远程仓库的链接。

## 撤销

`git restore 文件名`: 丢弃工作区的修改

`git restore --staged 文件名`: 取消暂存文件



获取到需要回退的哈希ID复制。


# github

> **官方文档**: [https://docs.github.com/zh](https://docs.github.com/zh "gitbub官方文档")


# .gitignore文件

> `.gitignore` 是 Git 中核心的忽略配置文件，指*不需要纳入版本控制*的文件/目录。

## 通配符

| 通配符  |         作用          |
| :--: | :-----------------: |
| `*`  | 匹配任意数量的任意字符(不包含`/`) |
| `?`  |  匹配单个任意字符(不包含`/`)   |
| `[]` |     匹配括号内的单个字符      |
| `**` |      匹配任意层级的目录      |

**注**: 若文件已经被 Git 追踪，修改 `.gitignore` 无法忽略该文件。
*解决方案*: 

```bash
# 移除单个文件的追踪(本地文件保留)
git rm --cached filename

# 移除整个目录的追踪
git rm --cached -r Filename

# 提交 .gitignore 和移除追踪的变更
git commit -m "change .gitignore -untrack filename"
```

`git check-ignore -v filename`: 查看文件是否被忽略。
`git ls-files --ignored --exclude-standard`: 列出所有被忽略的文件。	