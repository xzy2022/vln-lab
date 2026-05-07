---
name: git-commit
description: 用于提交暂存代码.
disable-model-invocation: true
---

# git-commit

**Conventional Commits**：

type(scope): summary

例如：

- feat(loader): 添加地形缓存以加快启动速度

- fix(readme): 修正 STEGO 权重下载链接

- docs(setup): 阐明 ROS 安装步骤

- ci(pre-commit): 阻止大于 2 MB 的文件提交

具体的信息使用中文

根据上面要求去提交当前的暂存.如果当前的暂存有多条修改,你需要阅读修改,进行分组多次提交;如果你认为多条修改应该一次同时提交,那也没问题.

但是每一次提交应该是相关的最小功能单位,这样我只看message就知道这次提交做了什么.

你只需要提交,不用push