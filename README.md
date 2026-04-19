# vln-lab

一个统一的研究工作空间，用于复现和扩展现代 VLN 基线。

## 结构

- `third_party/`：上游仓库，以 Git 子模块形式管理
- `data/`：共享数据集、特征、检查点
- `scripts/`：设置/冒烟/评估/训练入口点
- `envs/`：conda 环境文件
- `docker/`：Dockerfile 和容器配置
- `docs/`：环境、数据集、补丁、协议文档
- `patches/`：针对上游仓库的本地源补丁
- `reports/`：复现卡片、注释、消融报告、日志

## 初始化

```bash

git clone --recursive <your_repo_url>

```
