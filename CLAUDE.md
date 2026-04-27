# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 工作环境

所有 SAME 实验在 Docker 容器内运行。没有本地 Python 环境可直接使用。

```bash
# 启动 SAME 实验容器（默认工作环境）
bash scripts/setup/run_container.sh
conda activate test-v1

# 启动 Matterport3D 模拟器容器（用于渲染、连接图分析）
bash scripts/setup/run_mp3dsim_container.sh
```

不要尝试在容器外直接运行实验或分析脚本。

## 运行实验

```bash
# 单个 R2R 验证
python scripts/experiments/run_same.py --config configs/same/val_r2r_eval_only.yaml

# 全量 unseen（4 个数据集）
python scripts/experiments/run_same.py --config configs/same/val_r2r_reverie_cvdn_soon.yaml

# 可选参数
python scripts/experiments/run_same.py --config <config> --seed 0 --checkpoint ../../../data/same/ckpt/SAME.pt --tag smoke --option training.workers=2
```

`run_same.py` 会自动：应用 patches → 分配 experiment_id → 启动 SAME 子进程 → 归档产物并更新 `reports/tables/`。

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 单个测试文件
python -m pytest tests/test_run_same.py -v

# 单个测试用例
python -m pytest tests/test_run_same.py -k test_experiment_id_format -v
```

测试基于 `unittest`，通过 `importlib` 动态导入被测试的脚本模块。

## 分析命令

```bash
# 细粒度指标（需要先在 SAME 容器内运行）
python scripts/analysis/build_same_fine_metrics.py \
  --experiment-dir experiment_outputs/<experiment_id> \
  --connectivity-dir data/same/simulator/connectivity

# 逐样本轨迹渲染（需要在 MP3D 模拟器容器内运行）
python scripts/analysis/render_same_eval_item.py \
  <internal_item_id> \
  --experiment-dir experiment_outputs/<experiment_id> \
  --download-missing-scan
```

## 架构概览

### 核心设计原则（来自 开发说明.md）

1. **优先从父项目适配**：新增脚本、配置、归档流程，不直接改 `third_party/` 源码
2. **必要时用 patches**：修改子项目代码必须通过 `patches/` 下的 git patch 管理，运行前自动 apply
3. **不要直接改 `third_party/` 工作树**

### 实验生命周期

一个 SAME 实验的完整流程：

1. 开发者在 `configs/same/` 创建 YAML 配置（只覆盖需要改的字段，其余从 `third_party/SAME/src/configs/default.yaml` 继承）
2. `run_same.py` 应用 `patches/same/base/` 下的 patches 到 `third_party/SAME`
3. 分配规范化的 `experiment_id`（格式：`NNNN_same_<config-stem>_<ckpt-tag>_s<seed>[_<tag>]_v<rev>`，全仓库单调递增）
4. 合并配置 → 写入 `config_resolved.yaml`
5. 记录 GPU 信息、数据清单、git 信息、patch diff
6. 以子进程方式运行 `third_party/SAME/src/run.py`，实时流式输出日志
7. 从日志解析指标 → 写入 `metrics.json`，与 results JSON 交叉校验，与 `official_results.csv` 做只读对照
8. 追加行到 `reports/tables/runs.csv` 和 `reports/tables/metrics_long.csv`

### 实验归档结构

每次实验在 `experiment_outputs/<experiment_id>/` 下至少包含 9 个文件：

```
run.json              # 运行元数据（命令、状态、commit、patch_set）
config_resolved.yaml  # 合并后的最终配置
metrics.json          # 结构化 split-level 指标摘要
stdout.log / stderr.log
git_info.txt / data_manifest.txt / gpu_info.txt / patch.diff
```

外加 SAME 原始产物：`results/`、`ckpts/`、`tensorboard/`。

### 长期报表系统

`reports/tables/` 下维护三张 CSV：

- `runs.csv` — 每次实验的运行记录（experiment_id, commit, config, seed, status, duration 等）
- `metrics_long.csv` — 摊平的指标长表（experiment_id, dataset, split, metric, value, unit），便于筛选和绘图
- `official_results.csv` — 人工维护的官方参考结果，运行器只读不写

### 第三方子模块（third_party/）

- **SAME** (`third_party/SAME`) — 核心 VLN 模型，`src/run.py` 是入口。本项目通过 5 个 base patches 对其做了修改：eval-only 退出、stdout 重定向、CVDN/SOON 路径指标、eval items sidecar、决策轨迹导出
- **Matterport3DSimulator** (`third_party/Matterport3DSimulator`) — C++ 3D 模拟器，需要 CMake 构建 + EGL，用于连接图、候选视点、渲染验证
- **VLN-DUET** 和 **navnuances** — 计划用于实验，尚未集成

### 配置系统

`configs/same/` 下的 YAML 文件经过 OmegaConf 与 `third_party/SAME/src/configs/default.yaml` 合并。配置分为几个顶层组：`experiment.*`（输出路径、eval_only 标志）、`training.*`、`model.*`（MoE、预训练权重）、`simulator.*`（连接图、候选路径）、`task.*`（数据源、评估 split）。

### 数据组织

`data/` 下按项目分目录（`data/same/`、`data/duet/`、`data/navnuances/`），每个项目内按类型分：`ckpt/`（检查点）、`R2R/`（编码数据集）、`simulator/`（连接图）、`features/`（图像特征 HDF5）。大型文件通过 `.gitignore` 排除。

### v1 边界

当前明确不做：历史结果回填、SAME 以外方法的通用抽象、`official_results.csv` 自动补齐、新的 SAME 指标导出 patch。
