# S.01.01.01.A1-SAME评估接口确认

## 目标 / 任务

确认 `L.01.01.Ability Atlas 构建` 中阶段 A 的 A.1 是否已经满足：

| 检查项 | 成功标准 | 本轮结论 |
|---|---|---|
| `data/navnuances/` 目录结构与数据格式 | 原始 NavNuances annotation 可定位，且可转换为 SAME 可读 R2R eval split | 通过 |
| SAME `task.*` 配置 | SAME 能以 R2R eval 形式覆盖 DC/LR/RR/VM/NU，必要时包含 Standard `val_unseen` | 通过 |
| SAME 输出可解析性 | 至少能解析 per-episode trajectory；若有决策轨迹，应能支持后续 failure 诊断 | 通过 |
| A.1 停止条件 | 若 SAME 完全不支持 NavNuances 格式，则路线 A 需重新评估 | 未触发 |

本 stage 只确认 SAME 评估接口是否可用，不完成 NavGPT 对齐、Ability Atlas 统计脚本或 skill × evidence 矩阵解释。

## 背景与输入

### 上游行动计划

| 来源 | 对本 stage 的要求 |
|---|---|
| `research/VLN research/local_traces/L.01.01.Ability Atlas 构建.md` | A.1 要确认 SAME 在 NavNuances 数据集上的评估接口，包括数据目录、SAME split 配置、trajectory/action sequence 可解析性 |
| `开发说明.md` | NavNuances 通过伪装成 SAME 的 R2R eval split 运行推理，再使用 NavNuances evaluator 评分 |
| `docs/patches/same.md` | SAME base patch 已支持 `eval_items` sidecar 与 `decision_trace`，可用于逐样本诊断 |
| `docs/patches/navnuances.md` | NavNuances evaluator patch 已支持显式传入 connectivity 目录，并可按需跳过 Standard split |

### 已有工程入口

| 功能 | 路径 | 状态 |
|---|---|---|
| NavNuances 转 SAME R2R 编码 | `scripts/setup/prepare_navnuances_same_r2r.py` | 已实现 |
| SAME NavNuances eval 配置 | `configs/same/navnuances_r2r_eval_only.yaml` | 已实现 |
| SAME submission 导出 | `scripts/eval/export_same_navnuances_submissions.py` | 已实现 |
| SAME + NavNuances evaluator 统一入口 | `scripts/eval/run_same_navnuances_eval.py` | 已实现 |
| SAME 逐样本 sidecar | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/eval_items/` | 已生成 |
| NavNuances evaluator 结果 | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/navnuances_eval/results.json` | 已生成 |

## 执行记录

### 对行动计划各项的逐条回应

#### 1. 检查 `data/navnuances/` 目录结构与数据格式

NavNuances 原始 annotation 与 SAME 编码后的 R2R eval 文件均存在。五类 NavNuances split 的原始条目数、编码条目数、SAME 结果条目数和 `eval_items` 条目数一致。

| Split | 原始 annotation | SAME encoded | SAME results | eval_items |
|---|---:|---:|---:|---:|
| DC | 579 | 579 | 579 | 579 |
| LR | 685 | 685 | 685 | 685 |
| RR | 275 | 275 | 275 | 275 |
| VM | 170 | 170 | 170 | 170 |
| NU | 78 | 78 | 78 | 78 |
| Standard `val_unseen` | 783 | 783 | 2349 | 2349 |

说明：Standard `val_unseen` 的原始 annotation 是 path 级 783 条，SAME 输出为 instruction 展开后的 2349 条。该差异符合 R2R 常规三指令展开，不影响五类 NavNuances split 的 A1 确认。

#### 2. 检查 SAME `task.*` 配置是否支持 NavNuances split

`configs/same/navnuances_r2r_eval_only.yaml` 已将 NavNuances 作为 `R2R` eval split 接入：

| 配置项 | 当前值 | 结论 |
|---|---|---|
| `experiment.eval_only` | `true` | 只跑评估，不进入训练 |
| `experiment.decision_trace` | `true` | 输出逐步决策轨迹 |
| `experiment.moe_trace` | `true` | 输出 MoE router 诊断字段 |
| `experiment.data_dir` | `../../../data/navnuances/same` | 指向 NavNuances 的 SAME 编码目录 |
| `task.eval_splits.R2R` | `DC, LR, RR, VM, NU, val_unseen` | 覆盖五类 skill 和 Standard split |

因此，A.1 中“检查 SAME 的 `task.*` 配置是否支持 NavNuances split”的要求已经满足。

#### 3. 确认 SAME 输出中 trajectory/action sequence 的可解析性

实验 `0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2` 已成功生成 SAME 原始结果、submission、`eval_items.v3` 和 NavNuances evaluator 结果。

| 输出 | 路径 | 用途 |
|---|---|---|
| SAME 原始结果 | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/results/R2R_*_results.json` | 导出 NavNuances submission |
| NavNuances submission | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/navnuances_submission/submit_*.json` | 供 NavNuances evaluator 评分 |
| eval item sidecar | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/eval_items/R2R_*_eval_items.jsonl` | 逐样本反查 annotation、trajectory 和诊断字段 |
| NavNuances 评分 | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/navnuances_eval/results.json` | A1/A1 后续 per-skill 指标事实源 |

`eval_items.v3` 中每条样本的 `prediction` 至少包含：

| 字段 | 含义 | 对后续 Atlas 的意义 |
|---|---|---|
| `trajectory` | flatten 后的 viewpoint 序列 | 支持路径长度、loop/revisit、final endpoint 分析 |
| `pred_path_segments` | SAME agent 输出的分段路径 | 支持高层动作步数和 route 片段分析 |
| `official_saved_trajectory` | 与官方结果文件一致的 trajectory 格式 | 支持复现 submission 与 evaluator 对齐 |
| `decision_trace` | 每步 stop 概率、候选、选择、route、MoE 等信息 | 支持 early stop、candidate margin、global/local 选择偏向分析 |

当前没有保存 MatterSim 低层原子 action 序列。对 A1 来说，SAME 的高层 action/trajectory 已经可解析；如果后续需要严格的 heading/elevation 原子动作回放，需要另起 patch 或渲染链路扩展。

#### 4. 停止条件检查

| 停止条件 | 当前事实 | 结论 |
|---|---|---|
| SAME 完全不支持 NavNuances 格式，路线 A 需重新评估 | SAME 已通过 R2R eval split 跑通五类 NavNuances，并完成 evaluator 评分 | 未触发 |

### 代码实现原理

本 stage 没有新增代码，确认的是既有工程链路：

1. `scripts/setup/prepare_navnuances_same_r2r.py` 读取 `data/navnuances/annotations/NavNuances/R2R_*.json`，补充 SAME 需要的 `instr_encodings`，输出 `data/navnuances/same/R2R/R2R_*_enc.json`。
2. `configs/same/navnuances_r2r_eval_only.yaml` 将 `data/navnuances/same` 作为 SAME 的 R2R 数据目录，并把 `DC/LR/RR/VM/NU` 注册为 eval split。
3. `scripts/experiments/run_same.py` 应用 SAME base patches 后启动 `third_party/SAME/src/run.py`，产出 `results/` 与 `eval_items/`。
4. `scripts/eval/export_same_navnuances_submissions.py` 将 SAME `R2R_*_results.json` 转换为 NavNuances 需要的 `submit_*.json`。
5. `scripts/eval/run_same_navnuances_eval.py` 调用 `third_party/navnuances/evaluation/eval.py`，生成 `navnuances_eval/results.json`。

关键注意点：NavNuances 与 R2R 风格相似，但评估语义不同。五类 skill 的正式分数应以后处理的 `navnuances_eval/results.json` 为准，不应使用 SAME 在伪 R2R split 上给出的 `metrics.json` 作为最终结论。

### 执行脚本实例

完整链路对应命令如下：

```bash
python scripts/setup/prepare_navnuances_same_r2r.py
```

```bash
bash scripts/setup/run_container.sh
conda activate test-v1
python scripts/experiments/run_same.py \
  --config configs/same/navnuances_r2r_eval_only.yaml \
  --tag navnuances
```

```bash
bash scripts/setup/run_navnuances_container.sh
conda activate navnuances-eval
python scripts/eval/run_same_navnuances_eval.py \
  --experiment-id 0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2
```

本次文档生成前的只读核验命令：

```bash
python3 -m unittest tests.test_build_same_fine_metrics
```

结果：4 个测试通过。

### 结果分析与说明

#### 运行状态

| 字段 | 值 |
|---|---|
| experiment_id | `0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2` |
| status | `success` |
| exit_code | `0` |
| config | `configs/same/navnuances_r2r_eval_only.yaml` |
| checkpoint | `../../../data/same/ckpt/SAME.pt` |
| started_at | `2026-04-28T16:35:57.360693+08:00` |
| finished_at | `2026-04-28T16:43:09.278490+08:00` |
| duration_seconds | `431.917797` |

#### NavNuances evaluator 指标

这些指标是 SAME 在 NavNuances 上的正式评分事实源。

| Skill | 指标 | 数值 | 样本数 |
|---|---|---:|---:|
| Standard | SR | 76.29 | 2349 |
| Standard | SPL | 66.24 | 2349 |
| DC | SR | 63.56 | 579 |
| LR | SR | 31.39 | 685 |
| RR | SR | 89.82 | 275 |
| VM | SR | 85.88 | 170 |
| VM | SPL | 79.74 | 170 |
| NU | path_SR | 33.33 | 78 |
| NU | nDTW | 26.83 | 78 |

#### SAME `metrics.json` 的定位

SAME 自身对伪 R2R split 的 `metrics.json` 只能作为输出覆盖与 pipeline sanity check，不作为 NavNuances skill 能力结论。原因是 NavNuances 的任务样本和成功定义与标准 R2R 不完全对应。

| Split | SAME SR | SAME SPL | 备注 |
|---|---:|---:|---|
| DC | 0.00 | 0.00 | 不作为 NavNuances 正式指标 |
| LR | 1.90 | 0.00 | 不作为 NavNuances 正式指标 |
| RR | 1.09 | 0.00 | 不作为 NavNuances 正式指标 |
| VM | 85.88 | 79.74 | 与 NavNuances VM 口径较接近，但仍以后者为准 |
| NU | 0.00 | 0.00 | 不作为 NavNuances 正式指标 |
| Standard `val_unseen` | 76.29 | 66.24 | 标准 R2R sanity check |

## 结果复盘

A.1 可以标记为完成。SAME 在 NavNuances 上的评估接口已经具备完整闭环：数据适配、SAME eval 配置、模型推理、submission 导出、NavNuances evaluator 评分和逐样本 `eval_items.v3` 诊断材料都已经存在。

本轮也暴露出一个边界：`eval_items.v3` 保存的是高层 trajectory、分段路径和决策轨迹，而不是低层 MatterSim 原子 action 序列。对 Ability Atlas 的 A1、D1、D3 已经足够；若未来要做精确动作级 replay，需要单独扩展。

对上游 `L.01.01` 的反馈：

| 项 | 建议状态 | 理由 |
|---|---|---|
| A.1：确认 SAME 在 NavNuances 数据集上的评估接口 | 完成 | 已有成功实验与完整产物 |
| 阶段 B.1：运行 SAME val_unseen 评估，收集原始 trajectory 数据 | 基本完成，可在后续 local trace 中前移或合并记录 | `0016` 已收集五类 split 与 Standard 的 trajectory/decision_trace |
| 阶段 B.3：计算 per-skill 指标 | 部分完成 | `navnuances_eval/results.json` 已有 split-level per-skill 指标，但还缺 per-episode success/failure 对齐表 |

下一步应进入 A.2 或 A.3：若优先保留 SAME-only 路线，应先建立 `scripts/analysis/ability_atlas/`，把 `navnuances_eval/results.json` 与 `eval_items.v3` 合并为 per-skill/per-episode 的 Atlas 输入表。
