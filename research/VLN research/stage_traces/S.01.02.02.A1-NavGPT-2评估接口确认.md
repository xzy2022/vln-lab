# S.01.02.02.A1-NavGPT-2评估接口确认

## 目标 / 任务

确认 NavGPT-2 是否已经从“标准 R2R eval 可运行”推进到“NavNuances 五类 skill split 可运行、可评分、可用于 Ability Atlas 的 NavGPT-2 baseline”。

说明：上游行动计划中该项对应 `L.01.02.Ability Atlas 构建.md` 的 **A.2：确认 NavGPT-2 在 NavNuances 上的评估接口**，并承接 `S.01.02.01.A1-NavGPT-2评估接口确认.md` 中提出的下一步：为 NavGPT-2 增加 `--eval_splits DC LR RR VM NU` 后重跑五类 split。

| 检查项 | 成功标准 | 本轮结论 |
|---|---|---|
| NavGPT-2 能否读取 NavNuances 五类 split | `DC/LR/RR/VM/NU` 能被构建为 eval env，并生成 pred 文件 | 通过 |
| NavGPT-2 prediction 能否转为 NavNuances submission | 五类 `preds/submit_*.json` 可复制到 evaluator 所需目录，字段合法 | 通过 |
| NavNuances 官方 evaluator 能否完成评分 | 生成 `navnuances_eval/results.json`，且五类指标都有样本数与主指标 | 通过 |
| 输出是否支持 Ability Atlas 的 A2 对比 | 能得到 skill-level NavGPT-2 指标，可与 SAME 指标对齐 | 通过 |
| 输出是否支持 D3 细粒度 evidence attribution | 默认只有 `instr_id/trajectory`，缺少 stop prob、thought、candidate logits | 部分通过 |
| A.2 停止条件 | 若 NavGPT-2 无法适配 NavNuances，改用 SAME-only Atlas | 未触发 |

本 stage 完成 NavGPT-2 在 NavNuances 五类 skill 上的首次可复现推理与评分。它可以作为 Ability Atlas 中 NavGPT-2 侧的 skill-level baseline，但还不等价于 SAME vs NavGPT-2 的完整互补矩阵。

## 背景与输入

### 上游约束

| 来源 | 对本 stage 的要求 |
|---|---|
| `research/VLN research/local_traces/L.01.02.Ability Atlas 构建.md` | A.2 需要确认 NavGPT-2 在 NavNuances 上的评估接口；C.2 在无现成结果时尝试快速评估 |
| `research/VLN research/stage_traces/S.01.02.01.A1-NavGPT-2评估接口确认.md` | 标准 R2R XL eval 已跑通；NavNuances 适配缺口是 split/path 配置；建议新增 `--eval_splits` |
| `research/VLN research/研究记录规范.md` | stage trace 需要记录执行过程、结果分析与复盘，事实类信息优先用表格 |
| `tmp/nav-gpt2-nav推理.md` | 记录本轮 NavGPT-2 NavNuances 评分命令和五类 skill 的 evaluator 输出 |

### 已实现工程入口

| 功能 | 路径 | 当前事实 |
|---|---|---|
| NavNuances -> SAME/R2R encoded annotation | `scripts/setup/prepare_navnuances_same_r2r.py` | 已生成 `data/navnuances/same/R2R/R2R_{DC,LR,RR,VM,NU}_enc.json` |
| NavNuances annotation 安装到 NavGPT-2 数据根 | `scripts/setup/prepare_navnuances_navgpt2_r2r.py` | 将五类 `R2R_*_enc.json` 安装到 NavGPT-2 的 `R2R/annotations/` |
| NavGPT-2 split 参数化 patch | `patches/navgpt2/base/0002-eval-splits.patch` | 给 `parser.py` 增加 `--eval_splits`，给 `main_nav.py` 增加 split 解析 |
| XL eval 脚本 | `third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh` | 支持 `OUTPUT_DIR`、`BATCH_SIZE`、`CUDA_VISIBLE_DEVICES`、`EXTRA_ARGS` 等环境变量 |
| NavNuances evaluator 后处理 | `scripts/eval/run_navgpt2_navnuances_eval.py` | 检查并复制 `preds/submit_{DC,LR,RR,VM,NU}.json`，调用官方 evaluator |

### 本轮运行配置

| 项 | 值 |
|---|---|
| run_id | `navgpt2_navnuances_xl_20260504_213915` |
| 输出目录 | `experiment_outputs/navgpt2_navnuances_xl_20260504_213915` |
| 模型入口 | `third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh` |
| 评估 split | `DC LR RR VM NU` |
| batch size | `1` |
| 模型配置 | `arch=blip2_t5_instruct_nav`，`model_type=flant5xl`，`fusion=global` |
| Policy checkpoint | `/workspace/vln-lab/data/navgpt2/datasets/R2R/trained_models/best_val_unseen_xl` |
| Q-Former checkpoint | `/workspace/vln-lab/data/navgpt2/map_nav_src/models/lavis/output/NavGPT-InstructBLIP-FlanT5XL.pth` |
| `detailed_output` | `false` |
| `output_thought` | `false` |
| NavNuances evaluator 输出 | `experiment_outputs/navgpt2_navnuances_xl_20260504_213915/navnuances_eval/results.json` |

## 执行记录

### 对行动计划各项的逐条回应

#### 1. 补齐 NavGPT-2 的 NavNuances split 适配

`S.01.02.01` 的结论是：NavGPT-2 标准 R2R eval 可用，但 `main_nav.py` 固定评估 `val_train_seen/val_seen/val_unseen`，无法直接读取 `DC/LR/RR/VM/NU`。

本轮通过 experimental patch 补齐该缺口：

| 改动 | 作用 | 结论 |
|---|---|---|
| `parser.py` 增加 `--eval_splits` | 允许从命令行传入一个或多个 split 名称 | 可用 |
| `main_nav.py` 增加 `parse_eval_splits()` | 支持空格分隔或逗号分隔 split 列表 | 可用 |
| `build_dataset()` 优先使用 `args.eval_splits` | 覆盖默认 R2R validation split 列表 | 可用 |
| `EXTRA_ARGS="--eval_splits DC LR RR VM NU"` | 在不改 eval shell 脚本主体的情况下传入 NavNuances split | 可用 |

这说明 NavGPT-2 不再需要 SAME-only fallback；后续 Ability Atlas 可以纳入 NavGPT-2 侧 skill baseline。

#### 2. 生成五类 NavGPT-2 prediction

本轮在 `experiment_outputs/navgpt2_navnuances_xl_20260504_213915/preds/` 下生成五个 prediction 文件。

| Split | Prediction 文件 | 样本数 | 字段 |
|---|---|---:|---|
| DC | `preds/submit_DC.json` | 579 | `instr_id`, `trajectory` |
| LR | `preds/submit_LR.json` | 685 | `instr_id`, `trajectory` |
| RR | `preds/submit_RR.json` | 275 | `instr_id`, `trajectory` |
| VM | `preds/submit_VM.json` | 170 | `instr_id`, `trajectory` |
| NU | `preds/submit_NU.json` | 78 | `instr_id`, `trajectory` |

样本数合计为 1787，与本轮 NavNuances 五类 skill evaluation 范围一致。

#### 3. 转换 submission 并调用 NavNuances evaluator

执行后处理脚本：

```bash
python scripts/eval/run_navgpt2_navnuances_eval.py \
  --experiment-dir experiment_outputs/navgpt2_navnuances_xl_20260504_213915
```

后处理脚本完成三件事：

| 步骤 | 输出 |
|---|---|
| 检查五类 pred 文件 | 确认每条 prediction 包含 `instr_id` 与 `trajectory` |
| 复制为 NavNuances submission | 写入 `navnuances_submission/submit_{DC,LR,RR,VM,NU}.json` |
| 调用 evaluator | 写入 `navnuances_eval/results.json` |

实际 evaluator 命令为：

```bash
/opt/conda/envs/navnuances-eval/bin/python \
  /workspace/vln-lab/third_party/navnuances/evaluation/eval.py \
  --skip-standard \
  --annotation_root /workspace/vln-lab/data/navnuances/annotations/NavNuances \
  --submission_root /workspace/vln-lab/experiment_outputs/navgpt2_navnuances_xl_20260504_213915/navnuances_submission \
  --out_root /workspace/vln-lab/experiment_outputs/navgpt2_navnuances_xl_20260504_213915/navnuances_eval \
  --connectivity_dir /workspace/vln-lab/data/same/simulator/connectivity
```

`--skip-standard` 表示本轮只评估 NavNuances 五类 skill，不混入标准 `val_unseen`。

### 代码实现原理

1. `prepare_navnuances_same_r2r.py` 将原始 NavNuances annotation 转为 NavGPT-2/R2R 可读的 encoded JSON。
2. `prepare_navnuances_navgpt2_r2r.py` 把 `R2R_{DC,LR,RR,VM,NU}_enc.json` 安装到 NavGPT-2 data root 的 `R2R/annotations/`。
3. NavGPT-2 仍以 `--dataset r2r` 运行，因此 image feature、connectivity、candidate file 继续复用 R2R/Matterport3D 资源。
4. `--eval_splits DC LR RR VM NU` 让 `build_dataset()` 构建五个 NavNuances eval env，而不是默认 R2R validation env。
5. NavGPT-2 `valid()` 输出 `preds/submit_<split>.json`，每条记录保留 `instr_id` 与 viewpoint-level `trajectory`。
6. `run_navgpt2_navnuances_eval.py` 将 prediction 文件转换为 NavNuances evaluator 的 submission 目录结构，并调用 `third_party/navnuances/evaluation/eval.py` 计算 skill-specific metrics。

### 执行脚本实例

NavGPT-2 推理在 NavGPT-2 环境中执行：

```bash
python scripts/setup/prepare_navnuances_same_r2r.py
python scripts/setup/prepare_navnuances_navgpt2_r2r.py

RUN_ID="navgpt2_navnuances_xl_20260504_213915"
OUTPUT_DIR="/workspace/vln-lab/experiment_outputs/${RUN_ID}" \
BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
EXTRA_ARGS="--eval_splits DC LR RR VM NU" \
bash third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh
```

NavNuances 评分在 NavNuances evaluator 环境中执行：

```bash
python scripts/eval/run_navgpt2_navnuances_eval.py \
  --experiment-dir experiment_outputs/navgpt2_navnuances_xl_20260504_213915
```

本次文档生成时补充跑了脚本单测：

```bash
python3 -m unittest tests/test_navgpt2_navnuances.py
```

结果：7 个测试通过。

### 结果分析与说明

#### NavNuances 五类 skill 指标

| Skill | NavNuances 名称 | 样本数 | 主指标 | 补充指标 |
|---|---|---:|---:|---|
| LR | Landmark Recognition | 685 | SR 33.58 | towards 37.39, past 29.52 |
| RR | Room Recognition | 275 | SR 77.45 | oracle 91.27, into 63.81, exit 85.88 |
| VM | Vertical Movement | 170 | SR 83.53 | oracle 87.06, SPL 78.60, nDTW 80.66 |
| DC | Direction Change | 579 | SR 62.52 | left 63.54, right 55.73, around 68.21, pair_SR 32.81 |
| NU | Numerical Directional Region | 78 | path_SR 25.64 | nDTW 26.93 |

注意：NU 的主指标是 `path_SR`，不是与 DC/LR/RR/VM 完全同义的 `sr`；后续做跨 skill 汇总时需要在表头中明确指标口径。

#### 对 Ability Atlas 的意义

| 问题 | 当前答案 | 对后续步骤的影响 |
|---|---|---|
| 能否获得 NavGPT-2 NavNuances skill baseline | 可以 | A2/C2 可视为完成，后续可进入 SAME vs NavGPT-2 对齐 |
| 五类 skill 是否呈现差异 | 是 | VM/RR 明显强，LR/NU 明显弱，DC 居中但 pair_SR 低 |
| 是否能直接生成 SAME vs NavGPT-2 互补矩阵 | 还不能 | 需要导入 SAME 对应五类指标并统一 metric schema |
| 是否能直接做 evidence-level failure attribution | 还不能 | 默认输出缺少 stop probability、thought、candidate logits |
| 是否需要 SAME-only fallback | 不需要 | NavGPT-2 已具备可运行 NavNuances 评估链路 |

从结果上看，NavGPT-2 在 skill 维度上不是均匀退化：`Vertical Movement` 与 `Room Recognition` 较强，`Landmark Recognition` 与 `Numerical Directional Region` 较弱，`Direction Change` 的总体 SR 尚可但成对约束 `pair_SR` 很低。这些模式足以作为 Ability Atlas 的 NavGPT-2 侧事实基础，但还不能单独判断失败来自 evidence 缺失还是 policy 不足。

#### 输出粒度限制

| 需求 | 当前输出是否满足 | 说明 |
|---|---|---|
| per-skill SR / path_SR / SPL / nDTW | 满足 | evaluator 已生成 `results.json` |
| episode-level final endpoint | 满足 | 可从 `trajectory` 最后一段最后一个 viewpoint 解析 |
| loop/revisit 粗诊断 | 满足 | 可 flatten trajectory 后统计重复 viewpoint |
| stop probability | 不满足 | 需要重跑 `--detailed_output` |
| thought text | 不满足 | 需要重跑 `--detailed_output --output_thought` |
| per-candidate action logits/probs | 不满足 | 需要 patch `agent.py` 输出 sidecar |

## 结果复盘

本轮达成了 `S.01.02.01` 之后的关键推进：NavGPT-2 已经可以在 NavNuances `DC/LR/RR/VM/NU` 五类 split 上完成推理、转换 submission 并通过官方 evaluator 评分。`L.01.02` 中 A.2 的停止条件未触发，Ability Atlas 不需要退回 SAME-only 版本。

对上游行动计划的状态更新：

| 上游项 | 建议状态 | 理由 |
|---|---|---|
| A.2：确认 NavGPT-2 在 NavNuances 上的评估接口 | 完成 | 五类 split 已跑通并生成 evaluator 指标 |
| C.2：若 NavGPT 结果不存在，尝试快速评估 | 完成 | 已获得 NavGPT-2 XL 的 NavNuances 五类 skill baseline |
| D.2：分析 SAME vs NavGPT skill 互补模式 | 可启动 | NavGPT-2 侧指标已就绪，下一步需要导入 SAME 侧指标 |
| D.3：trajectory evidence failure attribution | 只能部分启动 | 默认 trajectory 足够做 endpoint/loop 粗分析，但不够做 stop/thought/logits 诊断 |

下一步应优先把 SAME 的 NavNuances 五类 skill 指标与本轮 NavGPT-2 指标对齐，生成 Ability Atlas 的第一版 skill-level 对比表；若后续要分析 failure attribution，再针对小 split 重跑 `--detailed_output --output_thought` 或增加 logits sidecar。
