# S.01.02.01.A1-NavGPT-2评估接口确认

## 目标 / 任务

确认 `research/VLN research/local_traces/L.01.02.Ability Atlas 构建.md` 中阶段 A 的 NavGPT-2 评估接口是否可用于后续 Ability Atlas 构建。

说明：上游行动计划中该项编号为 **A.2：确认 NavGPT-2 在 NavNuances 上的评估接口**；本文件名中的 `A1` 视作 `L.01.02` 下第一个 NavGPT-2 接口确认 stage。

| 检查项 | 成功标准 | 本轮结论 |
|---|---|---|
| 是否有现成 NavGPT-2 eval 脚本或已运行结果 | 能定位官方/本地 eval 入口，并至少跑通标准 R2R split | 通过 |
| NavGPT-2 输入格式与 NavNuances 的兼容性 | NavNuances 样本能映射到 NavGPT-2 所需的 R2R 字段，且改造成本可控 | 部分通过 |
| NavGPT-2 输出可解析性 | 至少能解析 per-episode trajectory；若要细粒度诊断，需确认详细输出能力 | 基础通过，细粒度诊断需补充参数或 patch |
| A.2 停止条件 | 若 NavGPT-2 无法适配 NavNuances，改用 SAME-only Atlas | 未触发 |

本 stage 只确认 NavGPT-2 评估接口和当前推理结果，不完成 NavGPT-2 在 NavNuances 五类 skill 上的正式评估，也不生成 SAME vs NavGPT-2 的 skill-level 互补矩阵。

## 背景与输入

### 上游行动计划

| 来源 | 对本 stage 的要求 |
|---|---|
| `research/VLN research/local_traces/L.01.02.Ability Atlas 构建.md` | A.2 要确认 NavGPT-2 是否有现成 eval 脚本或已运行结果；若无，需要检查 NavGPT-2 输入格式与 NavNuances 的兼容性 |
| `tmp/nav-gpt2推理结果.md` | 记录了一次 `BATCH_SIZE=2` 的 OOM 失败和一次 `BATCH_SIZE=1` 的成功 R2R eval |
| `third_party/NavGPT-2` | NavGPT-2 本地项目目录，本 stage 只读核验 README、eval 脚本和 R2R 评估代码 |

### 已有工程入口

| 功能 | 路径 | 当前事实 |
|---|---|---|
| 官方运行说明 | `third_party/NavGPT-2/README.md` | README 给出 `cd map_nav_src && bash scripts/val_r2r_xl.sh` 的 R2R eval 入口 |
| XL eval 脚本 | `third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh` | 支持通过环境变量覆盖 `DATA_ROOT`、`OUTPUT_DIR`、`QFORMER_CKPT`、`RESUME_FILE`、`BATCH_SIZE`、`EXTRA_ARGS` |
| 参数与路径派生 | `third_party/NavGPT-2/map_nav_src/r2r/parser.py` | `--dataset` 支持 `r2r/r4r/rxr-en/REVERIE`，R2R annotation、feature、connectivity、candidate 路径由 `root_dir` 派生 |
| eval split 构造 | `third_party/NavGPT-2/map_nav_src/r2r/main_nav.py` | test 模式固定评估 `val_train_seen`、`val_seen`、`val_unseen` |
| annotation 读取 | `third_party/NavGPT-2/map_nav_src/r2r/data_utils.py` | 对官方 split 读取 `<dataset>_<split>_enc.json`，并将多条 instruction 展开为 `instr_id` |
| R2R 环境与评分 | `third_party/NavGPT-2/map_nav_src/r2r/env.py` | 使用 MatterSim、EVA-CLIP feature、connectivity、candidate 文件，输出 SR/SPL/nDTW/SDTW/CLS 等指标 |
| 结果导出 | `third_party/NavGPT-2/map_nav_src/r2r/agent_base.py` | 默认输出 `instr_id` 与 `trajectory`；`--detailed_output` 时可额外输出 `details` 和 `thoughts` |

### 本轮已有结果

| 项 | 值 |
|---|---|
| 结果记录 | `tmp/nav-gpt2推理结果.md` |
| 输出目录 | `experiment_outputs/navgpt2_r2r_xl_20260504_163943` |
| 模型入口 | `third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh` |
| 模型配置 | `arch=blip2_t5_instruct_nav`，`model_type=flant5xl`，`fusion=global` |
| Q-Former checkpoint | `/workspace/vln-lab/data/navgpt2/map_nav_src/models/lavis/output/NavGPT-InstructBLIP-FlanT5XL.pth` |
| Policy checkpoint | `/workspace/vln-lab/data/navgpt2/datasets/R2R/trained_models/best_val_unseen_xl` |
| 最终可运行 batch size | `BATCH_SIZE=1` |
| GPU 约束 | 失败日志显示 GPU 总显存约 7.50 GiB，`BATCH_SIZE=2` 会在 val_seen 阶段 OOM |

## 执行记录

### 对行动计划各项的逐条回应

#### 1. 检查是否有现成 NavGPT-2 eval 脚本或已运行结果

NavGPT-2 已有标准 R2R eval 脚本，并且本轮已经在 XL 模型上跑通 `val_seen` 与 `val_unseen`。

| 项 | 事实 | 结论 |
|---|---|---|
| 现成 eval 脚本 | `third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh` | 存在 |
| 脚本前置检查 | 检查 R2R annotation、candidate、connectivity、EVA-CLIP feature、policy checkpoint、Q-Former checkpoint | 入口具备工程可复现性 |
| test 模式 | 脚本调用 `r2r/main_nav.py ... --test` | 只评估，不训练 |
| 已运行结果 | `experiment_outputs/navgpt2_r2r_xl_20260504_163943/preds/submit_val_seen.json` 与 `submit_val_unseen.json` | 已生成 |
| 官方指标 sanity check | README 中 XL `R2R unseen` SR/SPL 为 69.89/58.86；本轮 val_unseen SR/SPL 为 69.65/58.71 | 与官方量级一致 |

需要注意：第一次 `BATCH_SIZE=2` 运行已完成 `val_train_seen` 预测后在 `val_seen` 阶段 OOM。第二次沿用同一个 `OUTPUT_DIR` 并设置 `BATCH_SIZE=1`，由于 `valid()` 会跳过已存在的 pred 文件，最终 `submit_val_train_seen.json` 来自第一次运行，`submit_val_seen.json` 与 `submit_val_unseen.json` 来自第二次运行。

#### 2. 检查 NavGPT-2 输入格式与 NavNuances 的兼容性

NavGPT-2 当前没有直接暴露 NavNuances split 配置，但它使用的 R2R annotation 字段与 `data/navnuances/same/R2R/R2R_*_enc.json` 的核心字段基本兼容。

| NavGPT-2 需要的字段/资源 | 代码位置 | NavNuances 当前状态 | 结论 |
|---|---|---|---|
| `scan` | `R2RNavBatch._get_obs()` | `data/navnuances/same/R2R/R2R_DC_enc.json` 等文件已有 | 兼容 |
| `path_id` | `construct_instrs()` 用于生成 `instr_id` | NavNuances 编码文件已有，且可为字符串 | 兼容 |
| `path` | `R2RNavBatch.reset()` 与 evaluator 使用 | NavNuances 编码文件已有 | 兼容 |
| `heading` | `R2RNavBatch.reset()` 使用 | NavNuances 编码文件已有 | 兼容 |
| `instructions` | `construct_instrs()` 展开 | NavNuances 编码文件已有 | 兼容 |
| `instr_encodings` | annotation 文件中存在，当前 NavGPT-2 代码主要使用原始 instruction 文本 | NavNuances 编码文件已有 | 兼容 |
| connectivity | `parser.py` 固定为 `<root_dir>/R2R/connectivity` | NavNuances 与 R2R/Matterport3D graph 共享基础资源，但需在 NavGPT-2 data root 下可见 | 需整理路径 |
| candidate 文件 | `parser.py` 固定为 `<root_dir>/R2R/annotations/scanvp_candidates.json` | 当前 NavNuances 编码目录没有 candidate 文件 | 需复用 NavGPT-2/R2R candidate |
| EVA-CLIP feature | `parser.py` 固定为 `<root_dir>/R2R/features/MP3D_eva_clip_g_can.lmdb` | 当前 host 侧 `data/navgpt2` 不完整；container 运行时路径可用 | 需保持 container 数据挂载 |

当前直接阻塞不是样本字段，而是 split 与路径约定：

| 约束 | 当前代码事实 | 对 NavNuances 的影响 |
|---|---|---|
| split 固定 | `main_nav.py` 中 `val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']` | 不会自动读取 `DC/LR/RR/VM/NU` |
| annotation 路径固定 | `data_utils.py` 读取 `R2R_<split>_enc.json` | 需要把 NavNuances 五类 split 放到 NavGPT-2 的 R2R annotation 目录，或给代码增加 `--eval_splits` |
| dataset choices 固定 | `parser.py` 的 `--dataset` choices 不包含 `navnuances` | 更适合继续伪装成 `--dataset r2r`，而不是新增 dataset |

因此，NavGPT-2 **不能无修改直接跑 NavNuances 五类 skill split**，但可以通过小改适配：

| 方案 | 思路 | 风险 |
|---|---|---|
| 最小 patch | 给 `main_nav.py`/`parser.py` 增加 `--eval_splits DC,LR,RR,VM,NU`，并让 `root_dir` 指向同时包含 annotation、features、connectivity、candidate 的 NavGPT-2 data root | 低，改动集中 |
| 文件 symlink | 将 `R2R_DC_enc.json` 等 NavNuances 文件放入 NavGPT-2 的 `R2R/annotations`，再临时替换 `val_env_names` | 中，容易污染数据目录 |
| SAME-only fallback | 若之后适配失败，只使用 SAME 结果构建 Atlas | 当前无需触发 |

#### 3. 确认 NavGPT-2 输出中 trajectory/action sequence 的可解析性

当前生成的 pred 文件可以解析 per-episode trajectory，但默认不包含逐步 logits、完整 action distribution 或低层 MatterSim 原子动作。

| 输出文件 | 样本数 | 字段 | 对 Ability Atlas 的用途 |
|---|---:|---|---|
| `experiment_outputs/navgpt2_r2r_xl_20260504_163943/preds/submit_val_train_seen.json` | 150 | `instr_id`, `trajectory` | sanity check；来自第一次 `BATCH_SIZE=2` 运行 |
| `experiment_outputs/navgpt2_r2r_xl_20260504_163943/preds/submit_val_seen.json` | 1021 | `instr_id`, `trajectory` | 标准 R2R seen 对比 |
| `experiment_outputs/navgpt2_r2r_xl_20260504_163943/preds/submit_val_unseen.json` | 2349 | `instr_id`, `trajectory` | 标准 R2R unseen 对比 |

当前 `trajectory` 是分段 path，形如：

```json
{
  "instr_id": "4837_2",
  "trajectory": [
    ["471a6f3beedb4cc7a71edc7fc1c5275b"],
    ["cc96c884cf7c4218a00005a03143b889"],
    ["31ef98acc7f44716abedd5a0e0747b71"]
  ]
}
```

| 可解析项 | 当前是否支持 | 说明 |
|---|---|---|
| final endpoint | 支持 | `trajectory` 最后一段最后一个 viewpoint 即预测终点 |
| high-level navigation steps | 支持 | `len(trajectory) - 1` 可近似 action steps |
| flatten path | 支持 | `sum(trajectory, [])` 可得到 evaluator 使用的 viewpoint 序列 |
| loop/revisit | 支持 | flatten 后统计重复 viewpoint |
| path length / nav error | 支持 | 需要连接 connectivity shortest distance |
| stop probability | 需重跑 | 使用 `--detailed_output` 可输出 visited node 的 `stop_prob` |
| thought text | 需重跑 | 需要同时使用 `--detailed_output --output_thought` |
| per-candidate action logits | 当前不支持 | 需要 patch `agent.py` 保存 `nav_logits/nav_probs` |
| 低层 MatterSim 原子动作 | 当前不支持 | 当前输出是 viewpoint-level path，不是 heading/elevation action replay |

对 A.2 来说，NavGPT-2 的 trajectory 输出已经足够支持 SR/SPL、endpoint、loop/revisit 等基础对齐。若后续要把 NavGPT-2 纳入 D3 的 evidence-level failure attribution，需要额外打开详细输出或新增 sidecar。

#### 4. 停止条件检查

| 停止条件 | 当前事实 | 结论 |
|---|---|---|
| NavGPT-2 无法适配 NavNuances，改用 SAME-only Atlas | NavGPT-2 已跑通标准 R2R；NavNuances 编码文件具备 R2R 核心字段；主要缺口是 eval split/path 配置 | 未触发 |

### 代码实现原理

本 stage 没有新增代码，只确认既有 NavGPT-2 评估链路。

1. `scripts/val_r2r_xl.sh` 进入 `map_nav_src`，根据环境变量组装 `DATA_ROOT`、`OUTPUT_DIR`、`QFORMER_CKPT` 和 `RESUME_FILE`。
2. 脚本在启动前检查 R2R annotation、candidate、connectivity、EVA-CLIP feature、policy checkpoint 和 Q-Former checkpoint 是否存在。
3. `r2r/parser.py` 从 `root_dir` 派生 `img_ft_file`、`connectivity_dir`、`anno_dir`、`candidate_file_dir`、`pred_dir` 等路径。
4. `r2r/main_nav.py` 在 `--test` 模式下调用 `valid()`，固定构建 `val_train_seen`、`val_seen`、`val_unseen` 三个 eval env。
5. `R2RNavBatch` 读取 annotation、connectivity graph、candidate 文件和 image feature，执行 MatterSim viewpoint-level navigation。
6. `GMapNavAgent.rollout()` 每步先经过 NavGPT thought/panorama/action 分支，再选择 stop 或下一个 viewpoint，并把路径写入 `traj['path']`。
7. `valid()` 调用 `env.eval_metrics(preds)` 计算指标，并将 predictions 保存到 `preds/submit_<split>.json`。

### 执行脚本实例

第一次运行使用 `BATCH_SIZE=2`，在 `val_train_seen` 完成后 OOM：

```bash
OUTPUT_DIR="/workspace/vln-lab/experiment_outputs/${RUN_ID}" \
BATCH_SIZE=2 \
CUDA_VISIBLE_DEVICES=0 \
bash third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh
```

失败位置与显存事实：

| 项 | 值 |
|---|---|
| 已完成 split | `val_train_seen`，150 predictions |
| 失败 split | `val_seen` |
| 异常类型 | `torch.OutOfMemoryError` |
| OOM 位置 | T5 encoder self-attention softmax |
| 申请显存 | 156.00 MiB |
| GPU 总显存 | 7.50 GiB |
| OOM 时剩余显存 | 150.50 MiB |
| PyTorch allocated | 6.13 GiB |

第二次运行降到 `BATCH_SIZE=1`，沿用同一输出目录并成功完成 `val_seen` 与 `val_unseen`：

```bash
OUTPUT_DIR="/workspace/vln-lab/experiment_outputs/${RUN_ID}" \
BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh
```

后续若要收集更细粒度诊断输出，可通过现有脚本的 `EXTRA_ARGS` 追加参数：

```bash
OUTPUT_DIR="/workspace/vln-lab/experiment_outputs/${RUN_ID}" \
BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
EXTRA_ARGS="--detailed_output --output_thought" \
bash third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh
```

该命令会增加输出体积和可能的运行开销；在 8GB 显存环境中建议先对小 split 试跑。

### 结果分析与说明

#### 运行状态

| 字段 | 值 |
|---|---|
| output_dir | `experiment_outputs/navgpt2_r2r_xl_20260504_163943` |
| final batch_size | `1` |
| checkpoint iter | `185000` |
| total parameters | `1473468964` |
| trainable parameters | `62724132` |
| `max_action_steps` | `15` |
| `max_instr_len` | `200` |
| `num_beams` | `5` |
| `detailed_output` | `false` |
| `output_thought` | `false` |

#### 标准 R2R 指标

本轮 `BATCH_SIZE=1` 成功运行后，`logs/valid.txt` 中记录的指标如下：

| Split | 样本数 | action_steps | steps | lengths | nav_error | oracle_error | SR | oracle_SR | SPL | nDTW | SDTW | CLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `val_seen` | 1021 | 5.83 | 6.19 | 12.48 | 2.98 | 1.77 | 72.97 | 79.92 | 65.15 | 72.40 | 62.68 | 70.72 |
| `val_unseen` | 2349 | 6.12 | 6.71 | 12.80 | 3.35 | 1.83 | 69.65 | 78.42 | 58.71 | 66.51 | 56.50 | 65.00 |

与 NavGPT-2 README 中 XL 官方 `R2R unseen` SR/SPL 69.89/58.86 对比，本轮 val_unseen SR/SPL 差异为 -0.24pp/-0.15pp，说明当前评估接口、checkpoint 和数据路径基本可信。

#### 对 Ability Atlas 的意义

| 问题 | 当前答案 | 对后续步骤的影响 |
|---|---|---|
| 能否获得 NavGPT-2 标准 R2R baseline | 可以 | 可作为 sanity baseline 和论文指标参照 |
| 能否直接得到 NavGPT-2 五类 NavNuances skill 指标 | 还不能 | 需要先做 `--eval_splits` 或等价路径适配 |
| 能否对齐 episode-level trajectory | 可以 | 后续适配 NavNuances 后，可与 SAME 一样按 `instr_id`/`path_id` 对齐 |
| 能否直接做 evidence-level failure attribution | 当前不足 | 需要 `--detailed_output`、`--output_thought` 或新增 logits sidecar |
| 是否需要退回 SAME-only Atlas | 暂不需要 | NavGPT-2 适配风险可控，建议先做小 patch |

## 结果复盘

本轮可以把 `L.01.02` 阶段 A 中的 NavGPT-2 **评估接口确认** 标记为完成，但不能把 NavGPT-2 **NavNuances 五类 skill 评估** 标记为完成。

核心结论：

| 项 | 建议状态 | 理由 |
|---|---|---|
| A.2：确认 NavGPT-2 在 NavNuances 上的评估接口 | 接口确认完成 | 标准 R2R eval 已跑通，输出 schema 可解析，NavNuances 样本字段与 R2R 输入基本兼容 |
| A.2 停止条件 | 未触发 | 当前不是无法适配，而是需要补一个 eval split/path 适配层 |
| C.1：若 NavGPT 结果已存在，直接导入并对齐 episode ID | 暂不可直接推进 | 现有结果是标准 R2R，不是 NavNuances 五类 split |
| C.2：若 NavGPT 结果不存在，尝试快速评估 | 建议作为下一步 | 在 NavGPT-2 中新增 `--eval_splits DC,LR,RR,VM,NU` 后重跑五类 split |

对上游 Ability Atlas 的反馈：

| 发现 | 影响 |
|---|---|
| `BATCH_SIZE=2` 在 7.50 GiB GPU 上会 OOM | 后续 NavGPT-2 eval 默认使用 `BATCH_SIZE=1` |
| 标准 `val_unseen` 2349 条耗时约 2200s | 五类 NavNuances 共 1787 条，单卡 batch 1 预计可在同一量级内完成 |
| 默认 pred 只有 `instr_id` 与 `trajectory` | A2 的 SR/SPL 互补分析足够；D3 的细粒度 evidence 分析需补详细输出 |
| NavGPT-2 split 名称硬编码 | 下一步最小工程任务是增加可配置 eval splits，而不是改模型主体 |

下一步建议进入一个很小的工程 stage：给 NavGPT-2 eval 增加 `--eval_splits` 参数，并把 NavNuances 五类 `R2R_*_enc.json`、R2R feature/connectivity/candidate 组织到同一个 `root_dir` 下，先用 `DC` 单 split 验证输出和 NavNuances evaluator 对齐，再扩展到 `DC/LR/RR/VM/NU`。
