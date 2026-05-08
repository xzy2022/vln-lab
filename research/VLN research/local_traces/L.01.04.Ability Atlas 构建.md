# L.01.04.Ability Atlas 构建

## 目标 / 任务

在 `L.01.03.Ability Atlas 构建.md` 的基础上，将 Ability Atlas 从“计划 + 目标表格”更新为 **skill-level Atlas v0**。

本版本的核心任务不是重开路线，而是完成三件事：

1. 固化 SAME / NavGPT4v / NavGPT-2 XL 在 NavNuances 五类 skill 上的第一版对齐表。
2. 判断 H4（NavNuances 作为能力坐标系）是否已经得到初步支持。
3. 将后续行动重点从“接口确认”收紧到 “per-episode failure attribution 与 evidence gap 诊断”。

当前判断：**不需要大的路线变动**。主要表格已经可以填出，Ability Atlas v0 已经具备事实基础；尚未完成的是 evidence-level attribution，因此不能把 `Evidence Gap` 写成定论。

## 背景与输入

### 来自 G.01 的决策输出

路线 A 已由全局轨迹（`G.01.证据自适应导航状态验证.md`）明确选定：

**直接目的**：搜集信息（A），为验证 H1/H4 做准备。

**主要思路**：从诊断视角出发，先用 SAME + NavNuances 构建能力坐标地图，明确哪些 skill 失败是 evidence 缺失导致的、哪些是 policy 不足导致的。

**预期结果**：
- 理想时：skill × failure 的可解释矩阵直接指导 verifier 的 evidence schema 设计；贡献形式为 NavNuances benchmark 上的新诊断报告。
- 不理想时：说明 SAME 当前的失败模式不够结构化，无法映射到 individual skill × evidence。

### 本轮新增事实

| 来源 | 当前事实 | 对本 trace 的意义 |
|---|---|---|
| `S.01.01.01.A1-SAME评估接口确认.md` | SAME 已能在 NavNuances 五类 split 上运行、导出 submission、通过 evaluator 评分，并生成 `eval_items` sidecar | A.1 完成；SAME 侧 skill-level 指标可信 |
| `S.01.02.02.A1-NavGPT-2评估接口确认.md` | NavGPT-2 XL 已能读取 `DC/LR/RR/VM/NU`，生成五类 prediction，并通过 NavNuances evaluator 评分 | A.2 / C.2 完成；不需要 SAME-only fallback |
| `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/navnuances_eval/results.json` | 提供 SAME 的五类 NavNuances 正式指标 | Ability Atlas 的 SAME 侧事实源 |
| `experiment_outputs/navgpt2_navnuances_xl_20260504_213915/navnuances_eval/results.json` | 提供 NavGPT-2 XL 的五类 NavNuances 正式指标 | Ability Atlas 的本地 NavGPT-2 侧事实源 |
| NavNuances paper / G.01 记录 | 提供 `NavGPT4v (0-shot)` 五类指标 | 作为纯 MLLM / zero-shot reasoning 风格参照，不是本地复现结果 |

### 关键假设状态

| ID | 假设 | L.01.04 当前状态 | 说明 |
|---|---|---|---|
| H1 | 不同 instruction skill 需要不同 evidence budget | 待验证 | 当前只有 skill-level performance，尚未证明 skill-specific evidence 优于 uniform evidence |
| H3 | endpoint recovery 需要 final-vs-candidate evidence gap | 待验证 | 当前没有 final-vs-candidate verifier margin |
| H4 | NavNuances 应超出 benchmark 角色，成为 route 的能力坐标系 | 初步支持 | 五类 skill 呈现清晰差异，SAME vs NavGPT4v 有强互补模式 |
| H5 | 旧 endpoint negative result 可转化为论文级贡献 | 保留可能 | 若后续 evidence attribution 或 verifier 不稳定，仍可形成 diagnostic 贡献 |

## 验证设计

### 验证目标

本 local trace 聚焦于 **H4 的阶段性验证**：

> NavNuances 是否能作为 EANSV 路线中的能力坐标系，而不只是一个 benchmark 附属表？

L.01.04 的判别口径是：如果 SAME、NavGPT4v、NavGPT-2 XL 在五类 skill 上呈现稳定、可解释、非随机的能力差异，则 H4 获得初步支持；但 evidence 缺失归因仍需要后续 D.3 的逐样本分析。

### 验证思路

Ability Atlas v0 暂时分为两层比较：

| 比较层 | 对象 | 作用 | 注意事项 |
|---|---|---|---|
| 论文参照层 | SAME vs NavGPT4v paper | 观察训练型 VLN policy 与 zero-shot MLLM reasoning 的互补模式 | NavGPT4v 非本地复现，不用于工程可复现主线 |
| 本地复现层 | SAME vs NavGPT-2 XL ours | 观察 SAME 与 LLM+VLN policy 融合模型在同一 evaluator 下的 skill 差异 | NavGPT-2 XL 不是纯 LLM baseline，不能等同于 NavGPT4v |

指标口径：

| Skill | 使用指标 | 说明 |
|---|---|---|
| DC | SR | Direction Change |
| LR | SR | Landmark Recognition |
| RR | SR | Room Recognition |
| VM | SR | Vertical Movement |
| NU | path_SR | Numerical Directional Region 的主指标，不与 SR 完全同义 |
| Mix | 标准 R2R val_unseen SR | 仅作 sanity / aggregate 参照，不代表 NavNuances 五类均值 |

### 子验证状态

| 子验证 | 验证什么 | 当前判断 | 证据与限制 |
|---|---|---|---|
| A1 | SAME 在五类 skill 上的分布是否有差异 | 初步通过 | SAME 从 LR 31.39 到 RR 89.82，差异很大；但 95% CI / bootstrap 尚未计算 |
| A2 | SAME vs NavGPT 是否有互补模式 | 通过 | 对 NavGPT4v paper：DC/LR 明显强于 SAME，RR/VM 明显弱于 SAME，NU 略强于 SAME |
| A3 | SAME failure 是否能归因到 evidence 类型缺失 | 未完成 | 需要 per-episode success/failure + trajectory/decision_trace 分析 |
| A4 | skill × evidence 矩阵是否可解释 | 部分通过 | skill-level pattern 可解释，但 evidence-level 矩阵未填实 |
| A5 | 诊断报告是否有独立论文价值 | 部分通过 | 已有非平凡 skill pattern；还需要 failure attribution 或 verifier 对照增强贡献 |

## 结果汇总表

指标口径：DC/LR/RR/VM 使用 SR，NU 使用 path_SR，Mix 使用标准 R2R val_unseen SR。SAME 列来自本地 `0016` 实验；NavGPT-2 XL 列来自本地 `navgpt2_navnuances_xl_20260504_213915`；NavGPT4v 列采用 NavNuances paper / G.01 已记录结果。

| Skill | 样本数 | 指标 | SAME | NavGPT4v (0-shot, paper) | NavGPT-2 XL (ours) | SAME - NavGPT4v | SAME - NavGPT-2 XL | Pattern v0 | Evidence Gap |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| DC | 579 | SR | 63.56 | 92.68 | 62.52 | -29.12pp | +1.04pp | NavGPT4v 强；SAME≈NavGPT-2 XL | 待 D.3，可能涉及方向语义、turn pair、heading/topology |
| NU | 78 | path_SR | 33.33 | 39.13 | 25.64 | -5.80pp | +7.69pp | 整体弱；NavGPT4v 略优，SAME 次之 | 待 D.3，可能涉及 counting、ordinal progress、multi-step state tracking |
| LR | 685 | SR | 31.39 | 62.87 | 33.58 | -31.48pp | -2.19pp | NavGPT4v 强；本地 SAME/NavGPT-2 XL 都弱 | 待 D.3，可能涉及 landmark visual-semantic evidence |
| RR | 275 | SR | 89.82 | 56.25 | 77.45 | +33.57pp | +12.36pp | SAME 强；NavGPT-2 XL 次之 | 待 D.3，可能涉及 room transition / topo prior |
| VM | 170 | SR | 85.88 | 13.64 | 83.53 | +72.24pp | +2.35pp | SAME≈NavGPT-2 XL 强；NavGPT4v 很弱 | 待 D.3，可能涉及 elevation / vertical topology |
| Mix | 2349 | R2R SR | 76.29 | 41.30 | 69.65 | +34.99pp | +6.64pp | SAME 标准 R2R aggregate 更强 | 不能直接用于 skill-level evidence attribution |

NavGPT4v 官方 R2R aggregate 行同时报告 `R2R SR=41.30`、`R2R nDTW=54.78`、`R2R SPL=36.84`；上表的 `Mix` 只使用 R2R SR，避免把 nDTW 误写成 SR。

### 补充观察

| 观察 | 解释 |
|---|---|
| SAME vs NavGPT4v 的互补非常清楚 | DC/LR 是 NavGPT4v 强项，RR/VM 是 SAME 强项，NU 上 NavGPT4v 只略高于 SAME |
| SAME vs NavGPT-2 XL 的差异没有 NavGPT4v 那么大 | NavGPT-2 XL 已融合 VLN policy，因此更像 policy-fused baseline，而不是纯 LLM reasoning baseline |
| LR 对 SAME 与 NavGPT-2 XL 都是短板 | 如果要做 evidence-adaptive verifier，landmark visual-semantic evidence 是优先候选 |
| RR/VM 对 SAME 已经很强 | 这些 skill 可作为 verifier 安全性 sanity slice，不一定是第一优先 recovery 目标 |
| NU 样本数少且绝对性能低 | 有诊断价值，但方法贡献上需要谨慎，避免被小样本噪声主导 |

## 当前结论

### 对 H4 的判断

H4 获得 **初步支持**。

理由：

1. 五类 NavNuances skill 不是均匀分布，SAME 在 LR/NU 与 RR/VM 上呈现明显差异。
2. SAME vs NavGPT4v paper 呈现强互补：NavGPT4v 更擅长方向变化与地标识别，SAME 更擅长区域识别和垂直移动；NU 上 NavGPT4v 仅小幅领先。
3. NavGPT-2 XL 的本地结果说明 policy-fused 模型可以缩小部分 gap，但不能直接替代 Ability Atlas 的 skill 诊断。

### 对 EANSV 的意义

Ability Atlas v0 已经足够支撑下一步 evidence schema 的优先级设计，但还不足以证明“某个 skill 失败就是某类 evidence 缺失导致的”。

当前更稳妥的表述是：

| 结论层级 | 是否可以写 | 说明 |
|---|---|---|
| skill-level 能力差异 | 可以 | 已有 evaluator 指标 |
| SAME vs NavGPT4v 互补模式 | 可以 | paper 数据与本地 SAME 指标可对齐 |
| SAME vs NavGPT-2 XL 本地对比 | 可以 | 同一 evaluator 下已跑通 |
| evidence gap / failure attribution | 暂不可以写成结论 | 需要 D.3 的逐样本分析 |
| verifier evidence portfolio | 只能写成设计动机 | 还没有 verifier margin 或 recovery/harm 结果 |

## 行动计划

### 阶段 A：工程准备

- [x] A.1：确认 SAME 在 NavNuances 数据集上的评估接口
  - 依据：`S.01.01.01.A1-SAME评估接口确认.md`
  - 产物：`0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2`

- [x] A.2：确认 NavGPT-2 在 NavNuances 上的评估接口
  - 依据：`S.01.02.02.A1-NavGPT-2评估接口确认.md`
  - 产物：`navgpt2_navnuances_xl_20260504_213915`
  - 结论：SAME-only fallback 不触发

- [ ] A.3：建立 Ability Atlas 诊断脚本框架
  - 建议路径：`scripts/analysis/ability_atlas/`
  - 最小输入：SAME `eval_items/*.jsonl`、SAME `navnuances_eval/results.json`、NavGPT-2 `preds/submit_*.json`、NavGPT-2 `navnuances_eval/results.json`
  - 最小输出：统一的 per-skill summary table 与 per-episode alignment table

### 阶段 B：SAME Eval on NavNuances

- [x] B.1：运行 SAME val_unseen / NavNuances 评估，收集 trajectory 数据
  - 已有五类 skill 与 Standard `val_unseen` 输出。

- [x] B.2：编写 NavNuances episode -> SAME format 适配脚本
  - 已有 `scripts/setup/prepare_navnuances_same_r2r.py`。

- [x] B.3a：对齐 SAME 输出与 NavNuances evaluator，计算 split-level per-skill 指标
  - 已有 `navnuances_eval/results.json`。

- [ ] B.3b：生成 per-episode success/failure 对齐表
  - 目标字段：`instr_id`、`skill`、`success`、`final_vp`、`gt_path/end_vp`、`trajectory`、`nav_error` 或 evaluator 可反查字段。
  - 用途：支撑 D.3 failure attribution。

### 阶段 C：对NavGPT与NavGPT-2的结果获取

- [x] C.1：导入 NavGPT4v paper 数据点
  - 用途：作为 zero-shot MLLM reasoning 参照。
  - 限制：非本地复现，不作为工程主 baseline。

- [x] C.2：完成 NavGPT-2 XL 本地 NavNuances 快速评估
  - 依据：`S.01.02.02.A1-NavGPT-2评估接口确认.md`
  - 输出：五类 `preds/submit_*.json` 与 evaluator `results.json`。

### 阶段 D：诊断矩阵构建

- [ ] D.1：计算 SAME per-skill 置信区间
  - 建议：二项近似 CI 或 bootstrap。
  - 注意：不要把单次 run 写成 `SAME σ`；应改为 `SAME CI` 或 `SAME SE`。

- [x] D.2：构建 SAME vs NavGPT / NavGPT-2 的 skill-level 互补矩阵 v0
  - 本文件的结果汇总表即为 v0。

- [ ] D.3：基于 trajectory / decision_trace 做 failure attribution
  - SAME 可用字段：`trajectory`、`pred_path_segments`、`decision_trace`。
  - NavGPT-2 当前可用字段：`instr_id`、`trajectory`。
  - 初始诊断：early stop、loop/revisit、path length、final endpoint distance proxy、DC pair failure、RR into/exit failure。

- [ ] D.4：构建 skill × evidence 矩阵
  - 行：DC / NU / LR / RR / VM。
  - 列：trace evidence、visual-semantic evidence、topological evidence、progress/count evidence、candidate/final endpoint gap。
  - 要求：每个格子标注 evidence 是否可获得、是否可能解释 failure、是否适合作为 verifier 输入。

### 阶段 E：报告与阶段决策

- [ ] E.1：汇总 Ability Atlas 报告
  - 当前 v0 已有主表，但还缺 failure attribution 图表。

- [ ] E.2：根据 A1-A5 通过数量判断路线状态
  - L.01.04 暂定：H4 初步支持，路线 A 继续推进。

- [ ] E.3：更新 G.01 的假设检验状态
  - 建议等 D.3 / D.4 完成后再更新，避免把 skill-level 结果过度解释成 evidence-level 结论。

## 执行记录

### 已完成的阶段输入

| Stage | 对应行动项 | 当前结论 |
|---|---|---|
| `S.01.01.01.A1-SAME评估接口确认.md` | A.1 / B.1 / B.2 / B.3a | SAME 侧 NavNuances evaluation 闭环已完成 |
| `S.01.02.01.A1-NavGPT-2评估接口确认.md` | A.2 前置 | NavGPT-2 标准 R2R eval 可运行，NavNuances 适配风险可控 |
| `S.01.02.02.A1-NavGPT-2评估接口确认.md` | A.2 / C.2 | NavGPT-2 五类 NavNuances split 已完成推理与评分 |

### 当前事实源

| 模型 / 结果 | 路径 |
|---|---|
| SAME NavNuances evaluator | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/navnuances_eval/results.json` |
| SAME trajectory / decision sidecar | `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/eval_items/` |
| NavGPT-2 NavNuances evaluator | `experiment_outputs/navgpt2_navnuances_xl_20260504_213915/navnuances_eval/results.json` |
| NavGPT-2 predictions | `experiment_outputs/navgpt2_navnuances_xl_20260504_213915/preds/submit_{DC,LR,RR,VM,NU}.json` |
| NavGPT-2 standard R2R sanity baseline | `experiment_outputs/navgpt2_r2r_xl_20260504_163943/` |

## 结果分析与说明

### Ability Atlas v0 的非平凡发现

| 发现 | 支撑事实 | 可能意义 |
|---|---|---|
| DC/LR 是 NavGPT4v 相对 SAME 的强项 | DC: +29.12pp；LR: +31.48pp | 方向语义和地标语义可能需要更强 language/visual-semantic reasoning |
| RR/VM 是 SAME 相对 NavGPT4v 的强项 | RR: +33.57pp；VM: +72.24pp | 训练型 policy 的拓扑、区域、垂直移动建模仍很重要 |
| NavGPT-2 XL 与 SAME 在 DC/VM 上接近 | DC 差 1.04pp；VM 差 2.35pp | policy-fused 模型缩小了纯 MLLM 与训练型 policy 的差异 |
| LR 对本地 SAME/NavGPT-2 XL 都弱 | SAME 31.39；NavGPT-2 XL 33.58 | landmark evidence 可能是 verifier 优先方向 |
| NU 绝对表现仍不高 | NavGPT4v 39.13；SAME 33.33；NavGPT-2 XL 25.64 | numerical/progress tracking 是长期问题，但样本数只有 78，需谨慎处理 |

### 对 evidence schema 的初步指向

| Skill | 当前能力模式 | 后续 evidence 优先级 | 备注 |
|---|---|---|---|
| DC | NavGPT4v 明显强，本地 SAME/NavGPT-2 接近 | 中高 | 需要区分单次方向词理解与 trajectory-level turn pair consistency |
| LR | NavGPT4v 强，本地模型弱 | 高 | 优先考虑 landmark caption/object/CLIP/MLLM evidence |
| RR | SAME 强，NavGPT-2 次之 | 中低 | 可作为 topology/room evidence 的 sanity slice |
| VM | SAME/NavGPT-2 强，NavGPT4v 弱 | 中低 | 可作为 elevation/topology evidence 的 sanity slice |
| NU | 整体不高，NavGPT4v 略优 | 高但谨慎 | 样本少，适合作为 diagnostic slice，未必适合作为第一版方法主战场 |

## 结果复盘

L.01.04 的主要结论是：Ability Atlas 不需要大改方向，当前应该从“能不能跑、表格能不能填”推进到“失败样本为什么错、哪些 evidence 能解释错”。

当前已经可以写入上游路线的事实是：

1. SAME / NavGPT4v / NavGPT-2 XL 的 skill-level Atlas v0 已形成。
2. H4 得到初步支持，NavNuances 可以作为 EANSV 的能力坐标系。
3. A.2 / C.2 已完成，SAME-only fallback 不触发。

当前不能过度写入的结论是：

1. 不能说 LR/DC 的失败已经被证明是 visual/semantic evidence 缺失导致的。
2. 不能说 RR/VM 不需要 verifier，只能说 SAME 当前在这些 skill 上更强。
3. 不能把 NavGPT-2 XL 当成纯 LLM baseline；它更适合作为 policy-fused local baseline。

下一步最自然的 stage 是 `D.3`：构建 per-episode failure attribution 表，把 SAME 的 `eval_items` 与 NavNuances evaluator success/failure 对齐，再输出 skill × failure pattern 的第一版 evidence matrix。
