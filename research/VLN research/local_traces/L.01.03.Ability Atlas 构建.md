# L.01.01.Ability Atlas 构建

## 目标 / 任务

基于路线 A：从 SAME + NavNuances 构建能力坐标地图（Ability Atlas），诊断哪些 skill 失败由 evidence 缺失导致、哪些由 policy 不足导致，为后续 evidence-adaptive verifier 设计提供优先级指引。

## 背景与输入

### 来自 G.01 的决策输出

路线 A 已由全局轨迹（`G.01.证据自适应导航状态验证.md`）明确选定：

**直接目的**：搜集信息（A），为验证 H1/H4 做准备。

**主要思路**：从诊断视角出发，先用 SAME + NavNuances 构建能力坐标地图，明确哪些 skill 失败是 evidence 缺失导致的、哪些是 policy 不足导致的。

**预期结果**：
- 理想时：skill × failure 的可解释矩阵直接指导 verifier 的 evidence schema 设计；贡献形式为 NavNuances benchmark 上的新诊断报告
- 不理想时：说明 SAME 当前的失败模式不够结构化，无法映射到 individual skill × evidence

**可行性**：单人 2-3 周完成；无需重新训练，仅做 SAME eval + NavNuances 适配脚本；风险低。

### 关键假设摘要

来自 G.01：

| ID | 假设 | 验证指标 |
|---|---|---|
| H1 | 不同 instruction skill 需要不同 evidence budget | skill-specific verifier margin > uniform verifier margin |
| H3 | endpoint recovery 需要 final-vs-candidate evidence gap | gap-based decision 后 harm 显著降低 |
| H4 | NavNuances 应超出 benchmark 角色，成为 route 的能力坐标系 | Atlas 中 skill × evidence 矩阵有可解释的模式 |
| H5 | 旧 endpoint negative result 可转化为论文级贡献 | diagnostic 结果与 verifier 结果形成对照 |

本 local trace 聚焦于 **H4 的前置验证**——Ability Atlas 是 skill × evidence 矩阵的基础。

## 验证设计

### 验证目标

针对 **H4**（NavNuances 应成为 route 的能力坐标系）进行系统验证，同时为 H1（skill-specific evidence）提供事实基础。

### 验证思路

构建 Ability Atlas = SAME/NavGPT 能力 × NavNuances skill 类型的二维诊断矩阵，核心问题是：**SAME 在各 skill 上的失败模式是否足够结构化，能为后续 evidence 设计提供依据？**

具体分 5 个子验证项：

| 子验证 | 验证什么 | 判别标准 |
|---|---|---|
| A1 | SAME 在 NavNuances 五类 skill 上的 SR/SPL 分布 | 能否观察到 5 类 skill 的显著差异（任一 skill 的 CI 不与全量 CI 完全重叠即通过） |
| A2 | SAME vs NavGPT 在各 skill 上的互补模式 | 至少 3 个 skill 呈现明确互补（SAME 优 vs NavGPT 优），而非全部一致 |
| A3 | SAME 的 failure 是否可归因到 evidence 类型缺失 | 同一 skill 内，错误样本的 evidence 类型（trace/trajectory/candidate/visual）是否呈现偏向性 |
| A4 | failure pattern 的 skill × evidence 矩阵是否可解释 | 矩阵中大多数格子能用已知知识（训练数据分布/模型架构/数据规模）解释，而非随机噪声 |
| A5 | 诊断报告能否形成独立的论文价值 | 至少 2 个非平凡发现（如：某 skill 的 same 弱于 chance、某 evidence 类型在特定 skill 缺失） |

### 预期结果与应对

| 预期结果 | 意味着什么 | 应对策略 |
|---|---|---|
| Atlas 呈现清晰的结构化模式（4/5 子验证通过） | H4 得到支持；H1 的 skill-aware portfolio 有充足事实基础 | 推进 stage 1.1 的 evidence schema 设计；路线 B 可以异步启动 |
| Atlas 部分结构化（2-3 子验证通过） | H4 部分成立；需要区分"可解释 skill"与"随机 skill" | 聚焦可解释 skill 的 evidence 设计；随机 skill 归入 policy 问题（非本路线重点） |
| Atlas 呈随机噪声（0-1 子验证通过） | H4 不成立；SAME/NavNuances 失败模式不可结构化 | 路线 A 降级为纯诊断报告；主要贡献转向 H5（negative diagnostic） |

### 结果汇总表

指标口径：DC/LR/RR/VM 使用 SR，NU 使用 path_SR，Mix 暂用标准 R2R SR。SAME 列来自 `experiment_outputs/0016_same_navnuances_r2r_eval_only_same_s0_navnuances_v2/navnuances_eval/results.json`；NavGPT 列采用 NavNuances 原论文中的 `NavGPT4v (0-shot)`，不复现；NavGPT-2 列采用 `S.01.02.02.A1` 中本地 `NavGPT-2 XL / FlanT5-XL` 五类 split 评估结果。

| Skill | SAME | NavGPT4v (0-shot, paper) | NavGPT-2 XL (ours) | SAME vs NavGPT4v | SAME vs NavGPT-2 XL | SAME σ | Pattern | Evidence Gap |
|---|---:|---:|---:|---:|---:|---|---|---|
| DC | 63.56 | 92.68 | 62.52 | -29.12pp | +1.04pp | TBD | NavGPT4v 强；SAME≈NavGPT-2 XL | TBD |
| NU | 33.33 | 13.64 | 25.64 | +19.69pp | +7.69pp | TBD | 整体弱；SAME 暂优 | TBD |
| LR | 31.39 | 62.87 | 33.58 | -31.48pp | -2.19pp | TBD | NavGPT4v 强；SAME≈NavGPT-2 XL 弱 | TBD |
| RR | 89.82 | 56.25 | 77.45 | +33.57pp | +12.36pp | TBD | SAME 强；NavGPT-2 XL 次之 | TBD |
| VM | 85.88 | 39.13 | 83.53 | +46.75pp | +2.35pp | TBD | SAME≈NavGPT-2 XL 强；NavGPT4v 弱 | TBD |
| Mix | 76.29 | 54.78 | TBD | +21.51pp | TBD | TBD | TBD | TBD |

## 行动计划

### 阶段 A：工程准备（预计 2-3 天）

- [x] A.1：确认 SAME 在 NavNuances 数据集上的评估接口
  - 检查 `data/navnuances/` 目录结构、数据格式（episodes、annotations）
  - 检查 SAME 的 `task.*` 配置是否支持 NavNuances split（val_unseen）
  - 确认 SAME 输出中 trajectory/action sequence 的可解析性
  - **停止条件**：若 SAME 完全不支持 NavNuances 格式，路线 A 需重新评估

- [ ] A.2：确认 NavGPT-2 在 NavNuances 上的评估接口
  - 检查是否有现成的 NavGPT eval 脚本或已运行的结果
  - 若无，检查 NavGPT-2 的输入格式与 NavNuances 的兼容性
  - **停止条件**：若 NavGPT-2 无法适配 NavNuances，改用 SAME-only Atlas（仅分析 SAME 的 skill 分布）

- [ ] A.3：建立诊断脚本框架
  - 在 `scripts/analysis/` 下创建 `ability_atlas/` 目录
  - 确认连接图（connectivity）和 SAME 原始输出的可访问路径
  - **停止条件**：无

### 阶段 B：SAME Eval on NavNuances（预计 3-5 天）

- [ ] B.1：运行 SAME val_unseen 评估，收集原始 trajectory 数据
  - 使用现有 SAME 容器环境
  - 输出要求：per-episode 的 trajectory、action sequence、candidate endpoints（若 ranker 输出）
  - **停止条件**：若评估超时或 OOM，降至单 GPU 模式

- [ ] B.2：编写 NavNuances episode → SAME format 适配脚本
  - 将 NavNuances 注解（instruction、skill label、goal object、panorama viewpoint）映射为 SAME 的输入格式
  - 确保 skill label 与 episode 一一对应
  - **停止条件**：若适配脚本运行失败，检查数据格式差异，逐字段排查

- [ ] B.3：对齐 SAME 输出与 NavNuances ground truth，计算 per-skill 指标
  - skill 分类维度：DC（方向变化）、NU（数量理解）、LR（地标识别）、RR（区域识别）、VM（垂直移动）
  - 指标：SR（成功率）、SPL（路径长度效率）、steps（导航步数）
  - 输出：per-skill metrics JSON + per-episode 详细日志
  - **停止条件**：若某 skill 的 episode 数 < 5（统计意义不足），在 Atlas 中标记为"insufficient data"并跳过该 skill 的子验证 A1/A3

### 阶段 C：NavGPT 对比（预计 1-2 天）

- [ ] C.1：若 NavGPT 结果已存在，直接导入并对齐 episode ID
  - 检查 episode ID 匹配率；若匹配率 < 80%，记录缺失原因
  - **停止条件**：无（N/A）

- [ ] C.2：若 NavGPT 结果不存在，尝试快速评估或使用已有论文数据点
  - 引用 G.01 中的已有数据（SAME vs NavGPT 互补结果表）作为补充证据
  - **停止条件**：若无法获得任何 NavGPT 数据，在 Atlas 中注明"SAME-only comparison"

### 阶段 D：诊断矩阵构建（预计 2-3 天）

- [ ] D.1：构建 SAME SR/SPL 的 per-skill 分布表（对应子验证 A1）
  - 计算每个 skill 的均值、标准差、95% CI
  - 与全量结果做统计对比（t-test 或 bootstrap）

- [ ] D.2：分析 SAME vs NavGPT 的 skill 互补模式（对应子验证 A2）
  - 标记：SAME 显著优于 NavGPT 的 skill（delta > +5pp）、NavGPT 显著优于 SAME 的 skill（delta < -5pp）、相近的 skill（|delta| < 5pp）

- [ ] D.3：基于 trajectory evidence 分析 failure 归因（对应子验证 A3）
  - 对每个 skill 的失败 episode，提取：stop 是否过早（early stop ratio）、是否 loop/revisit、candidate endpoint 与最终 trajectory 的距离
  - 关联 evidence 类型：trace evidence（历史动作序列）、visual evidence（panorama object/room）、topological evidence（room transition）
  - 输出：per-skill failure pattern 的 evidence 分布饼图

- [ ] D.4：验证 skill × evidence 矩阵的可解释性（对应子验证 A4）
  - 用已知知识（训练数据量、架构特性、skill 语义复杂度）解释矩阵中的每个 pattern
  - 标记"随机噪声"格子（如：某 skill 失败但无 evidence 类型偏向）

### 阶段 E：报告与阶段决策（预计 1-2 天）

- [ ] E.1：汇总 Ability Atlas 报告
  - 整理 Atlas 矩阵、per-skill 指标、failure pattern 分析
  - 标注哪些 skill 是 evidence 类型缺失导致的、哪些是 policy 不足导致的

- [ ] E.2：根据子验证 A1-A5 的通过数量，判断结果落在哪个预期区间
  - ≥4 通过：推进路线 B（evidence-adaptive verifier）
  - 2-3 通过：聚焦可解释 skill，路线 B 降级为"部分 skill verifier"
  - ≤1 通过：路线 A 降级为纯诊断报告；主要贡献转向 H5

- [ ] E.3：更新 G.01 的假设检验状态，将 Atlas 结果反馈到全局轨迹

## 执行记录

（实验启动后实时更新）

### 代码实现原理

### 执行脚本实例

### 结果分析与说明

## 结果复盘

（实验结束后填写）
