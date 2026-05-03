# 01.证据自适应导航状态验证

## 目标 / 任务

将研究主线从 STOP endpoint 修补升级为**证据自适应导航状态验证（Evidence-Adaptive Navigation State Verification, EANSV）**框架。

核心问题：训练型 VLN agent 在不确定状态下，如何通过额外证据安全地验证当前状态、候选 endpoint、instruction progress 与目标语义。

成功标准：
- offline verifier 的 recovered/harmed evidence margin 能在 dev 与 val_seen 上显著分开
- weighted val_unseen dSR >= +0.20pp，harm <= 1pp，max dataset harm <= 1pp
- verifier 调用率在高 uncertainty 下显著高于低 uncertainty
- NavNuances 五大能力切片至少有 DC / NU / LR / RR / VM 的诊断解释

## 背景与输入

### 外部文献事实

| 论文 | 核心贡献 | 对本路线的意义 |
|---|---|---|
| DUET (CVPR 2022) | dual-scale graph transformer，强训练型 baseline | SAME/DUET 类模型应作为主 backbone |
| Mind the Gap (ACM MM 2023) | SR-OSR gap 建模，up to 9% | STOP gap 是真实问题，但不能作为唯一主线 |
| NavGPT (AAAI 2024) | 纯 LLM zero-shot navigation | LLM 强在 reasoning，但不等于强 navigation policy |
| NavGPT-2 (ECCV 2024) | LLM + VLN policy 融合 | 架构借鉴对象，不作为主实验母体 |
| SAME (ICCV 2025) | 统一多任务 MoE，simulator-free | 当前最适合的 quick reference backbone |
| NavNuances (EMNLP Findings 2024) | 五类细粒度能力评估 | 将 R2R aggregate 转向 ability-slice 诊断 |
| AdaNav (arXiv 2025) | uncertainty-based adaptive reasoning | uncertainty-triggered reasoning 的直接参考 |
| VLN-Copilot (ECCV 2024) | 困难状态向 LLM copilot 求助 | LLM 作为辅助推理层的架构参考 |
| VLN-MP (IJCAI 2024) | 多模态 prompt 增强导航 | evidence augmentation 的多模态方向 |
| NavBench (arXiv 2025) | MLLM embodied navigation 诊断 | temporal progress estimation 困难，支持 progress verification |
| LH-VLN (CVPR 2025) | long-horizon VLN benchmark | 远期连接，当前不优先推进 |
| SACA (arXiv 2026) | step-aware contrastive alignment | 向 step-level progress evidence 扩展的依据 |

### 个人实验事实

| 实验 ID | 内容 | 结论 |
|---|---|---|
| SAME baseline | R2R val_unseen SR=76.29, SPL=66.24；REVERIE SR=45.84 | CVDN/SOON 基线已建立 |
| NavNuances 复现 | SAME vs NavGPT：SAME 强 RR/VM（89.82/85.88），NavGPT 强 DC/LR（92.68/62.87），NU 普遍弱（~30-40%） | 两大类模型能力互补 |
| 旧 endpoint 项目 | trace-only gate + ranker recovery ≈ harm，无法稳定恢复 | trace-only insufficient，需外部证据 |
| SAME checkpoint eval | 已建立可复用工程资产 | 快速诊断的基础 |

### 工程约束

| 约束 | 对路线的影响 |
|---|---|
| 研一 VLN 学生，需积累可复现资产 | 不宜频繁换方向；优先轻量、离线、diagnostic-first |
| 本地 RTX 5060 / 云端 1-2×4090 / 实验室 8×4090 | 初期 offline verifier / prompt verifier 优于 full training |
| AI 做资料整合与代码构建，人类做长期一致性 | 路线文档非常重要，global/local/stage maps 须成为事实源 |
| val_unseen 只能做冻结后一次性报告 | 阈值/参数选择须在 train/dev 或 val_seen 上完成 |

## 共识与经验

### 学术共识

1. VLN 从单任务特化转向统一训练/多任务 agent（SAME 为代表）
2. 从 aggregate benchmark 转向 fine-grained diagnosis（NavNuances 为代表）
3. 从 LLM 直接导航转向 LLM 作为 reasoning/copilot/verifier（NavGPT-2、VLN-Copilot、AdaNav）
4. 从固定推理转向 uncertainty-adaptive reasoning（AdaNav）
5. 从 final outcome reward 转向 step-level progress / long-horizon consistency（NavBench、LH-VLN、SACA）

### 工程经验

1. 对当前资源，复现与扩展 SAME 比训练新 backbone 更稳
2. 轻量 offline verifier、数据诊断、evidence extraction 性价比高于直接 full training
3. 任何 endpoint correction 方法要先看 recovery/harm，不能只看平均 SR/SPL
4. LLM/MLLM 调用要有 budget 与 trigger，否则成本高且难复现

### 个人经验

- 已亲自验证：单条轨迹内部 trace 对 STOP/endpoint 有弱信号，但不足以形成稳定方法
- 已有 negative diagnostic 可直接作为新路线的动机：trace-only insufficient → 需要外部证据

## 理解与解释

### 为什么 LLM 不能直接替代训练型 backbone

NavGPT 的优势在 reasoning 与语言理解，但离散 VLN 的成功需要长期 trajectory consistency、topological state、action grounding、stop calibration、visual observation alignment。DUET/SAME 通过大量 VLN 数据、拓扑图、cross-modal policy learning 获得这些能力。

LLM 更适合担任**局部 reasoning / verification / explanation 模块**，而非主导航器。

### 为什么 SAME 与 NavGPT 在 NavNuances 上互补

| Skill | LLM/MLLM 可能优势 | 训练型 VLN 可能优势 |
|---|---|---|
| DC（方向变化） | 方向词语义、显式推理 | heading-action alignment |
| LR（地标识别） | open-vocabulary grounding | 训练数据内 landmark grounding |
| RR（区域识别） | 语义区域理解 | topo-map / route prior / trained room patterns |
| VM（垂直移动） | 语义 elevation | 3D action / graph elevation / trained path |
| NU（数量理解） | counting/ordinal | history tracking，普遍弱 |

### 旧 STOP 项目的重新定位

旧路线 pipeline 中缺少的关键环节：

```
candidate endpoint 是否真的满足 instruction goal？
final endpoint 与 candidate endpoint 的证据差距是否足够大？
这个 episode 属于哪个 skill failure？
当前 policy uncertainty 是否足以触发额外推理？
```

STOP 应从"主任务"降级为"状态验证框架的一个评测切片"。

## 问题建模

### 系统边界

不重训主导航器（以 SAME 为 backbone）。研究对象是 backbone 之外的 verification layer。

```
Backbone policy B: 给定 instruction I 与 observation/history H_t，输出 action distribution、trajectory、candidate endpoints。
State s_t: 当前 viewpoint、heading、elevation、history、topological context。
Candidate set C: final endpoint + trajectory visited endpoints + top-k ranker candidates。
Uncertainty U: action entropy、stop probability shape、OSR-SR gap proxy、loop/revisit、score margin、progress inconsistency。
Evidence E: trace evidence、visual evidence、room/object/landmark metadata、topology/elevation、dialog evidence、instruction clause evidence。
Verifier V: 输入 I, s_t/c, E，输出 candidate 是否满足 instruction / progress / target 的 score 或 margin。
Decision D: 只在 V(candidate) - V(final) 足够大且 harm risk 低时介入。
```

### 关键变量

| 变量 | 类型 | 可能实现 |
|---|---|---|
| `skill_type` | categorical | DC / NU / LR / RR / VM / endpoint / dialog-goal |
| `policy_uncertainty` | continuous | entropy、top1-top2 margin、stop probability variance |
| `trace_risk` | continuous | loop、revisit、trajectory length、late stop、candidate distance proxy |
| `visual_semantic_score` | continuous | CLIP / object tag / room classifier / MLLM caption match |
| `instruction_clause_alignment` | continuous / structured | current clause match、completed clause count、entity overlap |
| `topological_consistency` | continuous | room transition、elevation transition、candidate neighborhood |
| `verifier_margin` | continuous | best non-final vs final evidence score |
| `decision` | binary / multiclass | keep final / switch endpoint / ask verifier / no action |

### 因果链条

```
Instruction skill type
    -> required evidence type
        -> backbone internal representation adequacy
            -> uncertainty / confusion
                -> verifier trigger
                    -> evidence-grounded state decision
                        -> recovery or harm
```

旧 STOP 项目只覆盖了链条中的 trace evidence → gate/ranker → recovery or harm。

### 失败位置

- trace-only verifier 无法分离 recovered/harmed（已由 stage 4.7/4.8 验证）
- candidate ranker 与 final endpoint 的 evidence gap 未被建模

### 指标关系

| 指标 | 方向 | 与核心目标的关系 |
|---|---|---|
| recovered/harmed evidence margin | 越大越好（margin 显著分开） | 直接反映 verifier 区分能力 |
| weighted val_unseen dSR | >= +0.20pp | 端点恢复的最终价值 |
| harm rate | <= 1pp per dataset | 端点切换的安全上限 |
| verifier call rate (high vs low uncertainty) | high > low | 验证 uncertainty-triggered 假设 |
| NavNuances skill coverage | >= 5 skills | 贡献的可解释性 |

## 假设

### 核心假设

1. **H1：不同 instruction skill 需要不同 evidence budget**
   - 统一 verifier 对 DC/NU/LR/RR/VM 各技能的 evidence 适用性不同
   - skill-aware portfolio 优于 uniform evidence allocation
   - 判别标准：skill-specific verifier 的 recovered/harmed margin 优于统一 verifier

2. **H2：uncertainty-triggered verifier 比固定调用 verifier 更稳**
   - 高 uncertainty 状态下调用 verifier 时 recovery/harm 比率优于低 uncertainty 固定调用
   - 判别标准：triggered 版本在相近 dSR 下 harm 更低或调用比例显著降低

3. **H3：endpoint recovery 需要 final-vs-candidate evidence gap**
   - 只比较 candidate score 不够安全；必须比较 `V(candidate) - V(final)`
   - 判别标准：引入 gap 的决策规则后 harm 显著降低

### 支撑假设

4. **H4：NavNuances 应超出 benchmark 角色，成为 route 的能力坐标系**
   - SAME vs NavGPT 互补结果可映射到 skill-specific evidence 设计
   - 判别标准：Atlas 中 skill × evidence 矩阵有可解释的模式

5. **H5：旧 endpoint negative result 可以转化为论文级贡献**
   - diagnostic / evidence insufficiency 的表述比包装成强方法更具学术价值
   - 判别标准：诊断结果（trace-only insufficient）可与 verifier 结果形成对照

### 隐含假设

- SAME backbone 在验证集上足够稳定，不会因额外推理而意外恶化
- offline verifier 的证据可离线收集，不依赖实时 simulator
- 领域/类型先验不冲突：单个 episode 可以同时属于多个 skill 类别

## 当前判断

1. **当前目标**：建立 EANSV 框架，从旧 STOP 项目过渡到结构化的 evidence 自适应验证系统
2. **面临困难**：旧路线证明 trace-only insufficient，但新路线需要明确从何处获取"外部证据"；Ability Atlas 尚未构建，无法做 skill × evidence 的系统设计
3. **关键假设能否解决困难**：H3 的 evidence gap 建模是直接针对"从哪里获取外部证据"的可操作路径；H1 的 skill-aware portfolio 能为不同 evidence 类型提供优先级指引；H4 的 Ability Atlas 是系统设计的基石，但需要较多工程投入
4. **关键假设证伪能否转化困难**：若 evidence gap 和 skill-aware 方向均不成立，则问题可能根本不在 evidence 类型不足，而在 policy 内部表征本身；此时诊断路线（D）将成为主要贡献形式
5. **若3和4均为否的应对**：若 H1/H2/H3 均被证伪，说明 EANSV 框架本身方向错误；此时回退到 D 路线（以 negative diagnostic 为核心贡献），同时考虑将诊断经验扩展到更广泛的 VLN agent 通用验证协议

## 决策分析

### 路线 A：从 Ability Atlas 构建开始

**直接目的**：搜集信息（A），为验证 H1/H4 做准备。

主要思路：从诊断视角出发，先用 SAME + NavNuances 构建能力坐标地图，明确哪些 skill 失败是 evidence 缺失导致的、哪些是 policy 不足导致的。

结果理想时：skill × failure 的可解释矩阵直接指导 verifier 的 evidence schema 设计；贡献形式为 NavNuances benchmark 上的新诊断报告。

结果不理想时：说明 SAME 当前的失败模式不够结构化，无法映射到 individual skill × evidence。

可行性评估：
- 时间：能力范围内，单人 2-3 周完成
- 算力：不需要重新训练，仅做 SAME eval + NavNuances 适配脚本
- 难度：主要是工程适配，风险低
- 论文价值：中等——诊断报告有价值，但缺方法贡献

### 路线 B：从 offline verifier 原型开始

**直接目的**：搜集信息（A），同时部分验证 H3。

主要思路：绕过完整的 Ability Atlas，直接用 CLIP / topology-based evidence 构建一个轻量 offline verifier，在 candidate endpoint 排序任务上验证 H3。

结果理想时：evidence gap 的决策规则在 dev 上证明有效，直接推进 stage_map 验证

结果不理想时：说明 gap-based decision 在 offline setting 下不稳定，或证据类型不对

可行性评估：
- 时间：需要探索 evidence extraction 的接口定义，周期较长
- 算力：CLIP inference 需要 GPU，但量不大
- 难度：需要理解 SAME 的 candidate endpoint 接口，有一定工程风险
- 论文价值：高——方法贡献清晰

### 路线 C：两条路线并行但优先级不同

**直接目的**：搜集信息（A），两条线互为补充。

主要思路：快速启动路线 A（低风险），同时以异步方式准备路线 B 的证据接口代码。

结果理想时：Atlas 为路线 B 提供优先级指引；两条路线互为补充

结果不理想时：如果路线 A 产出低，则路线 B 的准备时间被浪费

可行性评估：
- 时间：需要同时跟进两条线，人力紧张
- 算力：无冲突，可并行
- 难度：需要在阶段初期明确拆分点，避免两条线都半途而废
- 论文价值：高且多元——诊断报告 + 方法贡献

### 路线 D：以旧路线的 negative diagnostic 为核心贡献

**直接目的**：信息已足够，验证 H5。

主要思路：将 endpoint completion 的 negative result 系统化，与 trace-only insufficient 的发现一起打包成"evidence insufficiency"的诊断框架。

结果理想时：形成有特色的论文贡献——诊断先行，方法为辅

结果不理想时：评审可能认为没有方法贡献，难以发表

可行性评估：
- 时间：最低，主要是整理和写作
- 算力：无额外消耗
- 难度：写作和叙事构建难度高
- 论文价值：取决于诊断的深度和独特性

**综合建议**：当前推荐**路线 C**（Atlas 先行 + verifier 原型异步准备），因为路线 B 的证据接口探索需要先有 Atlas 的能力坐标才能确定优先级。路线 D 可以作为备用保底策略，取决于后续论文投稿的反馈。

## 验证设计

TBD

（人类根据决策分析选定路线后，生成 Stage A-E 的验证设计）

## 行动计划

TBD

（与验证设计一同由人类明确后生成）

## 执行记录

（人类决策后启动，实时更新）

## 结果复盘

（实验结束后填写）
