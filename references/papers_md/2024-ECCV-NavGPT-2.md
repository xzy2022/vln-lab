# NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models
**Authors**: Gengze Zhou et al.
**Venue**: ECCV 2024 (LNCS 15065, published 2025)
**Tags**: [VLN] [VLM] [LLM] [Reasoning] [Policy] [R2R]

## TL;DR
NavGPT-2 keeps the LLM frozen, injects multi-view visual evidence through a Q-former, and pairs the resulting VLM latent with a DUET-style graph policy so the agent can both navigate competitively and explain each move in natural language.

## Problem & Motivation
- 直接把 LLM 当 zero-shot VLN agent 时，常依赖脆弱的 prompt、caption 和逐步总结，且和专门的 VLN 模型仍有明显性能差距。
- 直接 fine-tune LLM 做 VLN 往往会削弱其通用语言能力，最后还变成“黑箱”导航器。
- 论文想解决的是：能否在保留 LLM 解释与交流能力的同时，把它接进一个真正能做空间规划的导航系统。

## Contribution
1. 提出 NavGPT-2：用 InstructBLIP / Q-former 接入多视角视觉信息，再用冻结 LLM 生成可读的导航推理。
2. 把 VLM latent 作为统一的视觉-语言表示，送入图结构导航 policy 做 action prediction。
3. 通过 GPT-4V 合成 10k 条导航推理数据，并展示了数据效率、跨数据集泛化和可解释性。

## Method
### Pipeline
`instruction W + multi-view observation O_t`
`-> frozen ViT-g/14`
`-> Q-former (32 learnable queries, instruction-aware)`
`-> image tokens H_i^v`
`-> frozen LLM`
`-> navigational reasoning + hidden latents`

`LLM latents + topological graph memory`
`-> node embedding / graph-aware self-attention`
`-> action score over candidate viewpoints + stop`

### Key formulation
给定指令 $W=\{w_i\}_{i=1}^L$ 和图 $G=(V,E)$，策略为：

$$\pi(a_t \mid W, O_t; \Theta)$$

VLM 侧用冻结视觉编码器和 Q-former 把多视角图像投到 LLM 空间；policy 侧将 LLM latent 和图记忆结合，做节点级 action prediction。论文中给出的训练目标是：

$$\mathcal{L} = \lambda \mathcal{L}_{BC} + \mathcal{L}_{DAG}$$

图上节点表示和 cross-modal 编码可概括为：

$$V = \text{SelfAttn}\left(\frac{1}{M}\sum_i \left(\bar H_i^v + E_i^d + E_i^s\right)\right)$$

$$\text{GASA}(V)=\text{Softmax}\left(\frac{VW_q(VW_k)^T}{\sqrt d}+A(E_t)\right)VW_v$$

### Training
1. Stage 1: 用 R2R 中随机抽取的 10k intermediate steps，借助 GPT-4V 生成单步导航推理，先只训练 Q-former 和 projection layer。
2. Stage 2: 冻结 VLM，只训练下游 graph-based policy。
3. 实际训练时还使用 R2R + PREVALENT synthetic data，policy optimization 结合 BC 和 DAgger。

## Key Results
| Setting | Metric | NavGPT-2 | Baseline / note |
|---|---|---:|---|
| R2R val unseen, FlanT5-XXL w/ PREVALENT | SR / SPL | 72 / 61 | DUET: 72 / 60 |
| R2R test unseen, FlanT5-XXL w/ PREVALENT | SR / SPL | 71 / 60 | DUET: 69 / 59; NaviLLM: 68 / 60 |
| R2R val unseen, 50% training data | SR | 63.30 | DUET: 59.90 |
| RxR-EN unseen | SR / SPL | 28.75 / 22.36 | DUET: 25.07 / 19.65 |
| HM3D zero-shot | SR / SPL | 47.20 / 27.99 | DUET: 25.60 / 13.32 |

补充的人类研究里，NavGPT-2 生成的 reasoning 在 30 个样本上得到 1.66 / 1.93 / 1.78（accuracy / informativeness / rationality），而 GPT-4V 生成的数据约为 2.31 / 2.95 / 2.34。

最重要的发现是：它在保留解释能力的同时，把 LLM-based VLN 的性能拉到了和专门 VLN policy 接近的水平，且在跨数据集 HM3D 上的泛化提升很明显。

## Limitation
- 作者在补充材料里明确说，reasoning 目前是单步、局部的，没有把完整 navigation history 放进 VLM 内部，所以 reasoning 之间的一致性没有被充分建模。
- reasoning 和 action 不是严格同步的，这部分未来还需要显式对齐。
- communicative capability 没有被系统评估。
- 质化结果里仍能看到 hallucination、误判物体方向等问题。

## 与本项目的关系
NavGPT-2 和本仓库的关系很直接：它已经被放进 `third_party/NavGPT-2`，而且本项目的 NavGPT-2 接口确认记录已经说明标准 R2R eval 能跑通，trajectory 也能解析。

对当前研究来说，它适合做两个角色：

1. 作为 Ability Atlas / NavNuances 的 reasoning-heavy baseline，和 SAME 的 task-adaptive routing 形成互补对照。
2. 作为解释能力 baseline，帮助判断某个 skill 失败到底是 policy 不足，还是“有话说但走不对”。

当前更现实的结论是：NavGPT-2 不是本项目里最省事的主模型，但它是很好的对照组。标准 R2R 已经能评测，NavNuances 五类 split 还需要补一个小的 eval-split 适配层，才能做真正的 per-skill 对齐。
