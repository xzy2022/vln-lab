# SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts

**Authors**: Gengze Zhou, Yicong Hong, Zun Wang, Chongyang Zhao, Mohit Bansal, Qi Wu
**Affiliation**: The University of Adelaide, Adobe Research, UNC Chapel Hill, UNSW Sydney
**Venue**: ICCV, 2025
**Code**: https://github.com/GengzeZhou/SAME
**Tags**: [VLN] [Multi-Task] [MoE] [Unified Agent]

## TL;DR

将 7 个语言引导的视觉导航任务统一到一个框架中，提出 State-Adaptive Mixture of Experts (SAME) 模型——根据当前状态的跨模态特征（语言+视觉观测）动态路由到不同 expert，解决了多任务联合训练中的冲突问题，在单个模型上匹配各任务专用模型的性能。

## Problem & Motivation

- 现有导航任务（R2R、REVERIE、ObjectNav 等）被当作独立问题研究，各自的方法互相不通用
- 这些任务的本质差异在于**语言指令的粒度**不同：
  - **Fine-grained VLN**: 逐步描述动作序列（R2R, RxR, CVDN）
  - **Coarse-grained VLN**: 描述远程目标物体（REVERIE, SOON）
  - **Zero-grained VLN**: 仅给出目标类别（ObjectNav）
- 简单混合数据做多任务训练会导致**性能此消彼长**——尤其是 REVERIE 在混合训练后 SR 下降 6-7%
- 现有零样本 LLM 方案 [71] 在 VLN 上效果有限

## Contribution

1. 首次将 VLN 任务按语言粒度系统分类（fine / coarse / zero-grained），并分析了多任务训练冲突的根因
2. 提出 SAME——一种新的 MoE 路由机制，**基于状态（当前视觉观测 + 语言指令的跨模态特征）**而非任务 ID 或 token 来动态选择 expert
3. 在 7 个导航任务上训练统一 agent，首次达到与各任务专用模型可比或更优的性能

## Method

### 导航任务统一形式化

智能体在图 $G = \langle V, E \rangle$ 上执行动作序列 $\{s_0, a_0, s_1, a_1, \ldots, s_T, a_T\}$，状态 $s_t = \langle v_t, \theta_t, \phi_t \rangle$。指令 $W$ 按粒度分为三类。

### 基础架构：DUET

- Vision encoder 编码 36 视角图像 $O_t \to \hat{O}_t$
- Language encoder 编码指令 $W \to \hat{W}$
- Local cross-attention: $\text{CrossAttn}(\hat{O}_t, \hat{W})$ 做视觉-语言对齐
- Global cross-attention: $\text{CrossAttn}(\hat{G}_t, \hat{W})$ 利用拓扑图历史
- 最终导航分数: $s_i = \sigma_t s_i^l + (1 - \sigma_t)s_i^g$

### SAME 核心设计

#### MoE 形式化

路由器预测专家分配概率:
$$P(x_r) = \text{Softmax}(W x_r)$$

Top-k 专家加权输出:
$$\text{MoE}(x, x_r) = \sum_{i \in \mathcal{T}} P(x_r)_i f_i(x)$$

Load balancing loss:
$$\mathcal{L}_{\text{balance}} = N \sum_{i=1}^{N} F_i D_i$$

#### 路由策略对比

| 策略 | 路由特征 $x_r$ | 效果 |
|------|---------------|------|
| Token-wise MoE | $x_i$（单个 token） | 次优 |
| Task-wise MoE | $E^{task}$（任务嵌入） | 更差，显式任务信息反而妨碍 skill sharing |
| Text [CLS] | $\hat{W}^{CLS}$ | 中等 |
| **SAME (ours)** | $W_m [\frac{1}{L}\sum \hat{O}_t; \hat{W}^{CLS}]$ | **最优** |

SAME 路由特征融合了视觉和语言的跨模态信息，让 expert 选择基于当前状态，而非任务标签。

#### Expert 部署位置

将 MoE 专家部署在 cross-attention 的 **visual queries** $W_q$ 上效果最优（优于 FFN 和 textual key/value），因为跨模态注意力是 agent 决策的核心。

### 训练配置

- 数据: ScaleVLN pretrain + R2R + RxR-EN + CVDN + REVERIE + SOON + Habitat-Web (采样比 10:1:1:1:1:1:2)
- 7 个任务混合训练，**batch 内不混合数据**
- 初始化: ScaleVLN [108] pretrained weights
- Visual encoder: CLIP ViT-B/16
- DAgger 采样训练 + load balancing loss ($\lambda = 0.8$)
- 8 experts per layer, top-2 activated
- 参数: 215.74M (比 DUET 多 34.45M, 但 83.6% 参数共享)
- FLOPs 仅增加 5.2%

## Key Results

### 离散环境（MP3D）— 7 任务统一模型

SAME 作为单一模型在所有 7 个任务上取得 SoTA 或接近 SoTA：

| Task | Metric | SAME | 最佳专用模型 |
|------|--------|------|-------------|
| R2R test unseen | SR | 77 | ScaleVLN 79 |
| R2R test unseen | SPL | 68 | ScaleVLN 68 |
| RxR-EN val unseen | nDTW | 76.3 | ScaleVLN 79 |
| REVERIE test unseen | SR | 56.1 | GOAT 57.0 |
| REVERIE test unseen | SPL | 39.5 | GOAT 41.8 |
| CVDN test | GP | 7.07 | ScaleVLN 6.97 |
| SOON val unseen | SR | 48.6 | — |
| ObjectNav-MP3D val | SR | 76 | ScaleVLN 76 |
| ObjectNav-MP3D val | SPL | 43.4 | ScaleVLN 42.7 |

vs 其他统一模型: 比 NaviLLM 在 R2R 上 SR 高 6%, REVERIE 上高 9%, 参数少 30 倍。

### 连续环境（Habitat）— 零样本迁移

| 方法 | R2R-CE NE↓ | R2R-CE SR↑ | ObjectNav SR↑ |
|------|-----------|-----------|--------------|
| SAME | 5.31 | 47 | 43 |
| Habitat-Web | — | — | 35 |

### 关键消融发现

- **DAgger + 顺序采样** 比 imitation learning + batch混合 在 REVERIE 上提升 17.3% SR
- **MoE 平衡系数** $\lambda=0.8$ 最优（$\lambda=0.2$ 时各任务均下降）
- **VLN pretrain** 带来约 14% SR 提升（R2R）
- 添加**显式 Task Embedding** 到路由特征中反而降低性能（约 1% SR）

## Limitation

1. ObjectNav 仅使用离散环境中的 MP3D 数据进行实验，未见连续环境大规模 ObjectNav 的训练
2. 训练数据仍主要是 VLN 领域数据，对 zero-grained 任务（ObjectNav）的探索有限
3. MoE expert 数量固定为 8，未探索 expert 数量对任务可扩展性的影响
4. 论文主要关注离散导航，连续环境的评估仅为零样本迁移
5. 模型可解释性不足——未深入分析不同 expert 学到了哪些具体技能（如"探索"vs"指令跟随"）

## 与本项目的关系

本仓库 (`vln-lab`) 直接以 SAME 为实验基础设施，`third_party/SAME` 即该论文的官方代码仓库。本项目在此基础上：

- **复用了 SAME 的 DUET+MoE 架构和训练框架**，通过 `run_same.py` 包装实验流程
- 通过 5 个 base patches 对 SAME 做了适配性修改（eval-only 退出、stdout 重定向、CVDN/SOON 指标导出等）
- 当前实验主要聚焦 R2R/RxR/REVERIE/CVDN/SOON 这 5 个 VLN 任务，尚未启用 ObjectNav 能力
- **潜在方向**: SAME 的 MoE routing 机制为多任务 VLN 提供了良好的基础，可以考虑在此之上引入 NavNuances 等新评估基准的对接；也值得关注 SAME 原作者后续的 NavGPT-2 [127] 工作（已引入本仓库 `third_party/NavGPT-2`）
