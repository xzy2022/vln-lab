# NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models
**Authors**: Gengze Zhou, Yicong Hong, Qi Wu
**Venue**: AAAI 2024
**Tags**: [VLN] [LLM] [Zero-Shot] [Prompting] [ReAct] [R2R]

## TL;DR
NavGPT 把 GPT-4 直接当成 VLN agent：通过视觉基础模型把全景观测翻译成自然语言描述，再用 ReAct 风格的 Thought + Action 循环让 LLM 在 R2R 上做 zero-shot 序列动作预测，全程不训练任何模块。

## Problem & Motivation
- 既有 VLN 工作普遍依赖在 R2R/REVERIE 等数据上做 supervised pre-train + fine-tune，泛化到 unseen 环境受限于训练分布。
- 已有把 LLM 接入导航的尝试（landmark 解析、外部知识、ZSON 等）都把 LLM 当辅助模块，**LLM 本身的推理过程并未直接驱动导航决策**。
- 真正的问题：纯 LLM 能否理解“可交互世界”里的 action / consequence，并仅凭自然语言把整条 VLN 任务跑下来？这等于把 VLN 推理过程显式化、可观察、可控制。

## Contribution
1. 提出 NavGPT：第一个完全依赖现成 LLM、不引入任何可学习模块或交互式训练经验的 instruction-following VLN agent。
2. 系统性地探查当下 LLM 在 VLN 上的推理能力与瓶颈：可分解指令、识别 landmark、跟踪进度、应对异常并调整计划。
3. 揭示 LLM 具备根据导航历史生成高质量指令、画出 metric-level top-down 轨迹的能力，给“多模态 LLM 作为通用导航 agent”指出方向。

## Method

### VLN 形式化
策略 $\pi(a_t \mid \mathcal{W}, \mathcal{O}_t, \mathcal{O}_t^C, \mathcal{S}_t; \Theta)$，其中 $\mathcal{W}$ 是指令、$\mathcal{O}_t$ 是当前 viewpoint 的 $N$ 路全景观测、$\mathcal{O}_t^C$ 是 $M$ 个候选可导航视点（带相对角度 $a_i^C$）、$\mathcal{S}_t$ 是历史状态。NavGPT 直接把 $\Theta$ 替换成 LLM 的语料先验，不在 VLN 数据上学。

### Pipeline
```
instruction W + observation O_t
  ├─ Visual Foundation Models F: BLIP-2 (per-view caption) + Faster-RCNN (objects) + MP3D depth (3m filter)
  └─ navigable viewpoints with relative heading
        │
        ▼
Prompt Manager M
  ├─ System Principle P (rules: only emit existing viewpoint IDs, output Thought+Action format)
  ├─ formatted observation (8 directions, clockwise; FoV 45°, 24 views)
  ├─ history H_<t+1 = [<O,R,A>_1, ..., <O,R,A>_t] (older steps summarized by GPT-3.5)
  └─ navigation system principles
        │
        ▼
  Large Language Model (GPT-4)
        │
        ▼
   ⟨R_{t+1}, A_{t+1}⟩  ← ReAct: Thought 推理 + make_action("viewpointID") 或 stop
```

公式上：
$$\langle \mathcal{R}_{t+1}, \mathcal{A}_{t+1}\rangle = \text{LLM}\bigl(\mathcal{M}(\mathcal{P}), \mathcal{M}(\mathcal{W}), \mathcal{M}(\mathcal{F}(\mathcal{O}_t)), \mathcal{M}(\mathcal{H}_{<t+1})\bigr)$$

### 视觉感知细节
- 一个 viewpoint 切成 8 方向 × 3 elevation = 24 个 ego view（FoV=45°，heading 步长 45°，elevation 取 +30°/0°/-30°，邻接 view 重叠 15°）。
- BLIP-2 (ViT-G/FlanT5-XL) 给每个方向生成场景描述；Faster-RCNN 提对象框，结合 MP3D 深度过滤掉 3m 以外的对象，给 LLM 看“朝向 + 对象 + 距离 + 候选 viewpointID”。
- 用 GPT-3.5 把 top/middle/down 三层视图 caption 合并成一句话给该方向用。

### 历史与扩展动作空间
- 把 action space 扩展为 $\bar{\mathcal{A}}=\mathcal{A}\cup\mathcal{R}$：reasoning trace $\mathcal{R}$ 作为 ReAct 内部“无副作用动作”，会写回 history buffer。
- 当 history 太长时，prompt manager 调 GPT-3.5 摘要老 step 的观察，新 step 保持原文。

## Key Results

### R2R val unseen 主表（vs. supervised baseline）
| Schema | Method | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---|---:|---:|---:|---:|---:|
| Train Only | Seq2Seq | 8.39 | 7.81 | 28 | 21 | – |
| Train Only | Speaker Follower | – | 6.62 | 45 | 35 | – |
| Train Only | EnvDrop | 10.70 | 5.22 | – | 52 | 48 |
| Pretrain+FT | PREVALENT | 10.19 | 4.71 | – | 58 | 53 |
| Pretrain+FT | VLN⟳BERT | 12.01 | 3.93 | 69 | 63 | 57 |
| Pretrain+FT | HAMT | 11.46 | 2.29 | 73 | 66 | 61 |
| Pretrain+FT | DuET | 13.94 | 3.31 | 81 | 72 | 60 |
| **No Train** | DuET (Init. LXMERT) | 22.03 | 9.74 | 7 | 1 | 0 |
| **No Train** | **NavGPT (GPT-4)** | **11.45** | **6.46** | **42** | **34** | **29** |

要点：
- 完全 zero-shot 的 NavGPT 已经能压住 Seq2Seq 这类 train-only baseline，并接近 Speaker Follower。
- 与 HAMT/DuET 这类 supervised SOTA 仍有 30+ SR 的差距，作者把瓶颈归结为 (a) 视觉→语言翻译的信息损失、(b) history 摘要导致的 object tracking 退化。

### 视觉粒度消融（216 样本子集，GPT-3.5 baseline）
| Granularity | # views | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---:|---:|---:|---:|---:|---:|
| FoV@60 | 1 image | 12.38 | 9.07 | 14.35 | 10.19 | 6.52 |
| FoV@30 | 12 (heading only) | 12.67 | 8.92 | 15.28 | 13.89 | 9.12 |
| **FoV@45** | **24 (3 elev × 8 head)** | **12.18** | **8.02** | **26.39** | **16.67** | **13.00** |

→ FoV@45 + 24 views 的描述粒度最适合 LLM；过大 FoV 让描述变成“房间级泛化”，过小 FoV 又难辨物体。

### 视觉信息消融（同 216 样本）
| Observation | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---:|---:|---:|---:|---:|
| Baseline (caption only) | 16.11 | 9.83 | 15.28 | 11.11 | 6.92 |
| + Object | 12.07 | 8.88 | 23.34 | 15.97 | 11.71 |
| + Object + Distance | 12.18 | 8.02 | 26.39 | 16.67 | 13.00 |

→ 加 object 信息 SR +4.86，再加 depth 又抬 +0.7，证明“显著对象 + 与 agent 距离”是 LLM 决策的关键 signal。

### 高层规划/历史能力的定性证据
- 文中 Fig. 3：NavGPT 能做 sub-goal 拆解（“先穿过厨房，再上楼”）、landmark 跟踪（“picture frames on the wall”）、commonsense 补救（“sink not visible, but a bookcase 是 plausible 中间点”）、异常处理（错过 viewpoint 后 re-plan）。
- 文中 Fig. 4：仅给 history（不含 reasoning trace），GPT-4 能从历史 action+observation 自动重写一条新指令；并能 pyplot 出大致正确的 top-down 轨迹，说明历史里隐藏了 metric-level 空间感。

## Limitation
作者自己点出：
1. **信息损失**：BLIP-2 caption 与 RCNN 对象列表只是 ego view 的稀疏摘要，目标对象一旦没出现在 caption 里 LLM 就只能 explore；
2. **History 长度爆炸**：steps 增加后必须用 GPT-3.5 summarizer 压缩，老 step 的 object 跟踪能力急剧下降；
3. **与 supervised 模型差距明显**：SR 34 vs DuET 72；尤其是 OSR/SR 比例提示 LLM 经常“看到目标却不停”或“不知道距离已足够近就停”。

潜在 gap：
- 没有跨数据集结果（REVERIE / RxR / CVDN / SOON），所以“zero-shot 通用性”更多停留在 R2R-style 指令上。
- VFM 选用的是 BLIP-2 ViT-G/FlanT5-XL + Faster-RCNN，并未消融更强 MLLM 是否能直接替掉这条管线。
- Prompt 用法和具体 token 消耗、推理延迟没有量化（一条 R2R 轨迹要调用一次 GPT-4 + 多次 GPT-3.5 + 多次 BLIP-2，工程成本不低）。

## 与本项目的关系
本仓库的主轴是 SAME（一个 supervised + MoE 的 VLN 模型）和 NavNuances 这类细粒度评测；NavGPT 是“**纯 LLM、零训练**”的另一极，对当前实验的价值在三个层面：

1. **零样本下界 baseline**。在 R2R val unseen 上 NavGPT(GPT-4) 给出 SR 34 / SPL 29 / NE 6.46，可以作为讨论 SAME / DuET-style supervised 模型时一个明确的“纯 prompting 上限”参照点；同表里的 DuET(Init. LXMERT, no train) SR=1 也是常被引用的“没有训练就完全做不动”反例。
2. **NavGPT-2 的前身**。本项目 references 里已有 `2024-ECCV-NavGPT-2.md`：NavGPT-2 改成 frozen LLM + Q-former + DUET-style graph policy，正是为了解决 NavGPT 这两条核心瓶颈（vision→language 信息损失、history 难以追踪）。在阅读 NavGPT-2 时把它当作“NavGPT 的修复版”比单独读更易看到设计动机。
3. **NavNuances 评估的可行性参照**。memory 里记过 *NavNuances 的 R2R\_\* 是独立任务、需要重新 rollout 才能接评估*。NavGPT 这类 LLM-only agent 在 NavNuances 上理论上比 SAME 更好接（因为没有学习过 R2R 训练分布、不需要重新训练），但同时 NavGPT 的视觉 grounding 较粗，预计在 NavNuances 中需要细粒度物体/方向辨识的子任务上表现更弱——这是值得在后续实验里验证的对比角度。

不直接当 baseline 时，这篇论文最大的可复用价值是它对 *VLN-as-prompting* 的失败模式做了清晰诊断（caption 损失 + history 摘要损失），这两条对任何想在 SAME 之外接入 LLM 推理模块的工作都是必须先回答的问题。
