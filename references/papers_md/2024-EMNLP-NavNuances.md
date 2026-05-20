# Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation

**Authors**: Zehao Wang et al. (KU Leuven, Peking University, NTU, Fudan University)

**Venue**: Findings of EMNLP, 2024

**Tags**: `VLN` `Evaluation` `Benchmark` `Dataset` `Fine-grained` `CFG`

---

## TL;DR

现有 VLN 评价过度依赖 R2R 的终点成功率和路径对齐指标，导致模型能力被高估。本文提出 **NavNuances** 数据集和评估框架，通过上下文无关文法（CFG）将 VLN 指令分解为 5 个原子类别（方向变换、垂直移动、数值理解、地标识别、区域识别），在 90 个 Matterport 场景中系统诊断各类模型的细粒度能力缺陷。

## Problem & Motivation

- 当前 VLN 评价指标（SR、SPL、nDTW）过于粗粒度，SOTA 在 R2R 上已接近人类水平（Wang et al., 2023），但随机 agent 的 SR 也不可忽视
- 简单的指令干预（如将 "turn left" 替换为 "turn right"）并未在模型中引发一致的响应变化，说明模型并未真正理解原子概念
- LMM 增强 agent（如 GPT-4V）在标准 VLN 数据集上表现反而不如传统方法，但在其他多模态任务中表现强劲——这种反差说明评价方式本身有问题

## Contribution

1. **基于 CFG 的系统化评价框架**：借助 LLM 半自动构建 VLN 指令的上下文无关文法，从语法层面分解指令并定义原子能力类别
2. **NavNuances 数据集**：覆盖 5 个类别、1787 个实例，含人工精炼和 LLM 润色，在 90 个 Matterport 场景中构建
3. **全面基准测试与分析**：评测了 9 种基线模型（传统监督 + 零样本 LMM），揭示了多个关键发现（详见 Key Results）

## Method

### 核心流程

```
R2R/RxR 指令 → CFG 迭代构建（LLM 辅助解析 + 人工修正）
                    ↓
      CFG 定义 S → Vp → {ActionT, ActionS, ActionO+Landmark, ActionR+Region, ...}
                    ↓
     归纳 5 个原子类别 → 各类别路径提议策略
                    ↓
     CFG 驱动指令生成 → LLM 润色 → 人工精炼
                    ↓
              NavNuances 数据集（1787 条）
```

### CFG（上下文无关文法）

CFG 定义为四元组 $G = (N, T, P, S)$，其中：

- **非终结符**（大写）：`S`, `Vp`, `ActionT`, `ActionS`, `ActionO`, `ActionR`, `Landmark`, `Region`, `Modifier`, `Numerical`, `Direction`, `Room`, `Object`, `Attribute`
- **终结符**（小写）：`left`, `right`, `bed`, `table`, `chair`, `red`, `yellow`, …
- **产生式规则示例**：
  ```
  S → Vp
  Vp → ActionT | ActionS | ActionO + Landmark | ActionR + Region | Vp + Vp | Vp + Ir
  ActionT → "turn" + Direction | "turn around"
  ActionO → "walk towards" | "walk past" | "walk past from" + Direction
  ActionR → "go into" | "exit" | "walk through"
  ActionS → "go upstairs" | "go downstairs"
  Landmark → Modifier + Object
  Region → Modifier + Room
  ```

CFG 通过 GPT-4 解析标准数据集指令 + 人工修正，迭代约 10 轮直至无遗漏。完整 CFG 见论文 Appendix E。

### 五个原子类别

| 类别 | 来源 | 实例数 | 说明 |
|------|------|--------|------|
| Direction Change (DC) | `ActionT` | 579 | turn left/right/around，独立于观察 |
| Vertical Movement (VM) | `ActionS` | 170 | go upstairs/downstairs |
| Numerical Comprehension (NU) | `Numerical` | 78 | 沿走廊进入第 i 个房间 |
| Region Recognition (RR) | `ActionR + Region` | 275 | go into / exit 房间 |
| Landmark Recognition (LR) | `ActionO + Landmark` | 685 | walk towards / past 特定物体 |

### 数据构建

每个类别有专用的路径提议策略（基于 Habitat 语义标注 + 人工验证）：
- **DC**：选分支角度 > 45° 的路口，排除因导航图稀疏造成的假阳性
- **VM**：在 3D 楼梯标注框内取最长路径，人工调整起终点
- **NU**：筛选有足够多门的走廊，标注房间序号和左右侧
- **LR**：用 GPT-4V 生成实例级物体描述，人工确认可见性
- **RR**：记录区域内所有点作为正确响应，而非单一终点

## Key Results

### 5 类别主结果（Success Rate %）

| 模型 | DC | VM | LR | RR | NU | R2R SR |
|------|-----|-----|-----|-----|-----|--------|
| Random | 36.79 | 7.69 | 30.22 | 57.45 | 11.76 | 15.88 |
| Seq2Seq | 75.30 | 21.79 | 22.04 | 53.09 | 25.88 | 21.46 |
| CLIP-ViL | 77.20 | 29.49 | 36.78 | 74.18 | 69.41 | 52.15 |
| VLN-BERT | 72.02 | 29.49 | 36.05 | 80.36 | 75.29 | 62.75 |
| HAMT | 79.62 | 28.21 | 36.05 | 77.81 | 68.82 | 63.22 |
| DUET | 64.76 | 26.92 | 35.76 | 77.45 | 76.47 | 71.52 |
| BEVBERT | 63.21 | 24.35 | 30.22 | 80.36 | 84.12 | 75.18 |
| ScaleVLN | 72.88 | 26.92 | 29.92 | 84.73 | 84.71 | 80.97 |
| NavGPT3.5 (0-shot) | 81.87 | 20.51 | 58.54 | 39.63 | 7.06 | 40.82 |
| NavGPT4 (0-shot) | 91.87 | 34.78 | 54.83 | 67.61 | 11.36 | 47.53 |
| NavGPT4v (0-shot) | 92.68 | 39.13 | 62.87 | 56.25 | 13.64 | 54.78 |
| **Human** | **95.83** | **89.13** | **89.44** | **89.89** | **94.42** | **—** |

### 关键发现

1. **R2R 进步主要来自 VM 和 RR**：CLIP-ViL 相比 Seq2Seq 在 R2R SR 上提升 30.69%，对应 VM 从 25.88 → 69.41（+43.53）、RR 从 53.09 → 74.18（+21.09）。说明模型主要提升了空间布局理解，而非语言理解
2. **数值理解全面停滞**：NU 是所有模型的共同短板，即使是最强的 ScaleVLN（84.71 on VM）在 NU 上也仅 26.92。LMM 方法略有提升但仍远低于人类（94.42）
3. **零样本 LMM 在 DC 和 LR 上反超监督模型**：NavGPT4v 在 DC 达 92.68（超过所有监督方法），LR 达 62.87——说明大规模预训练的知识可以弥补小规模监督数据的局限
4. **监督模型存在选择偏差**：对 left/right 方向有偏好（如 ScaleVLN 右转比左转高 18.23%），Dual SR 指标远低于单方向准确率
5. **LMM 在 RR 和 VM 上显著落后**：NavGPT4v RR 仅 56.25 vs. ScaleVLN 84.73——LMM 缺乏精确的边界判断能力，往往在观察到目标区域时就过早停止

### 深入分析

- **LR 子集分析**：walk towards 远优于 walk past，后者需要理解基于连续观察的空间关系，LMM 在此出现"概念的 inconsistent 理解"
- **NU 消融**：引入两个随机 agent 控制布局理解和方向感知后，部分监督模型的性能与 Agent 1*（仅理解走廊布局）相当，说明其"数值理解"实际来自布局先验而非真正的计数能力

## Limitation

- **环境限制**：Matterport3D 是静态离散环境，无法编辑物体属性或位置，限制了数据多样性（如无法生成物体级数值理解："第 i 个苹果"）
- **仅覆盖原子能力**：不评估长指令下的多动作序列执行和纠错能力，这部分也是 VLN 的重要方面
- **半自动 CFG 构建**：虽然比纯手工高效，但在更复杂任务中仍需大量人工修正，扩展到法律/金融等专业领域可能有困难

## 与本论文的关系

- **项目已有 NavNuances 数据集**：`third_party/navnuances/` 已包含数据集和评估代码
- **SAME 模型与 NavNuances**：根据 memory 记录（navnuances-r2r-relationship.md），NavNuances 是 R2R-style 独立任务，SAME 需要重新 rollout 才能对接评估。本项目的 SAME 适配工作（patches 体系）可以帮助将 SAME 接入 NavNuances 评估框架
- **意义**：本文的诊断结果为 SAME 在本项目中的改进提供了方向——如果 SAME 在 NU、walk past 等关键维度表现不佳，可以有针对性地改进
- **与 NavGPT-2 的关系**：本项目同时跟进 NavGPT-2（参见 `third_party/NavGPT-2/`），而本文评测了 NavGPT（第一代），可以作为 NavGPT-2 的 baseline 参考。NavGPT4v（本文提出，GPT-4V 增强版）是 NavGPT 系列中首个引入直接视觉-指令对齐的变体，值得 NavGPT-2 参考
