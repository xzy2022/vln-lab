# VLN-MME: Diagnosing MLLMs as Language-guided Visual Navigation agents

**Authors**: Xunyi Zhao, Gengze Zhou, Qi Wu
**Venue**: arXiv:2512.24851 (cs.CV, cs.RO)
**Submitted**: 2025/12/31 | **Revised**: 2026/01/06
**Tags**: [VLN], [MLLM Evaluation], [Embodied AI], [Zero-shot Navigation]

---

## TL;DR

提出 VLN-MME，一个统一模块化的**无模拟器评估框架**，用于诊断 MLLM 作为零样本视觉语言导航智能体的能力。核心发现：**CoT 和自反思策略反而会降低性能**，揭示了 MLLM 在具身任务中**上下文感知能力差**、**3D 空间推理保真度低**的根本缺陷。

---

## Problem & Motivation

现有 VLN 评估面临三重困境：

1. **计算成本高**：高保真模拟器（Habitat、Matterport3D）在多轮交互中计算开销巨大
2. **评估碎片化**：不同数据集格式各异，难以系统比较
3. **诊断不足**：现有工作只报告端到端指标，缺乏细粒度错误分析

更关键的是，NavBench 等统一评估框架**只评估固定智能体设计**，无法解耦 MLLM 本身能力与智能体工程设计的贡献。

---

## Contribution

1. **模块化评估框架**：Model-Agent-Task 三组件分离，支持"即插即用"
2. **无模拟器评估设计**：预渲染图像替代实时渲染，显存降低 6x，访问速度提升 9x
3. **超越成功率的分析**：细粒度错误分类（循环、指令偏离、感知-行动脱节）
4. **开源标准化数据**：发布预处理数据集和环境工件

---

## Method

### Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Runner                                │
│  (配置管理、生命周期协调、指标计算)                          │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
    ┌────┴────┐         ┌────┴─────┐         ┌───┴────┐
    │  Model  │◄───────►│  Agent   │◄───────►│  Task  │
    │ (MLLM)  │  API调用  │ (决策模块) │  动作反馈  │ (数据集) │
    └─────────┘         └──────────┘         └────────┘
```

**Model**：统一接口，封装不同 MLLM 的 API 调用

**Task**：封装导航挑战和管理数据集切分

**Agent**：
- **Text Summarization Memory**（NavGPT 风格）：用自然语言描述历史动作和观察
- **Text Map Memory**（MapGPT 风格）：构建拓扑文本地图
- **推理策略变体**：Baseline / CoT / Reflection / CoT + Reflection

### Simulator-Free Design

核心思想：**空间换时间** —— 用预渲染图像替代实时渲染

| 指标 | 本方案 | Habitat | 提升 |
|------|--------|---------|------|
| VRAM 使用 | ~1.7 GB | ~10 GB | **5.9x 降低** |
| 观察访问 | ~0.016s | ~0.14s | **8.8x 加速** |
| 每步耗时 | $t$ | $t+1.5$s | **1.5s 更快** |
| 每回合耗时 | $T$ | $T+25$s | **~25s 更快** |

视觉表示：4 个不重叠的 90° FOV 图像（而非变形严重的等距矩形投影），与标准视觉编码器的预训练分布更对齐。

### Dataset Construction

从三个数据集的 val_unseen 切分构建：
- **Fine-Grained Navigation**：R2R（细粒度指令跟随）
- **Coarse-grained Navigation**：REVERIE（粗粒度高层指令 + 目标检索）
- **Object-Oriented Navigation**：ObjectNav（目标导航）

采用**分层采样**，沿三个轴保持多样性：
- 场景复杂度（按 Matterport3D scan ID 分层）
- 路径难度（按路径长度分桶）
- 语言丰富度（随机选择 3 条指令之一）

验证：benchmark 结果与全量 val_unseen 性能高度相关（偏差 < 2-3%）。

---

## Key Results

### 主实验结果（部分关键数字）

**Fine-Grained Navigation (R2R)**：

| Agent / MLLM | SR ↑ | SPL ↑ | OSR ↑ |
|--------------|------|-------|-------|
| **Gemini-2.5 Pro** (MapGPT) | **39.50** | **30.72** | **57.50** |
| GPT-5 (MapGPT) | 34.00 | 25.83 | 52.50 |
| claude2.5-VL-7B (NavGPT) | 27.50 | 17.11 | 44.00 |
| Qwen2.5-VL-7B w/ CoT | 21.00 ↓ | 11.41 ↓ | 37.50 ↓ |

**Object-Oriented Navigation (ObjectNav)**：

| Agent / MLLM | SR ↑ | SPL ↑ |
|--------------|------|-------|
| GPT-5 (MapGPT) | 44.00 | 20.91 |
| Gemini-2.5 Pro (MapGPT) | **49.50** | **26.24** |
| claude2.5-VL-7B (NavGPT) | 37.50 | 13.18 |

### 关键发现

1. **闭源模型领先**：GPT-5 和 Gemini-2.5 Pro 设立上界；开源中 claude2.5-VL-7B 最强（Fine-grained 上 27.5% SR vs LLaVA-OV 11.5%）

2. **CoT/Reflection 有害**：反直觉地，CoT 策略**降低**性能
   - claude2.5-VL-7B + CoT：SR 从 27.5% 降至 21.0%
   - 原因：模型缺乏**具身上下文感知**

3. **任务难度层级**：
   - 最简单：ObjectNav（SR 可达 ~50%）
   - 中等：Fine-grained（SR ~40%）
   - 最难：Coarse-grained（SR ~33%）

### 诊断实验

**Oracle 辅助（Hard Negatives）**：

| Method | SR ↑ | OSR ↑ | SPL ↑ |
|--------|------|-------|-------|
| Baseline (claude2.5VL-7B) | 0.00 | 0.00 | 0.00 |
| + Oracle (claude3VL-4B) | **52.00** | **68.00** | **41.28** |

Oracle 仅提供高层推理指导就带来巨大提升，说明基础模型**具备导航能力**，缺的是**战略规划**。

**Failure-Aware Few-Shot**：

| Method | SR ↑ |
|--------|------|
| Zero-shot | 0.00 |
| 1-shot Failure Example | 12.00 |
| 3-shot Failure Examples | 16.00 |

---

## Limitation

1. 计算资源限制，只覆盖代表性 MLLM 和智能体设计（框架本身可扩展）
2. 当前只评估标准指令跟随，未涵盖对话导航和多语言指令
3. 主要是**诊断性工作**：指出问题，未提出具体算法解决方案

---

## 与本项目的关系

### 方法对比

| 维度 | VLN-MME | SAME |
|------|---------|------|
| 目标 | MLLM 零样本评估 | VLN 专用模型训练 |
| 模拟器 | **Simulator-free** | 需要 Matterport3D 模拟器 |
| 数据 | 采样子集 | 全量 R2R/REVERIE/SOON |
| 指标 | 8 指标综合分析 | SR/SPL/nDTW 等 |

### 可作为 Baseline

- VLN-MME 的**零样本 MLLM 结果**（特别是 claude2.5-VL-7B 的 ~27.5% SR on R2R）可作为 SAME 的对比基线
- VLN-MME 的**诊断框架**（错误分类方法）可借鉴用于 SAME 的分析

### Gap

- VLN-MME 未涉及 CVDN（对话式导航）和 SOON（场景对象目标导航）
- 两者都基于 Matterport3D 环境，但 SAME 需要专门的微调训练
- **可探索方向**：用 SAME 微调后的模型 vs VLN-MME 零样本基线的差距分析

---

## References

- NavGPT (Zhou et al., 2024) - Text Summarization Memory Agent
- MapGPT (Chen et al., 2024) - Text Map Memory Agent
- R2R (Anderson et al., 2018)
- REVERIE (Qi et al., 2020)
- SAME (Zhou et al., 2025)
