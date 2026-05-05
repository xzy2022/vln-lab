---
name: ref-to-md
description: 将 references/papers/ 目录下的 PDF 论文转换为结构化 Markdown 格式，供后续论文写作参考。
disable-model-invocation: true
---

# ref-to-md

## 输入

1. 用户指定 PDF 文件路径，或直接指定 `references/papers/` 目录（批量处理）。
2. 当前的研究情况总览,默认为`research/VLN research`

## 处理流程

### 1. 读取 PDF

使用 Read 工具读取 PDF 文件，获取完整文本内容。

### 2. 提取核心信息

对每篇论文，提取以下内容：

**文件头元数据**
```
# {论文标题}
**Authors**: {第一作者 et al.}
**Venue**: {会议/期刊}, {年份}
**Tags**: [领域标签，按需添加]
```

**TL;DR** — 用一句话说清楚这篇论文在做什么、解决了什么问题。

**Problem & Motivation** — 问题的意义和困难在哪里，现有方案哪里不够。

**Contribution** — 作者声称的贡献（不超过 3 点）。

**Method** — 用流程图式的语言描述核心设计，不需要完整公式推导，保留关键方程的 LaTeX 形式。Pipeline 用 ASCII/文字箭头描述。

**Key Results** — 最核心的实验数字，用表格呈现（setting / metric / value）。标注最重要的发现和与 baseline 的对比。

**Limitation** — 作者自己承认的不足，或你发现的 gap。

### 3. 写作风格

- 保留 LaTeX 数学公式（AI 写稿时能直接引用）
- 关键数字精确到小数点
- 与本项目（参见研究情况总览）强相关的论文，在末尾加一段 **"与本论文的关系"** 分析（方法有何异同、是否可以作为 baseline、gap 在哪里）

### 4. 输出路径

每个 MD 文件写入 `references/md/{pdf文件名去掉扩展符}.md`，扩展名前缀 `@` 改为年份前缀：

```
@22-CVPR-DUET.pdf  →  references/papers_md/2022-CVPR-DUET.md
@24-AAAI-NavGPT.pdf →  references/papers_md/2024-AAAI-NavGPT.md
```

### 5. 批量处理

如果用户指定目录，按文件修改时间顺序处理。每处理完一篇报告进度。

## 注意事项

- 不要逐字转录原文，要做信息提炼
- 如果 PDF 文本质量差或缺失某些字段，在输出中标注 `（未从原文提取，标注来源）`
- 方法描述要能让一个不了解这篇论文的读者在 5 分钟内理解核心思想