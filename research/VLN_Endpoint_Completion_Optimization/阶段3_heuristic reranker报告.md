# 阶段 3：heuristic reranker 报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md`，只记录阶段 3 的作用、运行入口、实现注意事项和当前实验结果。

当前状态：

```text
reviewed：人工已读并认可口径
```

实验数据来源：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

## 1. 本阶段作用

阶段 3 对应总路线中的第三个问题：

```text
不训练模型、只靠 SAME trace 规则能否稳定 recover？
```

阶段 1 证明了存在 oracle gap：`val_unseen` 上有 200 条 pass-but-not-stop / overshoot 样本。阶段 2 证明了如果 oracle-style 从已访问 endpoint 中选最近点，`val_unseen` SR 可以从 76.29 提升到 84.80。

但阶段 1/2 都使用了 GT distance 或 success label 做离线分析。真实方法不能在推理时读取这些信息。因此阶段 3 必须回答一个中间问题：

```text
不用 GT distance，只看 SAME 自己的 trajectory / STOP trace / loop 信号，
能不能从已访问 endpoint 中选出更好的终点？
```

这就是阶段 3 的核心价值。它不是最终方法，而是从 oracle upper bound 走向 learned endpoint learning 之前的必要诊断桥。

### 1.1 为什么不能跳过阶段 3

如果直接从阶段 2 跳到阶段 4，会缺少三类证据：

1. 不知道 SAME trace 里是否真的有 endpoint correction 信号。
2. 不知道 rule-only 方法能吃掉多少 oracle gap，learned 方法缺少 baseline。
3. 不知道 endpoint rerank 的主要风险是不是 harm 原本成功的 final endpoint。

阶段 3 的帮助是：

```text
建立 no-training / no-GT baseline
验证 STOP / loop / revisit 等 trace 特征是否有信号
量化 recovery 与 harm 的权衡
判断是否需要 learned gate 来决定“何时不要动 final”
为阶段 4 的 candidate features、gate label 和 final-stay 约束提供依据
```

因此，阶段 3 即使没有带来正收益，也仍然有研究价值：它可以证明“规则有信号但不稳定”，从而合理推出阶段 4 的 learned gate + ranker。

## 2. 代码运行入口

通用入口：

```bash
python scripts/analysis/build_endpoint_heuristic_report.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

以当前实验数据来源为例：

```bash
python scripts/analysis/build_endpoint_heuristic_report.py \
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

默认输出目录：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/endpoint_heuristic_rerank/
```

输出文件：

```text
manifest.json
endpoint_heuristic_items.csv
endpoint_heuristic_summary.csv
endpoint_heuristic_report.md
```

当前 manifest 信息：

```text
schema_version = endpoint_heuristic_report.v1
eval_item_sources = 2
heuristics = 36
item_scope_heuristic_rows = 277200
summary_rows = 144
generated_at_utc = 2026-04-28T12:44:03.893724+00:00
```

## 3. 代码实现注意事项

### 3.1 与阶段 1/2 的关系

阶段 1/2 使用：

```text
scripts/analysis/build_oracle_gap_report.py
```

阶段 3 使用：

```text
scripts/analysis/build_endpoint_heuristic_report.py
```

阶段 3 的脚本会复用阶段 1/2 中的 target scope、success rule 和 fine metrics 对齐逻辑，但 endpoint 选择规则不能读取 GT distance 或 success label。

### 3.2 可用信息边界

heuristic 选择 endpoint 时只能使用 SAME 自身输出：

```text
trajectory
decision_trace.steps[*].stop_prob
decision_trace.steps[*].selected.kind
decision_trace.steps[*].selected.prob
stop margin / move margin
router entropy
route backtrack / loop / revisit
step position
```

不能用于选择：

```text
distance_to_nav_goal_by_step_m
final_success
oracle_success
nearest_endpoint_success
success_target_viewpoints
nav_goal_viewpoint
gt_path
```

GT distance 和 success label 只允许在 endpoint 选择完成之后用于评分。

### 3.3 decision_trace 要求

阶段 3 依赖 `decision_trace`。当前 R2R 原型实验 `0014_same_val_r2r_eval_only_same_s0_v5` 的 `eval_items` 包含 decision trace，适合做本阶段。

没有 `decision_trace` 的旧实验可以做阶段 1/2，但不适合做阶段 3。

### 3.4 已实现 heuristic family

脚本当前覆盖：

| heuristic | 含义 |
| --- | --- |
| `final` | 原始 SAME final endpoint |
| `max_stop_prob` | 选择全轨迹 STOP 概率最高的 step |
| `max_stop_margin` | 选择 STOP 相对 move margin 最大的 step |
| `last_k_max_stop` | 只在最后 K 个 decision step 中选 STOP 概率最高点 |
| `last_k_max_stop_margin` | last-k 的 stop margin 版本 |
| `first_stop_threshold` | 第一次 STOP 概率超过阈值就停 |
| `last_stop_threshold` | 最后一次 STOP 概率超过阈值的位置 |
| `last_high_stop_before_move` | 模型曾高 STOP 但继续移动时，回退到该类 step |
| `loop_guard` | 出现 loop / revisit 附近时，回退到窗口内 STOP 较高点 |
| `conservative_rerank` | 组合 last-k、high-stop-before-move、loop guard 的保守规则 |

默认阈值：

```text
thresholds = 0.1,0.2,0.3,0.4,0.5,0.6,0.7
last_k_values = 3,5,7
loop_window = 10
```

### 3.5 关键指标

本阶段不能只看 SR，还必须同时看 recovery 和 harm：

```text
recovery_rate = final failed but heuristic endpoint succeeds
harm_rate = final succeeded but heuristic endpoint fails
net_recovery_rate = recovery_rate - harm_rate
gap_capture_rate = (heuristic SR - final SR) / (nearest SR - final SR)
changed_endpoint_rate = heuristic endpoint != final endpoint
```

其中 `harm_rate` 是阶段 3 最重要的风险指标。只要 heuristic 会频繁修改原本成功的 final endpoint，即使它能 recover overshoot，也可能不是可用方法。

## 4. 当前实验结果

以下结果使用：

```text
experiment = 0014_same_val_r2r_eval_only_same_s0_v5
dataset = R2R
target_scope = official
```

### 4.1 与阶段 1/2 的基线衔接

阶段 1/2 给出的 `val_unseen` 背景是：

```text
final SR = 76.29
nearest endpoint SR = 84.80
oracle / nearest gap = 8.51pp
可恢复 overshoot = 200 条
```

阶段 3 的目标不是达到 nearest upper bound，而是测试不用 GT distance 时，规则能否稳定恢复其中一部分，同时不伤害原本成功的 final endpoint。

### 4.2 val_train_seen 结果

| heuristic | params | heur SR | delta SR | recovery | harm | changed | heur SPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `final` |  | 88.74 | +0.00 | 0.00 | 0.00 | 0.00 | 85.01 |
| `first_stop_threshold` | `threshold=0.2` | 89.47 | +0.73 | 2.13 | 1.40 | 21.39 | 87.12 |
| `last_high_stop_before_move` | `threshold=0.2` | 89.47 | +0.73 | 1.73 | 1.00 | 18.12 | 86.86 |
| `loop_guard` | `threshold=0.1,window=10` | 89.27 | +0.53 | 0.87 | 0.33 | 4.20 | 86.05 |
| `loop_guard` | `threshold=0.2,window=10` | 89.07 | +0.33 | 0.47 | 0.13 | 2.80 | 85.67 |

`val_train_seen` 上，简单规则确实能取得正收益。例子：

```text
first_stop_threshold threshold=0.2:
  recover = 32
  harm = 21
  changed = 321

loop_guard threshold=0.2:
  recover = 7
  harm = 2
  changed = 42
```

这说明 SAME trace 里有 endpoint correction 信号，不是纯噪声。

### 4.3 val_unseen 结果

| heuristic | params | heur SR | delta SR | recovery | harm | changed | heur SPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `final` |  | 76.29 | +0.00 | 0.00 | 0.00 | 0.00 | 66.19 |
| `loop_guard` | `threshold=0.2,window=10` | 76.20 | -0.09 | 0.64 | 0.72 | 3.87 | 66.78 |
| `loop_guard` | `threshold=0.1,window=10` | 76.03 | -0.26 | 1.06 | 1.32 | 5.92 | 67.11 |
| `last_high_stop_before_move` | `threshold=0.3` | 75.73 | -0.55 | 0.81 | 1.36 | 10.60 | 66.82 |
| `last_high_stop_before_move` | `threshold=0.2` | 75.05 | -1.23 | 2.00 | 3.24 | 21.80 | 67.15 |

更具体地看 `val_unseen`：

```text
loop_guard threshold=0.2:
  recover = 15
  harm = 17
  changed = 91

last_high_stop_before_move threshold=0.2:
  recover = 47
  harm = 76
  changed = 512
```

这说明规则确实能 recover 一部分 overshoot，但 harm 通常抵消甚至超过 recovery。

### 4.4 关键观察

观察 1：规则有信号。

```text
last_high_stop_before_move threshold=0.2 在 val_unseen recover 47 / 200 个 overshoot。
loop_guard threshold=0.2 在 val_unseen recover 15 / 200 个 overshoot。
```

如果 trace 完全没有信号，这些 recovery 不会出现。

观察 2：固定规则泛化不稳。

```text
val_train_seen 上多个规则 delta SR 为正。
val_unseen 上最保守的 loop_guard 也略微负收益。
```

这说明简单阈值规则容易对 split 分布敏感。

观察 3：harm 是主要瓶颈。

```text
last_high_stop_before_move threshold=0.2:
  val_unseen recover 47
  val_unseen harm 76
```

真正困难的不是“找不到任何可恢复 endpoint”，而是“何时不要动 final”。

观察 4：SPL 不能单独作为成功证据。

部分 heuristic 虽然 SR 下降，但 SPL 上升，例如 `val_unseen` 的 `last_high_stop_before_move threshold=0.2`：

```text
SR: 76.29 -> 75.05
SPL: 66.24 -> 67.15
```

这可能是因为规则倾向选择更早 endpoint，路径变短，但同时伤害了一些原本成功样本。因此阶段 3 必须优先看 SR / recovery / harm，而不是只看 SPL。

### 4.5 本阶段结论

阶段 3 的结论是：

```text
SAME trace 中确实存在 endpoint correction 信号；
rule-only heuristic 能 recover 一部分 overshoot；
但固定规则在 val_unseen 上 harm 不可控，整体 SR 没有稳定提升。
```

因此阶段 3 对研究的帮助是明确的：

```text
它证明阶段 4 不应只是追求更复杂的 loss，
而必须学习一个 conservative gate 来判断“何时不要动 final”，
再学习 endpoint ranker 来判断“如果要改，改到哪个 candidate”。
```

本阶段支持继续进入阶段 4：

```text
构造离线 endpoint learning 闭环，
学习 gate + ranker，
并用 frozen threshold / tau 在 val_unseen 上一次性报告。
```

本文档本轮更新状态：

```text
2026-04-29：补全阶段 3 heuristic reranker 报告草案，等待人工审核。
2026-04-29：已审核.
```
