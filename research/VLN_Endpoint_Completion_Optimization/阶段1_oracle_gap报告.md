# 阶段 1：oracle gap 报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md`，只记录阶段 1 的作用、运行入口、实现注意事项和当前实验结果。

该报告承接`research/VLN_Endpoint_Completion_Optimization/研究总路线.md`
当前状态：

实验数据来源
```text
reviewed：人工已读并认可口径
```

实验数据来源：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

需要给出:
1. 本阶段的具体作用,对于research/VLN_Endpoint_Completion_Optimization/研究总路线.md
2. 代码运行入口,以及以实验数据来源为示例的具体代码,输出的路径
3. 代码实现的注意事项
4. 当前实验结果的分析
## 1. 本阶段作用

阶段 1 对应总路线中的第一个问题：

```text
SAME 是否存在 final STOP / endpoint 可恢复失败？
```

它的作用不是提出 rerank 方法，也不是训练模型，而是先把 SAME 的 final endpoint 错误拆清楚：

```text
final_success：原始 SAME final endpoint 是否成功
oracle_success：轨迹中是否曾经进入成功区域
oracle_gap：oracle_success - final_success
overshoot：曾经成功，但最终没有停在成功区域
stop_too_early_proxy：从未成功，且 final 已是离目标最近点
never_reached：从未成功，且 final 也不是最近点
```

对于总路线来说，阶段 1 给出一个 go / no-go 判断：

```text
如果 oracle_gap 很小，endpoint correction 没有研究空间。
如果 oracle_gap 明显存在，继续做阶段 2 的无训练 endpoint 上界。
如果 gap 主要来自 overshoot / pass-but-not-stop，阶段 4 的 learned endpoint gate + ranker 才有明确目标。
```

本阶段只回答“有没有可恢复 failure，以及它们属于哪类”。`nearest_endpoint_success` 和 `nearest_endpoint_spl` 虽然由同一脚本产出，但属于阶段 2 的上界分析，不作为阶段 1 的核心结论。

## 2. 代码运行入口

通用入口：

```bash
python scripts/analysis/build_oracle_gap_report.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

以当前实验数据来源为例：

```bash
python scripts/analysis/build_oracle_gap_report.py \
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

默认输出目录：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/oracle_gap_for_rl_research/
```

输出文件：

```text
manifest.json
oracle_gap_items.csv
oracle_gap_summary.csv
oracle_gap_report.md
```



## 3. 代码实现注意事项

### 3.1 数据对齐

脚本通过 `eval_items/*_eval_context.json` 自动发现数据源，再读取对应的 `eval_items.jsonl`。每条 eval item 通过下面的键和 `fine_metrics_wide.csv` 对齐：

```text
dataset
split
internal_item_id
```

如果 fine metrics 中缺少对应行，脚本会直接报错，而不是静默跳过。

### 3.2 target scope

默认请求的 scope 是：

```text
official, goal, region, region_threshold
```

但当前数据是 R2R，因此实际只生成 `official` 和 `goal` 两类结果；`region` / `region_threshold` 对 R2R 不适用，会被跳过。

R2R 的 `official` scope 使用：

```text
success_mode = distance_threshold
distance_key = distance_to_nav_goal_by_step_m
success_threshold_m = 3.0
```

也就是距离目标小于 3m 视为成功。

### 3.3 step 定义

脚本基于 SAME 输出的完整 trajectory：

```text
final_step = len(trajectory) - 1
first_success_step = 第一次满足 success rule 的 step
best_distance_step = 第一次达到最小目标距离的 step
best_distance_step_last = 最后一次达到同一最小目标距离的 step
```

`best_distance_step` 使用“第一次最近点”，不是最后一次最近点。脚本同时保存 `best_distance_step_last`，方便后续检查 loop / revisit 情况。

### 3.4 failure bucket

本阶段的关键分类是：

```text
overshoot = final_success == false AND oracle_success == true AND first_success_step < final_step
stop_too_early_proxy = final_success == false AND oracle_success == false AND final_is_best_distance
never_reached = final_success == false AND oracle_success == false AND not stop_too_early_proxy
```

其中 `stop_too_early_proxy` 只是 proxy，不等价于真实“应该继续走就会成功”。它只表示：轨迹从未进入成功区域，并且 final endpoint 已经是这条轨迹里离目标最近的位置。

### 3.5 GT 信息边界

本阶段是离线诊断报告，可以读取 GT distance 和 success label 来做分析。它不能被误用为训练或推理特征：

```text
distance_to_nav_goal_by_step_m
success_target_viewpoints
final_success
oracle_success
```

这些字段只能用于离线评估、label 构造或 oracle-style 上界，不允许作为 rule-only heuristic、gate 或 ranker 的推理输入。

### 3.6 与阶段 2 的边界

`build_oracle_gap_report.py` 同时输出：

```text
nearest_endpoint_success
nearest_endpoint_spl
recovered_by_nearest_endpoint
```

这些字段服务于阶段 2 的“无训练 endpoint 上界”。阶段 1 只使用它们帮助确认脚本完整性，不把 nearest endpoint 上界作为本阶段结论展开。

## 4. 当前实验结果

以下结果使用：

```text
experiment = 0014_same_val_r2r_eval_only_same_s0_v5
dataset = R2R
target_scope = official
```

### 4.1 汇总表

| split | items | final success | final SR | oracle success | oracle SR | oracle gap | overshoot | stop-too-early proxy | never reached | final SPL | oracle-stop SPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val_train_seen | 1501 | 1332 | 88.74 | 1416 | 94.34 | 84 / 5.60pp | 84 / 5.60% | 18 / 1.20% | 67 / 4.46% | 85.01 | 93.37 |
| val_unseen | 2349 | 1792 | 76.29 | 1992 | 84.80 | 200 / 8.51pp | 200 / 8.51% | 79 / 3.36% | 278 / 11.83% | 66.24 | 79.27 |

### 4.2 val_unseen 失败拆解

`val_unseen` 一共有 2349 条 episode：

```text
final_success = 1792
final_fail = 557
oracle_success = 1992
oracle_gap = 200
```

557 个 final failure 可以拆成：

```text
overshoot / pass-but-not-stop = 200
stop_too_early_proxy = 79
never_reached = 278
```

其中 `oracle_gap = overshoot = 200`，说明当前 R2R official 口径下，所有可由“重新选择已访问 endpoint”恢复的失败，都表现为 pass-but-not-stop：

```text
轨迹曾经进入成功区域，但 SAME 最终没有停在成功区域。
```

这正好对应总路线里 endpoint completion 的核心目标。

### 4.3 seen / unseen 差异

`val_train_seen` 的 oracle gap 是 5.60pp，`val_unseen` 的 oracle gap 是 8.51pp。unseen 上的 gap 更大，说明 endpoint / STOP 错误不是只在 seen split 出现的偶然现象。

同时，`val_unseen` 的 final SPL 是 66.24，而 oracle-stop SPL 是 79.27。这说明如果能在第一次成功位置停住，不只是 SR 有空间，路径效率指标也有明显空间。

### 4.4 本阶段结论

阶段 1 的结论是：

```text
R2R val_unseen 上存在明确 oracle gap：8.51pp，200 条 episode。
这些可恢复 gap 在当前口径下全部属于 overshoot / pass-but-not-stop。
因此 endpoint / STOP correction 有继续研究的必要。
```

本阶段支持继续进入阶段 2：

```text
估计“如果从 SAME 已访问 endpoint 中选对终点”，理论上界有多大。
```

但本阶段还不能证明任何真实方法有效。它只说明存在可恢复空间；真实方法是否能低 harm 地恢复，需要阶段 3 的 rule-only heuristic 和阶段 4 的离线 endpoint learning 闭环继续验证。

本文档本轮更新状态：

```text
2026-04-29：补全阶段 1 oracle gap 报告草案，等待人工审核。
2026-04-29：已人工审核.
```