# 阶段 2：nearest endpoint upper bound 报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md`，只记录阶段 2 的作用、运行入口、实现注意事项和当前实验结果。

当前状态：

```text
reviewed：人工已读并认可口径
```

实验数据来源：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

## 1. 本阶段作用

阶段 2 对应总路线中的第二个问题：

```text
如果能在 SAME 已访问过的 endpoint 中选对终点，理论空间有多大？
```

阶段 1 已经回答“是否存在 final STOP / endpoint 可恢复失败”。阶段 2 在此基础上进一步估计无训练上界：

```text
不生成新轨迹
不重训 SAME
不使用 learned model
只在 SAME 已访问过的 trajectory step 中，oracle-style 选择离目标最近的 endpoint
```

对于总路线来说，阶段 2 的作用是判断 endpoint correction 是否值得继续投入：

```text
如果 nearest endpoint upper bound 提升很小，说明 endpoint rerank 空间有限。
如果 nearest endpoint upper bound 明显高于 final endpoint，说明“只改终点、不改路径”也有研究价值。
如果 upper bound 主要来自阶段 1 的 overshoot 样本，阶段 3/4 就应重点学习何时回退到已访问成功点。
```

本阶段不是一个真实可部署方法。它使用 GT distance 选择 endpoint，只能作为 oracle-style upper bound。

## 2. 代码运行入口

阶段 2 和阶段 1 使用同一个代码入口：

```bash
python scripts/analysis/build_oracle_gap_report.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

以当前实验数据来源为例：

```bash
python scripts/analysis/build_oracle_gap_report.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

默认输出目录也和阶段 1 相同：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/oracle_gap_for_rl_research/
```

输出文件：

```text
manifest.json
oracle_gap_items.csv
oracle_gap_summary.csv
oracle_gap_report.md
```

阶段 1 主要读取这些字段：

```text
final_success
oracle_success
oracle_gap_rate
overshoot
stop_too_early_proxy
never_reached
```

阶段 2 主要读取这些字段：

```text
nearest_endpoint_success
nearest_endpoint_spl
recovered_by_nearest_endpoint
best_distance_step
best_endpoint_path_length_m
best_distance_m
final_minus_best_distance_m
```

因此可以理解为：

```text
代码层面：阶段 1 和阶段 2 已经由同一个脚本同时产出。
研究层面：阶段 1 和阶段 2 回答不同问题，仍然分开写报告。
```


## 3. 代码实现注意事项

### 3.1 nearest endpoint 的定义

脚本对每条 trajectory 计算：

```text
best_distance_step = 第一次达到最小目标距离的 step
nearest_endpoint_success = best_distance_step 是否满足 success rule
nearest_endpoint_spl = 在 best_distance_step 停止时的 SPL
recovered_by_nearest_endpoint = final_success == false AND nearest_endpoint_success == true
```

当前 0017 official scope 使用：

```text
R2R / SOON / CVDN official:
  success_mode = distance_threshold
  distance_key = distance_to_nav_goal_by_step_m
  success_threshold_m = 3.0

REVERIE official:
  success_mode = exact_viewpoint
  distance_key = distance_to_nearest_success_target_by_step_m
  targets = success_target_viewpoints
```

也就是 R2R / SOON / CVDN 按距离目标小于 3m 视为成功；REVERIE official 按是否进入成功 viewpoint 集合视为成功。

### 3.2 这是 oracle-style 上界

nearest endpoint selection 使用 target scope 对应的 GT distance，例如 `distance_to_nav_goal_by_step_m` 或 `distance_to_nearest_success_target_by_step_m`。它不能作为真实推理特征：

```text
不能用于 heuristic reranker 的选择规则
不能用于 gate / ranker 的 inference feature
不能把 nearest endpoint 结果当成 learned method
```

本阶段只回答：

```text
如果 endpoint selector 理想地知道哪个已访问点离目标最近，最多能恢复多少？
```

### 3.3 与 oracle success 的关系

在当前 0017 official scope 下：

```text
oracle_success = trajectory 中至少有一个 step 满足 official success rule
nearest_endpoint_success = 全 trajectory 最近 target / region 的 step 满足 official success rule
```

因此当前 0017 official scope 下，`nearest_endpoint_success_rate` 与 `oracle_success_rate` 相同。

但二者的 SPL 含义不同：

```text
first_success_oracle_spl：在第一次成功 step 停止
nearest_endpoint_spl：在最近目标 step 停止
```

最近目标 step 可能晚于第一次成功 step，所以 nearest endpoint 的 SR 是上界，但 nearest SPL 不一定等于最高 SPL 上界。

### 3.4 与阶段 1 的边界

阶段 1 关注 failure 是否存在以及 failure bucket：

```text
oracle gap
overshoot
stop_too_early_proxy
never_reached
```

阶段 2 关注“只改 endpoint”的理论收益：

```text
nearest SR - final SR
nearest SPL - final SPL
recovered_by_nearest_endpoint
final_distance_m - best_distance_m
```

两个阶段共享脚本和输出目录，但不共享结论段落。

## 4. 当前实验结果

以下结果使用：

```text
experiment = 0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
datasets = R2R, REVERIE, SOON, CVDN
target_scope = official
```

### 4.1 汇总表

| dataset | split | items | final success | final SR | nearest success | nearest SR | delta SR | recovered by nearest | final SPL | nearest SPL | delta SPL | final dist | best dist | dist improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | train_eval | 14039 | 12263 | 87.35 | 12903 | 91.91 | +4.56pp | 640 / 4.56% | 82.86 | 89.02 | +6.16pp | 1.42m | 0.77m | 0.65m |
| R2R | val_seen | 1021 | 819 | 80.22 | 884 | 86.58 | +6.37pp | 65 / 6.37% | 74.36 | 82.56 | +8.20pp | 2.11m | 1.23m | 0.88m |
| R2R | val_train_seen | 1501 | 1332 | 88.74 | 1416 | 94.34 | +5.60pp | 84 / 5.60% | 85.01 | 92.00 | +6.99pp | 1.28m | 0.62m | 0.65m |
| R2R | val_unseen | 2349 | 1792 | 76.29 | 1992 | 84.80 | +8.51pp | 200 / 8.51% | 66.24 | 75.81 | +9.56pp | 2.72m | 1.40m | 1.32m |
| REVERIE | train_eval | 10466 | 8532 | 81.52 | 9389 | 89.71 | +8.19pp | 857 / 8.19% | 76.97 | 87.33 | +10.36pp | 0.93m | 0.43m | 0.51m |
| REVERIE | val_seen | 1423 | 823 | 57.84 | 935 | 65.71 | +7.87pp | 112 / 7.87% | 52.54 | 60.90 | +8.36pp | 4.38m | 2.46m | 1.92m |
| REVERIE | val_train_seen | 123 | 103 | 83.74 | 115 | 93.50 | +9.76pp | 12 / 9.76% | 79.20 | 91.12 | +11.92pp | 0.52m | 0.19m | 0.33m |
| REVERIE | val_unseen | 3521 | 1614 | 45.84 | 1924 | 54.64 | +8.80pp | 310 / 8.80% | 35.85 | 43.73 | +7.89pp | 4.93m | 2.36m | 2.57m |
| SOON | train_eval | 27800 | 19843 | 71.38 | 23311 | 83.85 | +12.47pp | 3468 / 12.47% | 64.77 | 77.87 | +13.10pp | 3.15m | 1.55m | 1.60m |
| SOON | val_seen | 1130 | 579 | 51.24 | 696 | 61.59 | +10.35pp | 117 / 10.35% | 40.38 | 49.96 | +9.58pp | 7.87m | 4.00m | 3.87m |
| SOON | val_unseen | 3390 | 1232 | 36.34 | 1831 | 54.01 | +17.67pp | 599 / 17.67% | 25.66 | 39.78 | +14.12pp | 8.13m | 4.52m | 3.62m |
| CVDN | train_eval | 4742 | 2161 | 45.57 | 3191 | 67.29 | +21.72pp | 1030 / 21.72% | 40.02 | 63.49 | +23.47pp | 8.11m | 3.57m | 4.53m |
| CVDN | val_seen | 382 | 108 | 28.27 | 180 | 47.12 | +18.85pp | 72 / 18.85% | 25.01 | 44.05 | +19.04pp | 12.72m | 6.48m | 6.24m |
| CVDN | val_unseen | 907 | 218 | 24.04 | 481 | 53.03 | +29.00pp | 263 / 29.00% | 16.98 | 43.96 | +26.98pp | 12.94m | 6.36m | 6.58m |

### 4.2 val_unseen 上界分析

如果允许 oracle-style 选择 SAME 已访问过的最近 endpoint，`val_unseen` 上界如下：

| dataset | items | final success | nearest success | recovered | SR | SPL | final dist | best dist | dist improvement |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| R2R | 2349 | 1792 | 1992 | 200 / 8.51% | 76.29 -> 84.80 (+8.51pp) | 66.24 -> 75.81 (+9.56pp) | 2.72m | 1.40m | 1.32m |
| REVERIE | 3521 | 1614 | 1924 | 310 / 8.80% | 45.84 -> 54.64 (+8.80pp) | 35.85 -> 43.73 (+7.89pp) | 4.93m | 2.36m | 2.57m |
| SOON | 3390 | 1232 | 1831 | 599 / 17.67% | 36.34 -> 54.01 (+17.67pp) | 25.66 -> 39.78 (+14.12pp) | 8.13m | 4.52m | 3.62m |
| CVDN | 907 | 218 | 481 | 263 / 29.00% | 24.04 -> 53.03 (+29.00pp) | 16.98 -> 43.96 (+26.98pp) | 12.94m | 6.36m | 6.58m |

这说明当前 SAME 的一部分失败不是“轨迹完全没有到过目标附近”，而是：

```text
轨迹访问过更好的 endpoint，但 final endpoint 没有停在那里。
```

### 4.3 与阶段 1 的衔接

阶段 1 中 `val_unseen` 的 oracle gap 与阶段 2 中 `val_unseen` 的 nearest recovery 一致：

| dataset | oracle gap | nearest recovery |
| --- | ---: | ---: |
| R2R | 200 / 2349 = 8.51pp | 200 / 2349 = 8.51% |
| REVERIE | 310 / 3521 = 8.80pp | 310 / 3521 = 8.80% |
| SOON | 599 / 3390 = 17.67pp | 599 / 3390 = 17.67% |
| CVDN | 263 / 907 = 29.00pp | 263 / 907 = 29.00% |

在当前 0017 official scope 下，两者相同，原因是：

```text
只要 trajectory 曾经进入成功区域，trajectory 的最近 target / region 点也一定成功。
```

所以阶段 2 的结论可以直接承接阶段 1：

```text
阶段 1 证明 0017 official scope 下存在 pass-but-not-stop 可恢复样本；
阶段 2 证明如果能 oracle-style 从已访问 endpoint 中选最近点，这些样本可以全部恢复到成功口径。
```

### 4.4 SPL 解释

`val_unseen` 的 first-success oracle-stop SPL 与 nearest endpoint SPL 如下：

| dataset | first-success oracle SPL | nearest SPL | nearest - first-success |
| --- | ---: | ---: | ---: |
| R2R | 79.27 | 75.81 | -3.46pp |
| REVERIE | 43.73 | 43.73 | +0.00pp |
| SOON | 42.67 | 39.78 | -2.89pp |
| CVDN | 47.41 | 43.96 | -3.45pp |

这不是矛盾。原因是：

```text
first_success_oracle_spl 假设一进入成功区域就停；
nearest_endpoint_spl 假设走到全轨迹最近目标点再停；
最近点可能更晚，路径更长，因此 SPL 可能低于 first-success oracle-stop SPL。
```

这对后续方法有一个提醒：

```text
如果只追求 SR，nearest endpoint 是合理 upper bound。
如果同时追求 SPL，ranker 不一定应该总是选距离最近点，而应学习 success + path efficiency 的折中。
```

### 4.5 本阶段结论

阶段 2 的结论是：

```text
0017 official scope 的 val_unseen 上，只在 SAME 已访问 endpoint 中 oracle-style 选最近点：
R2R：SR 76.29 -> 84.80，SPL 66.24 -> 75.81。
REVERIE：SR 45.84 -> 54.64，SPL 35.85 -> 43.73。
SOON：SR 36.34 -> 54.01，SPL 25.66 -> 39.78。
CVDN：SR 24.04 -> 53.03，SPL 16.98 -> 43.96。
```

这说明：

```text
不重新规划路径，只修正 final endpoint，也存在明显理论空间。
```

本阶段支持继续进入阶段 3：

```text
测试不使用 GT distance、只用 SAME 自身 trace 特征的 rule-only endpoint reranker，
能否实际恢复一部分 upper bound。
```

但本阶段还不能证明真实方法有效。nearest endpoint 使用 GT distance，只能作为 upper bound；真实方法必须在阶段 3 / 阶段 4 中验证 recovery 与 harm 的权衡。

本文档本轮更新状态：

```text
2026-04-29：补全阶段 2 nearest endpoint upper bound 报告草案，等待人工审核。
2026-04-29: 已人工审核
2026-04-29：R2R val 切换到 R2R / REVERIE / SOON / CVDN 数据集的全部split
```
