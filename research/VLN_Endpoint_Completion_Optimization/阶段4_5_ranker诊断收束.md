# 阶段4.5 ranker 诊断收束

状态：draft，待人工审核

本阶段只做诊断，不训练新模型，不使用 `val_unseen` 调参。

## 1. 目标

阶段 4.5 要回答的问题是：

```text
CE success ranker 为什么 candidate-level AUC 很高，
但在 episode group 内做 top1 endpoint selection 时仍然失败？
```

本阶段的产物是 ranker diagnostics report，用来给 4.6 的 preference objective 修正提供证据。

## 2. 实现入口

新增脚本：

```bash
scripts/analysis/diagnose_endpoint_ranker_top1.py
```

本次运行命令：

```bash
conda run -n plots python scripts/analysis/diagnose_endpoint_ranker_top1.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1 \
  --splits dev \
  --output-dir endpoint_ranker_diagnostics/0017_phase4_5_dev
```

诊断产物写入：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_ranker_diagnostics/0017_phase4_5_dev
```

主要输出：

| 文件 | 内容 |
| --- | --- |
| `ranker_diagnostics_report.md` | 汇总报告 |
| `episode_rank_diagnostics.csv` | 每个 episode 的 target rank / top1 / margin |
| `target_rank_summary.csv` | nearest / first-success / best-SPL / final 等目标的 rank 统计 |
| `top1_failure_summary.csv` | `should_rerank` episode 中 top1 失败 slice |
| `top1_by_group_size.csv` | candidate group 大小与 top1 成功率 |
| `top1_failure_feature_deltas.csv` | top1 失败点相对 oracle target 的 step/path/distance/score 差异 |
| `pair_agreement_by_type.csv` | preference pair 方向一致性 |
| `top1_failure_samples.csv` | 高 margin top1 失败样本 |

## 3. 诊断协议

输入沿用 4.1-4.4 的 frozen artifacts：

```text
candidate_csv = endpoint_learning/candidate_groups/endpoint_candidates.csv
score_csv = endpoint_learning/ranker_baseline/ranker_scores.csv
pair_csv = endpoint_learning/preference_pairs/preference_pairs.csv
frozen selected items = endpoint_learning/frozen_gate_ranker/dev_selected_items.csv
```

诊断 split：

```text
target_scope = official
split = dev
```

核心 target 定义：

| target | 含义 |
| --- | --- |
| `best_scored_success` | ranker 分数最高的成功候选，即 success candidate 的最好 rank |
| `nearest` | GT distance 最近的已访问 endpoint |
| `first_success` | 第一处成功 endpoint |
| `best_spl_success` | 成功候选中 SPL 最高的 endpoint |
| `final` | SAME 原始 final endpoint |

## 4. 关键结果

在 dev 的 976 个 `should_rerank` episode 中：

| 指标 | 数值 |
| --- | ---: |
| ranker top1 success | 209 / 976 = 21.41% |
| ranker top1 failed | 767 / 976 = 78.59% |
| failed top1 over best-success mean margin | 0.2483 |
| high-confidence failed top1, margin > 0.2 | 48.24% |

这说明 CE ranker 不是完全没有把成功候选排上来，而是在 group argmax 时经常让失败点压过成功点。换句话说：

```text
candidate-level success classification signal 存在；
episode-level top1 preference 仍然不可靠。
```

### 4.1 成功候选 rank

| target | top1 | top3 | top5 | median rank |
| --- | ---: | ---: | ---: | ---: |
| best scored success | 21.41% | 75.31% | 86.17% | 2 |
| nearest | 10.45% | 59.22% | 76.84% | 3 |
| first success | 4.92% | 38.22% | 60.76% | 4 |
| best SPL success | 6.66% | 49.28% | 68.14% | 4 |
| final | 44.36% | 73.98% | 81.05% | 2 |

最重要的诊断点是：

```text
best scored success 的 median rank = 2，
但 best SPL success / first success 的 median rank = 4。
```

这说明 CE ranker 往往能把某个成功候选排到前几名，但没有学会“哪个成功候选更适合作为 endpoint”。这正是 4.3 中 `better_spl_success_gt_lower_spl_success` 只有 32.66% 的 group-level 版本。

### 4.2 success-success preference 方向明显错位

dev pair agreement：

| pair type | pairs | accuracy | mean margin |
| --- | ---: | ---: | ---: |
| success > fail | 155213 | 85.46% | +0.4593 |
| better SPL success > lower SPL success | 15412 | 32.66% | -0.1102 |
| final success > failed non-final | 44239 | 94.43% | +0.6921 |

这组结果把问题定位得比较准：

```text
CE ranker 很会分 success vs fail；
也很会保护 final success；
但在成功候选之间，它倾向把更差 SPL 的成功点排得更高。
```

因此，继续优化 candidate binary CE 或继续扫 `tau`，很难从根上解决 group top1。

### 4.3 top1 失败点通常更晚、更长、更远

在 767 个 `should_rerank && top1 failed` episode 中，失败 top1 相比 `best_spl_success`：

| 差异 | mean | median | top1 greater rate |
| --- | ---: | ---: | ---: |
| candidate score | +0.3618 | +0.3259 | 100.00% |
| candidate step | +3.5437 | +2.0000 | 96.35% |
| path length | +7.5612 m | +5.4490 m | 96.35% |
| distance to goal | +5.1468 m | +3.6434 m | 100.00% |

同时，top1 失败 slice 中：

| 指标 | 数值 |
| --- | ---: |
| failed top1 is final | 56.45% |
| failed after last success | 88.01% |
| failed top1 is last-k | 85.53% |

这说明当前 CE ranker 有明显的 late / final-like 偏置：

```text
它把“越靠后、越像模型最终停点”的候选打高，
即使这些点已经越过成功 endpoint，距离目标更远。
```

这也解释了为什么 tau 能压 harm，但会挡掉 recovery：ranker 分数本身没有给早期成功 endpoint 足够 margin。

### 4.4 数据集差异

| dataset | should_rerank | top1 success | top1 failed | high-confidence failed | median best-SPL rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2R | 84 | 14.29% | 72 | 44.44% | 3.5 |
| REVERIE | 12 | 16.67% | 10 | 0.00% | 2 |
| SOON | 666 | 25.23% | 498 | 40.96% | 3 |
| CVDN | 214 | 12.62% | 187 | 71.66% | 6 |

CVDN 是最危险的 slice：top1 success 最低，高 confidence 失败最高，best-SPL target 的 median rank 也最差。这和 4.4 frozen config 中 CVDN dev `delta_SR = -0.21pp`、`harm = 0.63pp` 对得上。

## 5. 结论

4.5 的结论是：

```text
CE success ranker 的主要失败原因不是“没有 success/fail 信号”，
而是训练目标与推理目标错位。
```

更具体地说：

1. CE ranker 学到了 candidate-level `success_label`，但推理需要 episode-level argmax。
2. CE ranker 能较好地区分 success > fail，但没有学会 success-success 之间的 SPL / early-success preference。
3. 失败 top1 明显偏向更晚、更长、final-like 的 endpoint，导致 overshoot correction 反而继续 overshoot。
4. 目前的 score margin calibration 不足，tau 只能作为防 harm 补丁，不能创造 recovery。
5. CVDN 是最需要止损的 dataset slice，后续配置不能继续牺牲 CVDN 换取 weighted ALL 小幅收益。

因此，4.6 不应继续做 CE threshold / tau 微调，而应改 objective。

## 6. 对 4.6 的实现指向

4.6 建议按这个顺序推进：

| 优先级 | objective / module | 目的 |
| --- | --- | --- |
| P0 | SPL-aware pairwise ranker | 直接修复 `better_spl_success_gt_lower_spl_success` 只有 32.66% 的问题 |
| P0 | group/listwise endpoint objective | 让训练目标对齐 episode group top1，而不是 candidate binary AUC |
| P1 | final-preserving contrastive loss | final 成功时强约束不要被失败 non-final 覆盖 |
| P1 | actionable gate | gate 预测“当前 ranker 是否能安全恢复”，而不是 oracle `should_rerank` |
| P2 | candidate top-k / late-bias pruning ablation | 控制长轨迹 false positive 被 argmax 放大的风险 |

4.6 的最低观察指标不应再只看 candidate AUC，而应至少报告：

```text
recoverable should_rerank top1 success
best_spl_success top1 / top3 / median rank
better_spl_success_gt_lower_spl_success pair accuracy
weighted ALL dev delta_SR / delta_SPL
max_dataset_harm
CVDN delta_SR / harm
```
