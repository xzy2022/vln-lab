# 阶段4.6 endpoint preference objective 修正

状态：main ablation 已完成，dev 未过 continue line，已审核

本阶段目标是根据 4.5 诊断结论修正 ranker objective：

```text
不再继续优化 candidate-level CE success；
改为优化 episode group 内 endpoint preference，
重点修复 success-success SPL 排序、late/final-like 偏置和 final-success harm。
```

当前阶段结论：

```text
phase4_6 主 ablation 已完成；
SPL-aware pairwise / group-listwise 能改善 success-success preference，
但未转化成足够的 endpoint recovery，所有配置均未通过 continue line。
当前不进入 4.7，不运行 frozen val_unseen，不继续跑 actionable gate 主线。
```

## 1. 新增实现入口

### 1.1 preference ranker 单配置训练

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1

python scripts/analysis/train_endpoint_preference_ranker.py \
  --experiment-dir "$EXP" \
  --output-dir "$EXP/endpoint_learning/preference_ablation/runs/pairwise_spl2_final4" \
  --objective pairwise \
  --pair-weights success_gt_fail=1,better_spl_success_gt_lower_spl_success=2,final_success_final_gt_failed_nonfinal=4 \
  --epochs 40 \
  --max-pairs-per-type 200000
```

输出同 4.3 evaluator 协议兼容的：

```text
preference_ranker_scores.csv
```

其中包含：

```text
candidate_id
candidate_score
gate_score
```

### 1.2 4.6 ablation runner

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1

python scripts/analysis/run_endpoint_preference_ablation.py \
  --experiment-dir "$EXP" \
  --preset phase4_6
```

runner 会依次完成：

```text
train preference ranker
run evaluator bridge
run weighted ALL frozen selection
write preference_ablation_table.csv
write preference_ablation_report.md
```

运行时会向 stderr 打印阶段进度，例如：

```text
Loading candidates
Building vectorized candidate features
Building pair arrays
epoch 4/40
Running evaluator bridge
Selecting frozen config
```

如果需要安静模式，可加：

```text
--quiet
```

默认输出目录为：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/preference_ablation/
```

本次主实验产物：

| 文件 | 内容 |
| --- | --- |
| `preference_ablation_table.csv` | 5 个 objective 配置的 frozen dev selection 汇总 |
| `preference_ablation_report.md` | continue line 与主表报告 |
| `runs/*/endpoint_preference_ranker_report.md` | 每个 preference ranker 的训练、pair agreement 和 evaluator bridge |
| `runs/*/frozen_selection/dev_selection_report.md` | 每个配置的 weighted ALL frozen selection 结果 |

当前 `phase4_6` preset 包含：

| run | objective | 说明 |
| --- | --- | --- |
| `pairwise_spl1_final2` | pairwise | 低 SPL/final 权重 |
| `pairwise_spl2_final4` | pairwise | 默认 P0 |
| `pairwise_spl4_final8` | pairwise | 强 SPL/final preserving |
| `pairwise_listwise_spl2_final4_soft` | pairwise + listwise | soft reward group target |
| `pairwise_listwise_spl2_final4_best_spl` | pairwise + listwise | best-SPL one-hot group target |

## 2. Objective 设计

### 2.1 SPL-aware pairwise

使用 4.1 已冻结的三类 pair：

```text
success_gt_fail
better_spl_success_gt_lower_spl_success
final_success_final_gt_failed_nonfinal
```

训练目标是 Bradley-Terry pairwise logistic：

```text
L_pair = softplus(-(score_chosen - score_rejected))
```

默认权重：

```text
success_gt_fail = 1
better_spl_success_gt_lower_spl_success = 2
final_success_final_gt_failed_nonfinal = 4
```

### 2.2 group/listwise

listwise 使用 episode 内 softmax CE：

```text
p_i = softmax(score_i within episode)
target_i = softmax(label_reward_i / T)
```

其中 label reward 只用于训练 loss，不作为 inference feature。

当前支持：

```text
soft_reward
best_spl_onehot
first_success_onehot
```

### 2.3 final-preserving

final-preserving 不新增 oracle feature，而是提高：

```text
final_success_final_gt_failed_nonfinal
```

的 pairwise 权重，直接约束原本 final 已成功时不要被失败 non-final 覆盖。

## 3. Actionable gate

4.6 也实现了 actionable gate 的拼接入口，但当前主 ablation 未过线，因此本分支暂缓，不作为进入 4.7 前的必要步骤。

暂缓原因：

```text
当前 P0 / P0+listwise ranker 本身 top1 recovery 很弱；
actionable label 会退化为“当前 ranker 已经能 top1 recover 的极少数样本”，
更可能进一步收紧 gate，而不是创造新的 recovery。
```

### 3.1 生成 actionable label

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1

python scripts/analysis/build_endpoint_actionable_gate_labels.py \
  --experiment-dir "$EXP" \
  --score-csv "$EXP/endpoint_learning/preference_ablation/runs/pairwise_spl2_final4/preference_ranker_scores.csv" \
  --output-dir "$EXP/endpoint_learning/preference_ablation/actionable_labels/pairwise_spl2_final4" \
  --criterion top1_success \
  --tau 0.0 \
  --splits train,dev
```

### 3.2 用 actionable label 训练 gate

`train_endpoint_gate_baseline.py` 已新增后向兼容参数：

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1

python scripts/analysis/train_endpoint_gate_baseline.py \
  --experiment-dir "$EXP" \
  --label-csv "$EXP/endpoint_learning/preference_ablation/actionable_labels/pairwise_spl2_final4/actionable_gate_labels.csv" \
  --label-column actionable_rerank \
  --output-dir "$EXP/endpoint_learning/preference_ablation/actionable_gate/pairwise_spl2_final4" \
  --skip-reranker-eval
```

默认不传 `--label-csv` 时，gate baseline 行为保持为原来的 `should_rerank` 训练。

### 3.3 合并 ranker score 与 actionable gate score

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1

python scripts/analysis/merge_endpoint_ranker_gate_scores.py \
  --ranker-score-csv "$EXP/endpoint_learning/preference_ablation/runs/pairwise_spl2_final4/preference_ranker_scores.csv" \
  --gate-score-csv "$EXP/endpoint_learning/preference_ablation/actionable_gate/pairwise_spl2_final4/gate_scores.csv" \
  --output-csv "$EXP/endpoint_learning/preference_ablation/actionable_gate/pairwise_spl2_final4/merged_ranker_gate_scores.csv"
```

合并后的 CSV 可直接交给：

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1

python scripts/analysis/select_endpoint_frozen_config.py \
  --experiment-dir "$EXP" \
  --score-csv "$EXP/endpoint_learning/preference_ablation/actionable_gate/pairwise_spl2_final4/merged_ranker_gate_scores.csv" \
  --output-dir "$EXP/endpoint_learning/preference_ablation/actionable_gate/pairwise_spl2_final4/frozen_selection"
```

## 4. 评测与继续线

4.6 主表不以 candidate AUC 为核心，而看：

```text
weighted delta_SR / delta_SPL
recovery / harm
max_dataset_harm
CVDN delta_SR / harm
better_spl_success_gt_lower_spl_success accuracy
final_success_final_gt_failed_nonfinal accuracy
```

进入 4.7 前，dev 至少满足：

```text
weighted delta_SR >= +0.20pp
weighted delta_SPL >= 0
harm_rate <= 1pp
max_dataset_harm_rate <= 1pp
CVDN delta_SR >= -0.10pp
recovered / harmed > 1
```

`run_endpoint_preference_ablation.py` 会在表中写出：

```text
passes_continue_line
```

但不自动跑 `val_unseen`。

### 4.1 本次 dev ablation 结果

本次 `phase4_6` preset 共完成 5 个配置，全部成功训练、评测并完成 frozen dev selection，但全部 `passes_continue_line=false`：

| run | dSR | dSPL | recovery | harm | max harm | CVDN dSR | better SPL acc | continue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pairwise_spl1_final2` | +0.02pp | +0.22pp | 0.02pp | 0.00pp | 0.00pp | +0.00pp | 41.90% | false |
| `pairwise_spl2_final4` | +0.06pp | +0.29pp | 0.10pp | 0.04pp | 0.05pp | +0.21pp | 44.49% | false |
| `pairwise_spl4_final8` | +0.06pp | +0.58pp | 0.26pp | 0.20pp | 0.21pp | +0.63pp | 47.10% | false |
| `pairwise_listwise_spl2_final4_soft` | +0.09pp | +0.30pp | 0.13pp | 0.05pp | 0.07pp | +0.32pp | 47.13% | false |
| `pairwise_listwise_spl2_final4_best_spl` | +0.12pp | +0.33pp | 0.16pp | 0.04pp | 0.05pp | +0.32pp | 48.97% | false |

最佳配置是：

```text
pairwise_listwise_spl2_final4_best_spl
gate_threshold = 0.9000
tau = 0.0000
delta_SR = +0.12pp
delta_SPL = +0.33pp
recovery = 0.16pp
harm = 0.04pp
```

它满足 `recovery > harm`、`delta_SPL >= 0`、`max_dataset_harm <= 1pp` 和 `CVDN delta_SR >= -0.10pp`，但没有达到：

```text
weighted delta_SR >= +0.20pp
```

因此不能进入 4.7。

### 4.2 结果分析

4.6 的主要现象是：

```text
preference objective 确实修复了一部分 success-success SPL preference，
但没有把成功候选稳定推到 episode group top1。
```

和 4.5 CE 诊断相比，`better_spl_success_gt_lower_spl_success` dev accuracy 从 `32.66%` 提升到最高 `48.97%`，说明 SPL-aware / listwise 目标方向有效。但 dev SR 收益最高只有 `+0.12pp`，仍低于继续线。

更关键的是，当前 preference ranker 的 top1 recovery 能力反而偏弱：

```text
CE baseline 在 should_rerank dev 上 top1 success: 209 / 976 = 21.41%
best 4.6 preference ranker top1 success:          55 / 976 = 5.64%
```

这说明当前 4.6 objective 很可能过度保留 final / late-like endpoint。gate 已经负责“何时不要改 final”，ranker 再通过 final-preserving pair 和 final-like feature 学一遍保守偏置，会导致应该恢复时也倾向不改或仍选 final-like endpoint。

因此，本阶段失败点不再是单纯的 `tau` 或 gate threshold 没调好，而是：

```text
ranker objective 与推理职责仍然没有完全解耦。
gate 应负责是否允许修改 final；
ranker 应负责在允许修改后选择更好的非 final endpoint。
当前 preference ranker 仍把 final-preserving safety 混入 endpoint selection。
```

### 4.3 下一步计划

当前不继续执行 actionable gate 主线，也不进入 4.7 / val_unseen。

如果仍停留在 4.6，只允许一个最小 sanity ablation：

```text
final-bias 解耦 ranker
```

目标不是扩大 grid，而是验证一个诊断假设：

```text
去掉或显著降低 ranker 侧的 final-preserving / final-like 偏置后，
should_rerank dev top1 success 是否能从 5.64% 回到接近 CE baseline 的区间。
```

建议只观察诊断指标，不把它直接作为 4.7 候选：

```text
should_rerank top1 success
top1_is_final_rate
better_spl_success_gt_lower_spl_success accuracy
weighted dev delta_SR / harm
```

若该 sanity ablation 仍不能恢复 top1 endpoint selection，则阶段 4 的结论应收束为：

```text
endpoint upper bound 明确存在；
但仅靠当前 SAME trace feature 的离线 gate + ranker，
无法稳定吃掉该 upper bound。
后续应换问题定义或引入新信号，而不是继续在 4.6 内扫 objective。
```

## 5. 运行环境

```bash
bash scripts/setup/run_container.sh
conda activate endpoint-v1
```

本阶段不需要 `test-v1`，除非要重新生成 SAME 主实验 trace。

## 6. 已完成验证

已在容器 `endpoint-v1` 中完成：

```text
python -m py_compile
--help smoke
full 0017 CSV 上的 one-config smoke ablation
full 0017 CSV 上的 phase4_6 main ablation
```

smoke 仅验证训练、score 导出、evaluator bridge、frozen selection 和 ablation table 链路，不作为实验结果。`phase4_6 main ablation` 是本阶段当前决策依据。
