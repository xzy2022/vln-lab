# 阶段 4.3：endpoint ranker baseline 报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md`，只记录阶段 4.3 的实现入口、实验产物和当前 dev 结果。

当前状态：

```text
reviewed：人工已读并认可口径
```

实验数据来源：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

## 1. 本阶段作用

阶段 4.3 对应总路线中的问题：

```text
如果 gate 放行，能否在候选 endpoint 中选更好点？
```

本阶段训练 candidate-level endpoint scorer / ranker：

```text
输入：4.1 candidate group 中允许用于推理的 SAME trace / trajectory 派生特征
输出：candidate_score = P(success_label)
label：success_label
gate_score：默认复用 4.2 gate_baseline/gate_scores.csv
```

本阶段先使用轻量 CE baseline，不引入 pairwise / DPO / GRPO；这些 loss 对比留给 4.5 ablation。

## 2. 代码运行入口

新增入口：

```bash
python scripts/analysis/train_endpoint_ranker_baseline.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

以当前实验数据来源为例：

```bash
conda activate endpoint-v1
python scripts/analysis/train_endpoint_ranker_baseline.py \
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

如果从宿主机调用当前 docker 容器：

```bash
docker exec -w /workspace/vln-lab vln-same-cu128 bash -lc \
  'conda run -n endpoint-v1 python scripts/analysis/train_endpoint_ranker_baseline.py \
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5'
```

默认输出目录：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/endpoint_learning/ranker_baseline/
```

关键输出：

```text
manifest.json
ranker_features.csv
ranker_scores.csv
ranker_candidate_summary.csv
ranker_pair_summary.csv
ranker_feature_importance.csv
ranker_model.joblib
ranker_model.json
endpoint_ranker_report.md
eval_protocol/endpoint_learning_summary.csv
eval_protocol/tau_curve.csv
eval_protocol/recovery_harm_curve.csv
```

其中 `ranker_scores.csv` 是给统一 evaluator 使用的 score CSV，包含：

```text
candidate_id
candidate_score
gate_score
```

## 3. 实现口径

当前 baseline：

```text
sklearn LogisticRegression
loss = candidate success cross entropy
class_weight = balanced
episode-balanced sample weight = true
SimpleImputer(strategy = median)
StandardScaler
C = 1.0
max_iter = 2000
random_state = 17
```

默认训练 / 选择协议：

```text
train：protocol_split == train
dev：protocol_split == dev
val_unseen / test：不用于训练，不用于选 tau 或 threshold
```

默认 evaluator grid：

```text
gate_threshold = 0.0, 0.5, 0.6, 0.7
tau = 0, 0.02, 0.05, 0.1, 0.2
allow_change_final = true
```

其中：

```text
0.0 = all-pass ranker diagnostic
0.6 = 4.2 low_pass candidate
0.7 = 4.2 conservative candidate
```

## 4. 当前实验结果

以下结果使用：

```text
dataset = R2R
target_scope = official
train candidates = 8156
dev candidates = 1985
features = 32
```

### 4.1 candidate-level 指标

| split | candidates | episodes | positive rate | AUC | AP | brier | log loss | top1 success | top1 harm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8156 | 1204 | 34.66 | 0.9460 | 0.8903 | 0.0912 | 0.2987 | 89.70 | 0.83 |
| dev | 1985 | 297 | 33.45 | 0.9444 | 0.8863 | 0.0999 | 0.3155 | 87.88 | 0.00 |

这说明 candidate-level trace features 对 `success_label` 有明显可学习信号。

### 4.2 preference pair 一致率

| split | pair type | pairs | accuracy | mean margin |
| --- | --- | ---: | ---: | ---: |
| train | all | 19649 | 83.87 | 0.5377 |
| dev | all | 3916 | 89.04 | 0.6322 |
| dev | success_gt_fail | 2669 | 92.02 | 0.6442 |
| dev | final_success_final_gt_failed_nonfinal | 995 | 98.89 | 0.8119 |
| dev | better_spl_success_gt_lower_spl_success | 252 | 18.65 | -0.2049 |

当前 CE ranker 很会区分 success / fail，也很会保留 final-success 样本中的成功 final；但它没有学好“成功 candidate 之间按 SPL 排序”。这符合预期，因为 4.3 baseline 只优化 `success_label`，没有优化 SPL preference。

### 4.3 evaluator bridge

dev final baseline：

```text
final SR = 87.54
final SPL = 84.58
oracle / nearest endpoint SR = 92.93
overshoot items = 16
```

all-pass ranker diagnostic：

| gate | tau | SR | delta SR | SPL | delta SPL | recovery | harm | changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 87.88 | +0.34 | 85.33 | +0.76 | 0.34 | 0.00 | 17.85 |
| 0.0 | 0.05 | 87.88 | +0.34 | 84.89 | +0.31 | 0.34 | 0.00 | 8.75 |
| 0.0 | 0.2 | 87.88 | +0.34 | 84.73 | +0.15 | 0.34 | 0.00 | 3.70 |

这等价于在 dev 的 16 个 overshoot 中恢复 1 个，且当前 grid 下没有 harm。

与 4.2 gate 组合后：

| gate | tau | SR | delta SR | recovery | harm | changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 0.0 | 87.54 | 0.00 | 0.00 | 0.00 | 7.74 |
| 0.6 | 0.2 | 87.54 | 0.00 | 0.00 | 0.00 | 2.36 |
| 0.7 | 0.0 | 87.54 | 0.00 | 0.00 | 0.00 | 6.06 |
| 0.7 | 0.2 | 87.54 | 0.00 | 0.00 | 0.00 | 2.36 |

当前结论是：ranker 本身已经能选出少量更好 endpoint，但 4.2 的保守 gate 阈值与这个 ranker 组合后，在 dev 上没有产生 recovery。4.4 需要联合选择 `gate_threshold` 与 `tau`，不能直接把 4.2 的单独 gate 阈值视为最终配置。

## 5. 本阶段结论

阶段 4.3 当前结论：

```text
endpoint ranker baseline 已实现并跑通。
ranker_scores.csv 已接入统一 evaluator，可生成 tau_curve 与 recovery_harm_curve。
dev candidate AUC = 0.9444，AP = 0.8863，说明 candidate-level trace features 有强信号。
all-pass diagnostic 在 dev 上恢复 1/16 overshoot，harm = 0。
4.2 conservative gate + 当前 CE ranker 暂未带来 dev SR 提升。
```

下一步进入 4.4：

```text
在 train/dev 上联合冻结 gate_threshold、tau、allow_change_final。
重点检查 gate 是否过于保守，以及 CE success ranker 是否需要加入 SPL / pairwise preference。
val_unseen 仍然不能用于调参。
```

本文档本轮更新状态：

```text
2026-04-29：新增阶段 4.3 endpoint ranker baseline 实现与 dev 结果，等待人工审核。
2026-04-29：已审核
```
