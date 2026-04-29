# 阶段 4.3：endpoint ranker baseline 报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md`，只记录阶段 4.3 的实现入口、实验产物和当前 dev 结果。

当前状态：

```text
reviewed：人工已读并认可口径
```

实验数据来源：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
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
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

如果从宿主机调用当前 docker 容器：

```bash
docker exec -w /workspace/vln-lab vln-same-cu128 bash -lc \
  'conda run -n endpoint-v1 python scripts/analysis/train_endpoint_ranker_baseline.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1'
```

默认输出目录：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/ranker_baseline/
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
0.6 / 0.7 = 当前 ranker baseline 默认 gate grid 中的保守候选
```

## 4. 当前实验结果

以下结果使用：

```text
experiment = 0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
datasets = R2R, REVERIE, SOON, CVDN
target_scope = official
train candidates = 493121
dev candidates = 92303
features = 32
```

### 4.1 candidate-level 指标

| split | candidates | episodes | positive rate | AUC | AP | brier | log loss | top1 success | top1 harm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 493121 | 50494 | 19.83 | 0.9197 | 0.7467 | 0.1072 | 0.3428 | 75.89 | 2.63 |
| dev | 92303 | 8177 | 18.43 | 0.9151 | 0.7186 | 0.1048 | 0.3339 | 71.73 | 2.95 |

这说明 candidate-level trace features 对 `success_label` 仍有明显可学习信号；但在四数据集合并后，top1 直接改 endpoint 会产生更高 harm，不能只看 candidate AUC / AP。

### 4.2 preference pair 一致率

| split | pair type | pairs | accuracy | mean margin |
| --- | --- | ---: | ---: | ---: |
| train | all | 1116725 | 83.53 | 0.4779 |
| dev | all | 214864 | 83.52 | 0.4664 |
| dev | success_gt_fail | 155213 | 85.46 | 0.4593 |
| dev | final_success_final_gt_failed_nonfinal | 44239 | 94.43 | 0.6921 |
| dev | better_spl_success_gt_lower_spl_success | 15412 | 32.66 | -0.1102 |

当前 CE ranker 仍然主要学会区分 success / fail，也能较好保留 final-success 样本中的成功 final；但它没有学好“成功 candidate 之间按 SPL 排序”。这符合预期，因为 4.3 baseline 只优化 `success_label`，没有优化 SPL preference。

### 4.3 evaluator bridge

dev final baseline：

| dataset | items | final SR | final SPL | oracle / nearest SR | overshoot items |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2R | 1501 | 88.74 | 85.01 | 94.34 | 84 |
| REVERIE | 123 | 83.74 | 79.20 | 93.50 | 12 |
| SOON | 5607 | 71.98 | 65.33 | 83.86 | 666 |
| CVDN | 946 | 45.03 | 39.54 | 67.65 | 214 |
| weighted all | 8177 | 72.12 | 66.17 | 83.96 | 976 |

all-pass ranker diagnostic：

| gate | tau | SR | delta SR | SPL | delta SPL | recovery | harm | changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 71.73 | -0.39 | 67.21 | +1.04 | 2.56 | 2.95 | 27.63 |
| 0.0 | 0.05 | 71.77 | -0.34 | 66.73 | +0.56 | 1.60 | 1.94 | 15.18 |
| 0.0 | 0.2 | 71.79 | -0.33 | 66.37 | +0.20 | 0.61 | 0.94 | 6.11 |

按数据集拆开看，`gate = 0.0, tau = 0.0` 的 SR 变化并不一致：

| dataset | SR | delta SR | SPL | delta SPL | recovery | harm | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | 88.81 | +0.07 | 85.71 | +0.70 | 0.80 | 0.73 | 11.66 |
| REVERIE | 85.37 | +1.63 | 80.97 | +1.77 | 1.63 | 0.00 | 14.63 |
| SOON | 71.46 | -0.52 | 66.40 | +1.07 | 3.00 | 3.51 | 29.25 |
| CVDN | 44.40 | -0.63 | 40.90 | +1.36 | 2.85 | 3.49 | 45.03 |

与 4.2 gate 组合后，当前默认 grid 中较保守的 `gate_threshold = 0.7` 仍然不能带来整体 SR 提升：

| gate | tau | SR | delta SR | SPL | delta SPL | recovery | harm | changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 0.0 | 71.98 | -0.13 | 66.88 | +0.71 | 1.94 | 2.08 | 15.34 |
| 0.7 | 0.2 | 71.92 | -0.20 | 66.39 | +0.22 | 0.56 | 0.76 | 5.14 |

当前结论是：ranker 确实学到了 candidate success 信号，并且在 R2R / REVERIE 上可以恢复少量 endpoint；但四数据集加权后，SOON / CVDN 的 harm 抵消了 recovery。4.4 需要联合选择 `gate_threshold` 与 `tau`，并且需要把 4.2 在 0017 上推荐的 `gate_threshold = 0.85` 纳入冻结配置搜索，而不能只沿用本次默认 grid。

## 5. 本阶段结论

阶段 4.3 当前结论：

```text
endpoint ranker baseline 已实现并跑通。
ranker_scores.csv 已接入统一 evaluator，可生成 tau_curve 与 recovery_harm_curve。
dev candidate AUC = 0.9151，AP = 0.7186，说明 candidate-level trace features 有强信号。
all-pass diagnostic 在四数据集加权 dev 上 recovery = 2.56%，harm = 2.95%，SR 净变化 = -0.39pp。
当前默认 gate grid + CE ranker 暂未带来整体 dev SR 提升，但 R2R / REVERIE 上有少量 recovery。
```

下一步进入 4.4：

```text
在 train/dev 上联合冻结 gate_threshold、tau、allow_change_final。
重点检查 gate 是否能压住 SOON / CVDN 的 harm，以及 CE success ranker 是否需要加入 SPL / pairwise preference。
val_unseen 仍然不能用于调参。
```

本文档本轮更新状态：

```text
2026-04-29：新增阶段 4.3 endpoint ranker baseline 实现与 dev 结果，等待人工审核。
2026-04-29：已审核
2026-04-29：R2R val 切换到 R2R / REVERIE / SOON / CVDN 数据集的全部split
```
