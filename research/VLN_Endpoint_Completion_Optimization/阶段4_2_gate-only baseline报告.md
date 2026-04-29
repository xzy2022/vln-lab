# 阶段 4.2：gate-only baseline 报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md`，只记录阶段 4.2 的作用、运行入口、实现注意事项和当前实验结果。

当前状态：

```text
reviewed：人工已读并认可口径
```

实验数据来源：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

## 1. 本阶段作用

阶段 4.2 对应总路线中的问题：

```text
能否先学会“什么时候不要改 final”？
```

它只训练 episode-level gate，不训练 endpoint ranker。也就是说，本阶段学习：

```text
输入：SAME trace 的 episode-level / aggregated candidate features
输出：gate_score = P(should_rerank)
label：should_rerank = final_success == false AND oracle_success == true
```

本阶段的目标不是提升 SR，而是判断 SAME trace 中是否存在足够信号来预测“这个 episode 是否值得尝试 endpoint correction”。真正“如果要改，改到哪个 candidate”属于 4.3 endpoint ranker。

## 2. 代码运行入口

本阶段新增入口：

```bash
python scripts/analysis/train_endpoint_gate_baseline.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

以当前实验数据来源为例：

```bash
conda activate endpoint-v1
python scripts/analysis/train_endpoint_gate_baseline.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

如果从宿主机调用当前 docker 容器：

```bash
docker exec -w /workspace/vln-lab vln-same-cu128 bash -lc \
  'conda run -n endpoint-v1 python scripts/analysis/train_endpoint_gate_baseline.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1'
```

默认输出目录：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/gate_baseline/
```

输出文件：

```text
manifest.json
gate_features.csv
gate_episode_scores.csv
gate_scores.csv
gate_threshold_candidates.csv
gate_baseline_summary.csv
gate_feature_importance.csv
gate_model.joblib
gate_model.json
gate_baseline_report.md
eval_protocol/endpoint_learning_summary.csv
eval_protocol/tau_curve.csv
eval_protocol/recovery_harm_curve.csv
```

其中 `gate_scores.csv` 是给后续统一 evaluator 使用的关键文件，包含：

```text
candidate_id
candidate_score
gate_score
```

当前 `candidate_score` 暂时沿用 candidate `stop_prob`，只用于 evaluator 桥接；4.2 的新增能力只看 `gate_score`。

## 3. 代码实现注意事项

### 3.1 模型

当前 baseline 使用：

```text
sklearn LogisticRegression
class_weight = balanced
SimpleImputer(strategy = median)
StandardScaler
C = 1.0
max_iter = 2000
random_state = 17
```

这样做的原因是：

```text
正样本很少，dev base rate 只有 11.94%
需要一个可解释、可复现、轻量的 gate baseline
先验证 episode-level trace feature 是否有信号，再决定是否需要更复杂模型
```

### 3.2 训练 / 选择协议

当前协议：

```text
train：protocol_split == train
dev：protocol_split == dev
val_unseen / test：不用于训练，不用于选 threshold，默认不汇报指标
```

脚本默认只报告 train/dev。虽然 `gate_scores.csv` 会给所有 episode 导出 gate_score，便于后续冻结配置复用，但除非显式加 `--include-test-summary`，不会汇报 `val_unseen` gate 指标。

### 3.3 feature 边界

可用 feature 来自 SAME trace 和 trajectory aggregation，例如：

```text
trajectory_step_count
decision_trace_candidate_rate
route_expanded_without_decision_candidate_rate
revisit_rate
loop_region_rate
final_stop_prob
max_stop_prob
mean_stop_prob
last_k_max_stop_prob
max_stop_minus_final_stop
argmax_stop_step_frac
final_selected_prob
mean_router_entropy
final_router_entropy
path_length_m aggregation
```

禁止进入 feature 的字段包括：

```text
success_label
distance_to_goal_m
reward
is_nearest_candidate
final_success
oracle_success
nearest_endpoint_success
should_rerank
final_distance_m
best_distance_m
final_spl
nearest_endpoint_spl
```

这些字段只能用于 label、evaluation 或离线分析，不能作为推理输入。

## 4. 当前实验结果

以下结果使用：

```text
experiment = 0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
datasets = R2R, REVERIE, SOON, CVDN
target_scope = official
train episodes = 50494
dev episodes = 8177
features = 46
```

### 4.1 gate 分类结果

| split | items | positives | base rate | AUC | AP | brier | log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 50494 | 5115 | 10.13 | 0.8398 | 0.3832 | 0.1671 | 0.5041 |
| dev | 8177 | 976 | 11.94 | 0.8101 | 0.3876 | 0.1993 | 0.5870 |

`dev` 上 `average_precision = 0.3876`，明显高于 `base rate = 0.1194`。这说明 episode-level / aggregated trace features 对 `should_rerank` 仍有可学习信号；0017 覆盖四个数据集后，整体 AUC = 0.8101，说明多数据集 gate 任务更混杂、更难。

### 4.2 dev threshold 候选

| tag | threshold | pass rate | precision | recall | f1 | final-success pass rate | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best_f1 | 0.65 | 22.88 | 32.71 | 62.70 | 0.4299 | 14.69 | 612 | 1259 | 364 |
| low_pass | 0.70 | 18.61 | 34.76 | 54.20 | 0.4235 | 11.41 | 529 | 993 | 447 |
| conservative | 0.85 | 7.22 | 46.27 | 27.97 | 0.3487 | 3.37 | 273 | 317 | 703 |

当前推荐把 `gate_threshold = 0.85` 作为 4.2 的保守候选：

```text
只放行 7.22% dev episode
命中 273 / 976 个 should_rerank episode
precision = 46.27%，约为 base rate 的 3.9 倍
final-success pass rate = 3.37%，比低阈值更保守
```

这个阈值只是 4.2 的 dev 候选，不是最终 frozen 配置。最终配置需要等 4.3 ranker 输出后，在 4.4 中一起冻结。

### 4.3 evaluator 桥接结果

脚本同时把 `gate_scores.csv` 接入了：

```text
scripts/analysis/eval_endpoint_reranker.py
```

当前 evaluator 桥接仍使用：

```text
candidate_score = stop_prob
gate_score = learned gate output
```

因此它只能验证 gate score 与 evaluator 已经打通，不能作为最终 SR 提升结论。当前 dev 上 SR / recovery 没有变化，主要原因是 endpoint scorer 仍是临时的 `stop_prob`，还没有训练 4.3 ranker。

`gate_threshold = 0.85, tau = 0` 下的桥接结果如下：

| dataset | items | gate pass | gate precision | gate recall | selected SR | delta SR | selected SPL | delta SPL | changed endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | 1501 | 2.13 | 62.50 | 23.81 | 88.74 | 0.00 | 85.04 | 0.03 | 0.33 |
| REVERIE | 123 | 4.88 | 33.33 | 16.67 | 83.74 | 0.00 | 79.45 | 0.25 | 0.81 |
| SOON | 5607 | 7.08 | 45.59 | 27.18 | 71.98 | 0.00 | 65.53 | 0.19 | 1.05 |
| CVDN | 946 | 16.38 | 45.16 | 32.71 | 45.03 | 0.00 | 40.22 | 0.68 | 1.80 |

### 4.4 重要 feature

当前 logistic gate 的标准化系数绝对值最高的特征包括：

| feature | coef | abs coef |
| --- | ---: | ---: |
| mean_stop_prob | 1.8835 | 1.8835 |
| route_intermediate_candidate_rate | 1.3067 | 1.3067 |
| mean_selected_prob | -1.1187 | 1.1187 |
| final_stop_margin_prob | 1.1159 | 1.1159 |
| mean_stop_margin_prob | -0.9855 | 0.9855 |
| mean_path_length_m | -0.9026 | 0.9026 |
| final_stop_rank_frac | -0.8636 | 0.8636 |
| max_stop_margin_prob | -0.7147 | 0.7147 |
| max_stop_prob | -0.6633 | 0.6633 |
| max_path_length_m | 0.6629 | 0.6629 |

这些系数只用于诊断，不应过度解释因果关系。它们说明 gate 主要在利用 stop confidence 分布、route-expanded / intermediate 轨迹结构、path length 和 final stop margin 等信号。

## 5. 本阶段结论

阶段 4.2 当前结论是：

```text
gate-only baseline 已实现并跑通。
dev 上 AUC = 0.8101，AP = 0.3876，明显高于 11.94% base rate。
episode-level SAME trace features 可以学习“何时可能需要 rerank”。
推荐保留 gate_threshold = 0.85 作为后续 4.4 的候选之一。
```

但本阶段还不能证明 endpoint correction 有 SR 收益：

```text
4.2 只学习是否放行 endpoint correction；
当前 evaluator 桥接仍用 stop_prob 当 candidate_score；
真正选择 endpoint 的 ranker 属于 4.3。
```

因此，本阶段支持继续进入阶段 4.3：

```text
训练 endpoint ranker，学习“如果 gate 放行，应该改到哪个 candidate”。
```

本文档本轮更新状态：

```text
2026-04-29：补全阶段 4.2 gate-only baseline 报告草案，等待人工审核。
2026-04-29：已审核
2026-04-29：R2R val 切换到 R2R / REVERIE / SOON / CVDN 数据集的全部split
```
