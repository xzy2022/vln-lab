# 阶段 4.4：gate + ranker 冻结配置报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md` 和
`research/VLN_Endpoint_Completion_Optimization/阶段4_3_endpoint ranker baseline报告.md`，
只记录阶段 4.4 的实现入口、冻结选择协议、当前 dev 结果和后续人工决策点。

当前状态：

```text
reviewed：已切换到 0017 四数据集 joint 协议，并修复 4.4 selection 逻辑。
```

实验数据来源：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

注意：

```text
本阶段结果来自 0017 四数据集 joint 协议：
R2R / REVERIE / SOON / CVDN 的 train/dev 均参与 4.2 / 4.3 训练与 4.4 dev selection。
4.4 selection 已改为按 dev 四数据集 item 数加权聚合后选择 frozen config，
并额外检查单数据集 max harm，避免被某个数据集的局部收益牵引。
val_unseen 仍然不能用于调参。
```

## 1. 本阶段作用

阶段 4.4 对应总路线中的问题：

```text
learned gate + ranker 组合后是否比 rule-only 更稳？
```

本阶段不新增训练模型，只做冻结推理策略：

```text
gate model：复用 4.2 gate-only baseline
ranker model：复用 4.3 endpoint CE ranker baseline
selection split：dev
禁止使用 val_unseen 选择 gate_threshold、tau、allow_change_final
```

本阶段需要冻结：

```text
gate_threshold
tau
allow_change_final
```

并产出：

```text
frozen_config.json
dev_selection_grid.csv
dev_selection_report.md
failure_slice_summary.csv
```

## 2. 代码运行入口

新增入口：

```bash
python scripts/analysis/select_endpoint_frozen_config.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

以当前实验数据来源为例：

```bash
conda activate endpoint-v1
python scripts/analysis/select_endpoint_frozen_config.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
```

如果从宿主机调用当前 docker 容器：

```bash
docker exec -w /workspace/vln-lab vln-same-cu128 bash -lc \
  'conda run -n endpoint-v1 python scripts/analysis/select_endpoint_frozen_config.py \
  --experiment-dir experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1'
```

默认输出目录：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/frozen_gate_ranker/
```

关键输出：

```text
manifest.json
frozen_config.json
dev_selection_grid.csv
dev_selected_items.csv
failure_slice_summary.csv
dev_selection_report.md
dev_selection_dataset_grid.csv
dev_grid_eval_protocol/endpoint_learning_summary.csv
frozen_eval_protocol/endpoint_learning_summary.csv
```

## 3. 冻结选择协议

本阶段输入：

```text
candidate_csv = endpoint_learning/candidate_groups/endpoint_candidates.csv
episode_csv = endpoint_learning/candidate_groups/episode_groups.csv
score_csv = endpoint_learning/ranker_baseline/ranker_scores.csv
gate model metadata = endpoint_learning/gate_baseline/gate_model.json
ranker model metadata = endpoint_learning/ranker_baseline/ranker_model.json
```

推理规则：

```text
if gate_score < gate_threshold:
    choose final
else:
    best = argmax(candidate_score)
    if best_score <= final_score + tau:
        choose final
    else:
        choose best
```

默认 dev grid：

```text
gate_threshold = 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                 0.80, 0.85, 0.90, 0.95
tau = 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3
allow_change_final = true, false
```

默认 eligibility 约束：

```text
selection_aggregation = weighted
delta_SR >= 0
delta_SPL >= 0
harm_rate <= 1pp
max_dataset_harm_rate <= 1pp
allow_change_final = true 优先
```

这里的 `selection_aggregation = weighted` 表示：

```text
先按同一 gate_threshold / tau / allow_change_final 汇总四个 dev 数据集，
得到 dataset = ALL 的 item-weighted 指标；
再从 ALL 行中选择 frozen config。
原始 per-dataset 网格另存为 dev_selection_dataset_grid.csv，只用于诊断。
```

排序规则：

```text
1. max delta_SR
2. max net_recovery_rate
3. min harm_rate
4. min max_dataset_harm_rate
5. min changed_endpoint_rate
6. min gate_pass_rate
7. max delta_SPL
8. max gate_threshold
9. max tau
```

本阶段明确记录：

```text
val_unseen_used_for_selection = false
```

## 4. 当前冻结结果

当前 frozen config：

```text
gate_threshold = 0.90
tau = 0.20
allow_change_final = true
```

对应文件：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/frozen_gate_ranker/frozen_config.json
```

### 4.1 dev selection result

| split | aggregation | final SR | selected SR | delta SR | final SPL | selected SPL | delta SPL | recovery | harm | changed | gate pass | gate precision | gate recall | max dataset harm | worst dataset |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | weighted ALL | 72.12 | 72.20 | +0.09 | 66.17 | 66.38 | +0.21 | 0.32 | 0.23 | 2.01 | 4.12 | 54.90 | 18.95 | 0.63 | CVDN |

解释：

```text
dev items = 8177
overshoot items = 976
recovered items = 26
harmed final-success items = 19
dataset_count = 4
```

因此当前 CE ranker + learned gate 的冻结配置是低 harm 的，但 recovery 很弱。

### 4.2 train/dev frozen diagnostics

| dataset | split | SR | delta SR | SPL | delta SPL | recovery | harm | changed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | train | 87.29 | -0.06 | 82.88 | +0.02 | 0.06 | 0.11 | 0.51 |
| R2R | dev | 88.67 | -0.07 | 84.98 | -0.03 | 0.00 | 0.07 | 0.53 |
| REVERIE | train | 81.44 | -0.09 | 76.97 | -0.01 | 0.03 | 0.11 | 0.46 |
| REVERIE | dev | 83.74 | +0.00 | 79.22 | +0.02 | 0.00 | 0.00 | 0.81 |
| SOON | train | 71.22 | -0.00 | 64.74 | +0.11 | 0.26 | 0.27 | 1.69 |
| SOON | dev | 72.16 | +0.18 | 65.56 | +0.23 | 0.39 | 0.21 | 2.02 |
| CVDN | train | 45.57 | -0.13 | 40.37 | +0.23 | 0.29 | 0.42 | 4.85 |
| CVDN | dev | 44.82 | -0.21 | 40.06 | +0.52 | 0.42 | 0.63 | 4.44 |

该结果说明：

```text
在 0017 四数据集 joint dev 上，frozen config 的加权 SR 为小幅正收益。
但收益主要来自 SOON dev，R2R / REVERIE 没有明显恢复，CVDN dev 仍为负 delta_SR。
因此它是 low-harm learned baseline，不是强最终方法结果。
```

## 5. 与 4.3 的衔接

4.3 已经说明：

```text
candidate-level AUC / AP 很高，说明 success_label 可学习。
四数据集 all-pass weighted dev 为 recovery 2.56%、harm 2.95%、delta_SR -0.39pp。
R2R / REVERIE 有少量 recovery，但 SOON / CVDN 的 harm 会抵消总体收益。
4.2 在 0017 上推荐的 gate_threshold = 0.85 必须进入 4.4 grid。
```

4.4 的新增发现是：

```text
联合选择 gate_threshold 和 tau 后，最终入选 gate_threshold = 0.90、tau = 0.20。
4.2 推荐的 gate_threshold = 0.85 已进入搜索，但在 joint weighted selection 下没有胜出。
原因是 0.90 / 0.20 的 weighted delta_SR 更高，同时 max_dataset_harm_rate 仍低于 1pp。
```

这说明：

```text
gate-only 阶段的保守阈值不能直接视为最终阈值。
最终阈值必须和 ranker score margin / tau 联合选择。
4.4 不能再从 per-dataset row 中挑最优配置，否则会偏向 REVERIE 这种小样本高收益行。
```

但这也暴露出当前 CE ranker 的瓶颈：

```text
当前 CE ranker 主要学 success / fail 区分。
它没有学好 success candidate 之间的 SPL preference。
所以即使 gate 压住了 SOON / CVDN 的大部分 harm，
joint weighted dev 的 SR 增益仍然只有 +0.09pp。
```

## 6. failure slice

使用 frozen config 的 dev failure slice：

| slice | items | rate | should_rerank | recovered | harmed | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final_success_kept | 5861 | 71.68 | 0 | 0 | 0 | 0 |
| final_failure_kept | 1276 | 15.60 | 0 | 0 | 0 | 0 |
| should_rerank_gate_rejected | 791 | 9.67 | 791 | 0 | 0 | 0 |
| should_rerank_ranker_best_failed | 74 | 0.90 | 74 | 0 | 0 | 74 |
| should_rerank_ranker_best_failed_or_final | 56 | 0.68 | 56 | 0 | 0 | 0 |
| should_rerank_tau_blocked_success | 29 | 0.35 | 29 | 0 | 0 | 0 |
| final_failure_changed_unrecovered | 28 | 0.34 | 0 | 0 | 0 | 28 |
| recovered | 26 | 0.32 | 26 | 26 | 0 | 26 |
| final_success_harmed | 19 | 0.23 | 0 | 0 | 19 | 19 |
| final_success_changed_safe | 17 | 0.21 | 0 | 0 | 0 | 17 |

关键观察：

```text
976 个 should_rerank / overshoot dev episode 中：
26 个被恢复；
791 个被 gate 拒绝；
74 个 gate 放行但 ranker 改到 failed endpoint；
56 个没有被有效改动或 ranker best 仍失败 / final within tau；
29 个被 tau 留在 final，虽然存在成功 endpoint。
```

因此当前主要瓶颈不是单纯 gate 过严，而是：

```text
ranker 对 recoverable endpoint 的 top1 选择能力不足；
同时 tau 在减少 harm 的同时会挡掉一部分可恢复 endpoint。
```

## 7. 当前结论

阶段 4.4 当前结论：

```text
gate + ranker frozen selection 已按 0017 四数据集 joint 协议实现并跑通。
selection 逻辑已从 per-dataset row selection 修正为 weighted ALL selection。
frozen_config.json 已生成。
当前 0017 joint dev 上，frozen config 达到 low-harm，但 recovery 很弱。
该结果可以作为 CE learned baseline，不应被表述为强最终方法结果。
```

更具体地说：

```text
成立：离线 learned gate + ranker 的冻结闭环成立。
成立：联合选择 gate_threshold / tau 比直接使用 4.2 conservative threshold 更合理。
成立：selection 借鉴了 4.3 的建议，已纳入 gate_threshold = 0.85，并重点约束 SOON / CVDN harm。
成立：协议切换后不再由 REVERIE 单数据集小样本行决定 frozen config。
不充分：当前 CE ranker 在 8177 个 joint dev item 上只带来 +0.09pp SR。
不充分：CVDN dev 仍为 -0.21pp SR，说明 harm 被压低但没有彻底解决。
```

## 8. 协议切换结论

当时考虑从 0014 切到新实验协议，核心问题是：

```text
0014 只是 R2R trace 原型，dev 样本太小；
4.4 的 recover 0 / 1 个样本波动无法支撑稳定结论；
它也不覆盖 REVERIE / SOON / CVDN，无法回答 4.3 暴露出的 cross-dataset harm。
```

现在切换到 0017 四数据集 joint 协议后，已经解决的问题：

```text
1. 样本量问题：dev 从 297 item 扩展到 8177 item；
2. 数据集覆盖问题：selection 同时覆盖 R2R / REVERIE / SOON / CVDN；
3. 4.3 建议落实问题：gate_threshold = 0.85 已进入 grid，并新增更高阈值 0.90 / 0.95；
4. 选择偏置问题：selection 不再按单个 dataset 行排序，而是按 weighted ALL 行排序；
5. harm 诊断问题：frozen_config 记录 max_dataset_harm_rate 与 worst_harm_dataset。
```

仍未解决的问题：

```text
1. CE success ranker 的收益仍很弱，joint dev 只提升 +0.09pp SR；
2. CVDN dev 仍为负 delta_SR，说明 gate 只能压 harm，不能让 ranker 真正稳定恢复 endpoint；
3. ranker 仍可能需要 SPL-aware objective 或 pairwise preference。
```

## 9. 下一步

当前更合理的下一步是：

```text
基于 0017 四数据集 joint 协议进入 4.5；
优先做 SPL-aware / pairwise preference ranker；
继续用 dev weighted ALL 选择 frozen config；
继续把 max_dataset_harm_rate 作为硬约束；
val_unseen 只用于 frozen final report，不用于调参。
```

本文档本轮更新状态：

```text
2026-04-29：新增阶段 4.4 gate + ranker 冻结配置草案，等待人工审核。
2026-04-29：已审核.
2026-04-29：切换到 0017 四数据集 joint 协议，并修复 weighted selection 逻辑.
```
