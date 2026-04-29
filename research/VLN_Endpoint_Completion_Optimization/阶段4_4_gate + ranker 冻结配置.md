# 阶段 4.4：gate + ranker 冻结配置报告

本文档承接 `research/VLN_Endpoint_Completion_Optimization/研究总路线.md` 和
`research/VLN_Endpoint_Completion_Optimization/阶段4_3_endpoint ranker baseline报告.md`，
只记录阶段 4.4 的实现入口、冻结选择协议、当前 dev 结果和后续人工决策点。

当前状态：

```text
reviewed：人工已读并认可口径.将其进行实验来源更换,重塑前面阶段的文档.
```

实验数据来源：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

注意：

```text
本阶段结果来自 0014 R2R trace 原型。
它用于验证 learned gate + CE ranker 的冻结闭环是否成立，
不等价于最终正式 train / dev / val_seen / val_unseen 协议。
是否切换到 0017 多 split 实验，由人工在本文档后续小节中补充结论。
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
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

如果从宿主机调用当前 docker 容器：

```bash
docker exec -w /workspace/vln-lab vln-same-cu128 bash -lc \
  'conda run -n endpoint-v1 python scripts/analysis/select_endpoint_frozen_config.py \
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5'
```

默认输出目录：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/endpoint_learning/frozen_gate_ranker/
```

关键输出：

```text
manifest.json
frozen_config.json
dev_selection_grid.csv
dev_selected_items.csv
failure_slice_summary.csv
dev_selection_report.md
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
                 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
tau = 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3
allow_change_final = true, false
```

默认 eligibility 约束：

```text
delta_SR >= 0
delta_SPL >= 0
harm_rate <= 1pp
allow_change_final = true 优先
```

排序规则：

```text
1. max delta_SR
2. max net_recovery_rate
3. min harm_rate
4. min changed_endpoint_rate
5. min gate_pass_rate
6. max delta_SPL
7. max gate_threshold
8. max tau
```

本阶段明确记录：

```text
val_unseen_used_for_selection = false
```

## 4. 当前冻结结果

当前 frozen config：

```text
gate_threshold = 0.40
tau = 0.20
allow_change_final = true
```

对应文件：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/endpoint_learning/frozen_gate_ranker/frozen_config.json
```

### 4.1 dev selection result

| split | final SR | selected SR | delta SR | final SPL | selected SPL | delta SPL | recovery | harm | changed | gate pass | gate precision | gate recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 87.54 | 87.88 | +0.34 | 84.58 | 84.73 | +0.15 | 0.34 | 0.00 | 3.03 | 27.27 | 16.05 | 81.25 |

解释：

```text
dev items = 297
overshoot items = 16
recovered overshoot = 1
harmed final-success item = 0
```

因此当前 CE ranker + learned gate 的冻结配置是低 harm 的，但 recovery 很弱。

### 4.2 train/dev frozen diagnostics

| split | SR | delta SR | SPL | delta SPL | recovery | harm | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 89.53 | +0.50 | 85.31 | +0.19 | 0.58 | 0.08 | 2.66 |
| dev | 87.88 | +0.34 | 84.73 | +0.15 | 0.34 | 0.00 | 3.03 |

该结果说明：

```text
在当前 0014 原型 split 上，frozen config 没有明显 harm。
但 SR 提升来自极少数样本，不能作为强方法结果。
```

## 5. 与 4.3 的衔接

4.3 已经说明：

```text
candidate-level AUC / AP 很高，说明 success_label 可学习。
all-pass ranker diagnostic 只能恢复 1 / 16 个 dev overshoot。
4.2 的 gate_threshold = 0.6 / 0.7 与 4.3 CE ranker 组合后没有 recovery。
```

4.4 的新增发现是：

```text
联合选择 gate_threshold 和 tau 后，可以恢复 1 / 16 个 dev overshoot，且 harm = 0。
最终入选 gate_threshold = 0.40，而不是 4.2 单独推荐的 0.60 或 0.70。
```

这说明：

```text
gate-only 阶段的保守阈值不能直接视为最终阈值。
最终阈值必须和 ranker score margin / tau 联合选择。
```

但这也暴露出当前 CE ranker 的瓶颈：

```text
当前 CE ranker 主要学 success / fail 区分。
它没有学好 success candidate 之间的 SPL preference。
所以即使 gate 放行，ranker 也经常不能选中真正可恢复 endpoint。
```

## 6. failure slice

使用 frozen config 的 dev failure slice：

| slice | items | rate | should_rerank | recovered | harmed | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final_success_kept | 258 | 86.87 | 0 | 0 | 0 | 0 |
| final_failure_kept | 19 | 6.40 | 0 | 0 | 0 | 0 |
| should_rerank_ranker_best_failed_or_final | 8 | 2.69 | 8 | 0 | 0 | 0 |
| should_rerank_ranker_best_failed | 4 | 1.35 | 4 | 0 | 0 | 4 |
| should_rerank_gate_rejected | 3 | 1.01 | 3 | 0 | 0 | 0 |
| recovered | 1 | 0.34 | 1 | 1 | 0 | 1 |
| final_success_changed_safe | 2 | 0.67 | 0 | 0 | 0 | 2 |
| final_failure_changed_unrecovered | 2 | 0.67 | 0 | 0 | 0 | 2 |

关键观察：

```text
16 个 should_rerank / overshoot dev episode 中：
1 个被恢复；
3 个被 gate 拒绝；
4 个 gate 放行但 ranker 改到 failed endpoint；
8 个没有被有效改动或 ranker best 仍失败 / final within tau。
```

因此当前主要瓶颈不是单纯 gate 过严，而是：

```text
ranker 对 recoverable endpoint 的 top1 选择能力不足。
```

## 7. 当前结论

阶段 4.4 当前结论：

```text
gate + ranker frozen selection 已实现并跑通。
frozen_config.json 已生成。
当前 0014 dev 上，frozen config 达到 low-harm，但 recovery 很弱。
该结果可以作为 CE learned baseline，不应被表述为强最终方法结果。
```

更具体地说：

```text
成立：离线 learned gate + ranker 的冻结闭环成立。
成立：联合选择 gate_threshold / tau 比直接使用 4.2 conservative threshold 更合理。
不充分：当前 CE ranker 只恢复 1 / 16 dev overshoot，方法收益很弱。
不充分：当前 0014 dev 样本太小，不适合支撑 4.5 的稳定消融结论。
```

## 8. 人工待补：是否切换到新实验协议

待人工填写：

```text
是否从 0014 R2R trace 原型切换到
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
作为后续正式阶段 1-4 重走文档分析的主实验？
```

建议人工判断时重点确认：

```text
1. 是否使用 R2R-only 作为主线；
2. 是否将 R2R train_eval 作为训练集；
3. 是否将 R2R val_train_seen 作为 dev / model-selection split；
4. 是否将 R2R val_seen 与 val_unseen 只作为 frozen final report；
5. 是否暂缓 multi-dataset joint training，把它留作后续扩展或 robustness 分析。
```

人工结论：

```text
切换到0017实验, 先仅考虑R2R数据集:
train:      R2R train_eval
dev:        R2R val_train_seen
final seen: R2R val_seen
final unseen: R2R val_unseen
```

## 9. 下一步

在人工确认数据协议前，不建议直接进入正式 4.5 消融。

当前更合理的下一步是：

```text
先确定是否切换到 0017；
如果切换，则复用现有代码，按新协议重写阶段 1 / 2 / 3 / 4.1 / 4.2 / 4.3 / 4.4 文档结果；
再在更大的 dev split 上启动 4.5 loss / feature ablation。
```

如果继续沿用 0014，则 4.5 很可能只是在：

```text
recover 0 个 vs recover 1 个
```

之间波动，统计解释价值不足。

本文档本轮更新状态：

```text
2026-04-29：新增阶段 4.4 gate + ranker 冻结配置草案，等待人工审核。
2026-04-29：已审核.
```
