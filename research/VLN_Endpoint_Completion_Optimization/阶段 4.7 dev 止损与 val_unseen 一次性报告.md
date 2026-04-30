# 阶段4.7 dev 止损与 val_unseen 一次性报告

状态：frozen val_unseen 已完成，待审核

本阶段只做 4.6 已冻结候选的 `val_unseen` 一次性报告，不再根据
`val_unseen` 结果反向修改 objective、gate threshold、tau 或模型选择。

阶段 4.7 的核心问题是：

```text
4.6 final-bias 解耦 ranker 在 dev 上过线后，
冻结配置能否把 recovery 低 harm 地外推到 val_unseen？
```

当前结论：

```text
safety 主配置在 val_unseen 上没有 SR 收益，只带来极小 SPL 收益；
dev_best sensitivity 有更明显 SPL 收益，但 SR 只净增约 4 items，
且 harm 明显升高，尤其 SOON / CVDN。

因此阶段 4 不应继续在当前 trace-only gate + ranker 框架下扫参数。
较稳妥的收束口径是：
final-bias 解耦 ranker 职责在 dev 上成立，
但当前离线 gate 的跨 split 泛化不足，
val_unseen 上的 recovery 被 harm 基本抵消。
```

## 1. 冻结协议

本阶段使用 4.6 `phase4_6_final_bias_sanity` 产出的同一个 ranker score：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/preference_ablation/final_bias_sanity/runs/final_bias_decoupled_pairwise_listwise_best_spl/preference_ranker_scores.csv
```

frozen candidate 来源：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/preference_ablation/final_bias_sanity/frozen_candidates.json
```

评测 split：

```text
split = val_unseen
protocol_split = unseen_test
target_scope = official
allow_change_final = true
```

本阶段只报告两个预注册候选：

| candidate | 用途 | gate | tau | dev dSR | dev dSPL | dev recovery | dev harm | dev max harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `safety` | 主 frozen 报告 | 0.95 | 0.03 | +0.23pp | +0.34pp | 0.31pp | 0.07pp | 0.11pp |
| `dev_best` | sensitivity / 有动作上界 | 0.90 | 0.01 | +0.38pp | +0.79pp | 0.90pp | 0.53pp | 0.95pp |

注意：

```text
dev_best 不是根据 val_unseen 选择出来的配置；
它是 4.6 dev selection 中已经保存的收益候选。
```

## 2. 运行命令

### 2.1 safety

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
RUN=final_bias_decoupled_pairwise_listwise_best_spl

python scripts/analysis/eval_endpoint_reranker.py \
  --experiment-dir "$EXP" \
  --score-csv "$EXP/endpoint_learning/preference_ablation/final_bias_sanity/runs/$RUN/preference_ranker_scores.csv" \
  --output-dir "$EXP/endpoint_learning/preference_ablation/final_bias_sanity/val_unseen_safety" \
  --split val_unseen \
  --gate-thresholds 0.95 \
  --taus 0.03 \
  --allow-change-final true
```

### 2.2 dev_best sensitivity

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
RUN=final_bias_decoupled_pairwise_listwise_best_spl

python scripts/analysis/eval_endpoint_reranker.py \
  --experiment-dir "$EXP" \
  --score-csv "$EXP/endpoint_learning/preference_ablation/final_bias_sanity/runs/$RUN/preference_ranker_scores.csv" \
  --output-dir "$EXP/endpoint_learning/preference_ablation/final_bias_sanity/val_unseen_dev_best" \
  --split val_unseen \
  --gate-thresholds 0.90 \
  --taus 0.01 \
  --allow-change-final true
```

主要输出：

| output dir | 内容 |
| --- | --- |
| `.../final_bias_sanity/val_unseen_safety/` | safety 主配置评测 |
| `.../final_bias_sanity/val_unseen_dev_best/` | dev_best sensitivity 评测 |

每个目录包含：

```text
endpoint_learning_items.csv
endpoint_learning_summary.csv
tau_curve.csv
recovery_harm_curve.csv
endpoint_reranker_eval_manifest.json
```

注意：同一个 `--output-dir` 下这些文件会被 evaluator 直接重写，因此
`safety` 和 `dev_best` 必须分目录保存。

## 3. weighted ALL 结果

### 3.1 safety 主结果

`val_unseen_safety` 的 weighted ALL：

```text
items     = 10167
delta_SR  = 约 0.000pp
delta_SPL = +0.064pp
recovery  = 21 items = 0.207pp
harm      = 21 items = 0.207pp
changed   = 1.14%
```

分数据集：

| dataset | items | dSR | dSPL | recovery | harm | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | 2349 | +0.13pp | +0.11pp | 0.26pp | 0.13pp | 0.55pp |
| REVERIE | 3521 | -0.03pp | +0.05pp | 0.26pp | 0.28pp | 0.99pp |
| SOON | 3390 | +0.12pp | +0.11pp | 0.15pp | 0.03pp | 1.06pp |
| CVDN | 907 | -0.66pp | -0.19pp | 0.11pp | 0.77pp | 3.53pp |

观察：

```text
safety 配置确实保守，整体 changed 只有 1.14%；
但 recovery 与 harm 完全打平，SR 没有净收益；
主要外推失败来自 CVDN，CVDN dSR = -0.66pp。
```

### 3.2 dev_best sensitivity

`val_unseen_dev_best` 的 weighted ALL：

```text
items     = 10167
delta_SR  = +0.039pp，约净 +4 items
delta_SPL = +0.373pp
recovery  = 87 items = 0.856pp
harm      = 83 items = 0.816pp
changed   = 4.79%
```

分数据集：

| dataset | items | dSR | dSPL | recovery | harm | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | 2349 | +0.17pp | +0.34pp | 0.55pp | 0.38pp | 1.92pp |
| REVERIE | 3521 | +0.03pp | +0.28pp | 0.71pp | 0.68pp | 3.38pp |
| SOON | 3390 | -0.21pp | +0.33pp | 0.88pp | 1.09pp | 6.52pp |
| CVDN | 907 | +0.66pp | +0.98pp | 2.09pp | 1.43pp | 11.25pp |

观察：

```text
dev_best 比 safety 更像有动作的 sensitivity result；
SPL 收益更稳定，但 SR 只净增约 4 items；
SOON 出现负 dSR，CVDN harm 超过 1pp。
```

## 4. 是否达到阶段 4 成功标准

阶段 4 路线级最低标准是：

```text
val_unseen delta_SR > 0
harm_rate <= 1pp
SPL 不下降
```

更强标准是：

```text
recover >= 10% oracle gap
R2R val_unseen SR 提升 >= 0.85pp
harm_rate <= 1pp
```

本次结果判断：

| candidate | weighted dSR > 0 | weighted harm <= 1pp | SPL 不下降 | max dataset harm <= 1pp | 结论 |
| --- | --- | --- | --- | --- | --- |
| `safety` | false | true | true | true | 主结果未带来 SR 收益 |
| `dev_best` | 勉强 true | true | true | false | sensitivity 有微弱 SR / SPL 收益，但 harm 不稳 |

解释：

```text
safety 没有达到最低 SR 收益条件。
dev_best 按 weighted ALL 看只有 +0.039pp SR，约净 +4 items，
虽然 SPL 为正且 weighted harm < 1pp，
但 CVDN harm = 1.43pp，SOON dSR = -0.21pp。

因此不能把 dev_best 当成强方法结果。
它最多说明：
ranker 确实能找到一部分更短或更好的 endpoint，
但当前 gate 无法稳定区分 recovery 和 harm。
```

## 5. 阶段结论

4.6 的 dev 结论仍然成立：

```text
final-bias 解耦是必要的；
gate 应负责是否允许修改 final；
ranker 应负责在允许修改时选择更好的 endpoint。
```

但 4.7 的 frozen `val_unseen` 结果说明：

```text
当前 trace-only gate + final-bias decoupled ranker
没有把 dev 上的 recovery 稳定外推到 val_unseen。

SPL 有稳定小收益，
SR 几乎打平或只有极弱正收益，
harm 在更激进候选上快速上升。
```

因此阶段 4 建议收束为：

```text
ranker 职责解耦有效，但当前离线 gate 泛化不足；
阶段 4 可作为 weak / negative result 收束，
不继续在 val_unseen 上调 threshold / tau / objective。
```

## 6. 后续方向

不建议继续做：

```text
继续扫 4.6 objective
继续扫 val_unseen threshold / tau
把 dev_best 当作新的起点在 val_unseen 上调参
```

可以作为下一阶段新问题，而不是 4.6/4.7 的延伸：

```text
gate 泛化诊断：为什么 val_unseen precision / recall 迁移差
dataset-aware 或 uncertainty-aware gate
引入视觉目标证据、语言 grounding、dialog evidence
multi-trajectory consensus
把目标改成 SPL-preserving rerank，而不是单纯 SR recovery
```

当前 frozen 报告的最终口径：

```text
dev 上存在清楚上界和 ranker 解耦收益；
val_unseen 上主配置 safety 没有 SR 收益；
dev_best sensitivity 只得到极弱 SR 正收益并伴随更高 harm；
阶段 4 不进入更大规模 learned endpoint reranker 主实验。
```
