# 阶段4.8 gate 泛化诊断收束

状态：implemented，诊断已完成，待人工审核

本阶段正面回应 `阶段4_总结与下一步计划.md` 的 5.1 P0：

```text
新增一个只分析、不训练新模型的 gate 泛化诊断阶段，
用四张表解释 dev 到 val_unseen 的断点，
不改变 4.7 frozen val_unseen 结论。
```

当前结论：

```text
4.7 的断点不是单纯 gate calibration 问题；
gate 的跨 split 校准与 threshold net band 明显变弱，
同时 final-bias decoupled ranker 的 top1 / best-SPL 排序也从 dev 到 val_unseen 下降。

因此当前 single-trajectory trace-only gate + ranker 的外推失败应归因为 both：
gate 无法稳定过滤 harm，ranker top1 也没有稳定保留 dev 上的 endpoint preference。
```

这说明 5.2 的 P1 不应继续沿当前框架扫 threshold / tau / loss，而应优先选择：

```text
evidence-augmented endpoint verification
```

即让 trace-only ranker 只提供 top-k endpoint，再用额外 evidence verifier 判断候选 endpoint 是否真的满足 instruction / object / dialog goal。

## 1. 目标

阶段 4.8 要回答四个问题，对应 5.1 P0 的四张最小表：

| 表 | 问题 | 本阶段回答 |
| --- | --- | --- |
| A | dev 与 `val_unseen` 是否有 gate calibration drift | 有，AUC / AP 下降，Brier / ECE 变差 |
| B | 是否存在真正分开 recovery/harm 的 threshold | 只有很弱的低 harm band，val_unseen recovery 与 harm 基本贴边 |
| C | recovered / harmed / changed-safe / changed-unrecovered 的 trace feature 是否可分 | 不稳定，主要是 dataset-specific，而不是清晰可分 |
| D | 断点主要来自 gate 还是 ranker top1 漂移 | 两者都有；ranker top1 也明显从 dev 掉到 `val_unseen` |

本阶段只做诊断，不训练新模型，不使用 `val_unseen` 调参，不把任何 `val_unseen` threshold 反向写成新 frozen config。

## 2. 实现入口

新增脚本：

```bash
scripts/analysis/diagnose_endpoint_gate_generalization.py
```

本次运行命令：

```bash
EXP=experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1
RUN=final_bias_decoupled_pairwise_listwise_best_spl

conda run -n endpoint-v1 python scripts/analysis/diagnose_endpoint_gate_generalization.py \
  --experiment-dir "$EXP" \
  --score-csv "$EXP/endpoint_learning/preference_ablation/final_bias_sanity/runs/$RUN/preference_ranker_scores.csv" \
  --output-dir "$EXP/endpoint_learning/gate_generalization_diagnostics/phase4_8_final_bias_sanity" \
  --ranker-diagnostics-dir "$EXP/endpoint_ranker_diagnostics/phase4_8_final_bias_sanity_dev_val_unseen" \
  --splits dev,val_unseen \
  --target-scope official \
  --gate-thresholds 0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99 \
  --taus 0.01,0.03
```

诊断产物写入：

```text
experiment_outputs/0017_same_val_train_eval_all_r2r_reverie_cvdn_soon_same_s0_trace4ds_train_eval_v1/endpoint_learning/gate_generalization_diagnostics/phase4_8_final_bias_sanity
```

主要输出：

| 文件 | 内容 |
| --- | --- |
| `gate_generalization_diagnostics_report.md` | 四表汇总与自动诊断 |
| `manifest.json` | 输入、配置、no-training 协议与产物索引 |
| `gate_calibration_by_split_dataset.csv` | 表 A，split / dataset gate calibration |
| `gate_calibration_bins.csv` | 表 A 的 calibration bins |
| `gate_threshold_confusion.csv` | 表 B，weighted ALL threshold confusion |
| `gate_threshold_confusion_by_dataset.csv` | 表 B，dataset slice |
| `changed_item_feature_shift.csv` | 表 C，changed item feature shift |
| `changed_item_feature_shift_effects.csv` | 表 C，harm-recovered effect size |
| `ranker_top1_stability.csv` | 表 D，weighted ALL ranker top1 stability |
| `ranker_top1_stability_by_dataset.csv` | 表 D，dataset slice |

脚本依赖：

```text
pandas / numpy / sklearn.metrics
```

并复用：

```text
scripts/analysis/eval_endpoint_reranker.py
scripts/analysis/diagnose_endpoint_ranker_top1.py
```

## 3. 诊断协议

输入沿用 4.6 / 4.7 的 frozen artifacts：

```text
candidate_csv = endpoint_learning/candidate_groups/endpoint_candidates.csv
episode_csv = endpoint_learning/candidate_groups/episode_groups.csv
gate_features_csv = endpoint_learning/gate_baseline/gate_features.csv
score_csv = endpoint_learning/preference_ablation/final_bias_sanity/runs/final_bias_decoupled_pairwise_listwise_best_spl/preference_ranker_scores.csv
pair_csv = endpoint_learning/preference_pairs/preference_pairs.csv
```

诊断 split：

```text
target_scope = official
split = dev,val_unseen
```

固定诊断 grid：

```text
gate_threshold = 0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99
tau = 0.01,0.03
allow_change_final = true
```

其中：

| config | gate | tau | 对应含义 |
| --- | ---: | ---: | --- |
| `dev_best` | 0.90 | 0.01 | 4.7 sensitivity / 有动作配置 |
| `safety` | 0.95 | 0.03 | 4.7 主 frozen safety 配置 |

重要约束：

```text
val_unseen_used_for_selection = false
model_training = false
```

本阶段的 threshold grid 只用于诊断 recovery-harm 曲线形状，不产生新 frozen config。

## 4. 关键结果

### 4.1 表 A：gate calibration by split / dataset

weighted ALL：

| split | items | positives | base rate | mean gate | gap | AUC | AP | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 8177 | 976 | 11.94% | 41.49% | +29.56pp | 0.8101 | 0.3876 | 0.1993 | 0.2956 |
| `val_unseen` | 10167 | 1372 | 13.49% | 45.02% | +31.52pp | 0.7465 | 0.3379 | 0.2414 | 0.3152 |
| drift | - | - | +1.56pp | +3.52pp | +1.96pp | -0.0636 | -0.0497 | +0.0420 | +0.0196 |

诊断判断：

```text
gate 并不是完全失效，但从 dev 到 val_unseen 的排序与校准都变差：
AUC / AP 下降，Brier / ECE 变差，mean gate score 上升快于 base rate。
```

这正面回答了 5.1 表 A：dev 与 `val_unseen` 存在 calibration drift。

### 4.2 表 B：gate threshold confusion

weighted ALL 的关键 frozen / sensitivity 点：

| split | gate | tau | changed | recovery | harm | net | recovered | harmed | harm/recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.90 | 0.01 | 3.93% | 0.90% | 0.53% | +0.38pp | 74 | 43 | 0.58 |
| `val_unseen` | 0.90 | 0.01 | 4.79% | 0.86% | 0.82% | +0.04pp | 87 | 83 | 0.95 |
| dev | 0.95 | 0.03 | 1.16% | 0.31% | 0.07% | +0.23pp | 25 | 6 | 0.24 |
| `val_unseen` | 0.95 | 0.03 | 1.14% | 0.21% | 0.21% | +0.00pp | 21 | 21 | 1.00 |

按 dataset 看，`val_unseen gate=0.90,tau=0.01` 的 net 并不一致：

| dataset | recovered | harmed | net | changed |
| --- | ---: | ---: | ---: | ---: |
| R2R | 13 | 9 | +0.17pp | 1.92% |
| REVERIE | 25 | 24 | +0.03pp | 3.38% |
| SOON | 30 | 37 | -0.21pp | 6.52% |
| CVDN | 19 | 13 | +0.66pp | 11.25% |

诊断判断：

```text
如果只看 harm <= 1pp 且 net > 0，0.90/0.01 在 val_unseen 仍勉强过线；
但它的 net 只有 +4 items，harm/recovered 从 dev 的 0.58 恶化到 0.95。

0.95/0.03 safety 更直接显示问题：dev 上 recovery 明显多于 harm，
val_unseen 上 recovery 与 harm 完全打平。
```

因此，表 B 的结论不是“找到了稳定可用阈值”，而是：

```text
当前 gate threshold 只能沿 recovery-harm trade-off 移动；
dev 上看似可用的分离，在 val_unseen 上退化为 recovery≈harm。
继续扫 threshold 没有研究价值。
```

### 4.3 表 C：changed item feature shift

表 C 比较 `recovered / harmed / changed_safe / changed_unrecovered` 的 trace feature。weighted ALL 中，最稳定的差异是：

| split | config | feature | recovered | harmed | harm - recovered | standardized effect |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| dev | `dev_best` | selected distance delta | 74 | 43 | +13.69m | +2.20 |
| `val_unseen` | `dev_best` | selected distance delta | 87 | 83 | +10.86m | +2.23 |
| `val_unseen` | `dev_best` | score margin over final | 87 | 83 | +0.075 | +0.39 |
| `val_unseen` | `safety` | selected distance delta | 21 | 21 | +11.81m | +2.27 |
| `val_unseen` | `safety` | score margin over final | 21 | 21 | +0.164 | +0.70 |

解释：

```text
harmed item 通常被改到离目标更远的位置；
这说明 ranker/gate 放行后，错误修改的代价很尖锐。

但 gate_score、loop、revisit、last-k 等 trace-only 特征没有形成一个
简单、跨 split、跨 dataset 都可靠的 recovered/harmed 分界。
```

表 C 的自动诊断是：

```text
changed feature separation = dataset-specific
```

这正面回答了 5.1 表 C：changed item 的 feature shift 存在，但不是一个足以继续 trace-only gate 调参的清晰可分结构。

### 4.4 表 D：dev vs val_unseen top1 ranker stability

final-bias decoupled ranker 的 top1 稳定性：

| split | should_rerank | top1 success | best-SPL top1 | best-SPL top3 | best-SPL median rank | better-SPL pair acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 976 | 48.87% | 32.38% | 66.91% | 2 | 74.77% |
| `val_unseen` | 1372 | 37.54% | 23.47% | 55.32% | 3 | 70.43% |
| drift | - | -11.34pp | -8.91pp | -11.59pp | +1 | -4.34pp |

诊断判断：

```text
4.6 final-bias 解耦确实改善了 dev 上的 ranker preference；
但这种 improvement 没有稳定外推到 val_unseen。

因此 4.7 断点不能全部归因于 gate。
ranker top1 / best-SPL preference 本身也存在 split drift。
```

这正面回答了 5.1 表 D：断点来自 both，而不是单纯 gate。

## 5. 结论

4.8 的结论是：

```text
当前 trace-only gate + ranker 框架的 val_unseen 失败，
不是某一个阈值没调好，而是 gate 和 ranker 两端都没有稳定外推。
```

更具体地说：

1. gate calibration 从 dev 到 `val_unseen` 变差，AUC / AP 下降，Brier / ECE 上升。
2. dev 上看似正收益的 threshold band，在 `val_unseen` 上退化为 recovery≈harm。
3. changed item 的 trace feature shift 是 dataset-specific，不是清晰可分边界。
4. final-bias decoupled ranker 的 top1 success 与 best-SPL preference 从 dev 到 `val_unseen` 明显下降。
5. 因此，继续扫 `gate_threshold / tau / loss weight / final bonus` 只会在同一条 recovery-harm 曲线上移动。

阶段 4.8 不改变 4.7 frozen 结论，只把断点定位得更清楚：

```text
ranker 职责解耦在 dev 上成立；
但 trace-only evidence 不足以让 gate 和 ranker 在 unseen split 上同时稳定。
```

## 6. 对 P1 新方向的指引

5.2 要求“选择一个新方向，不同时开多个”。4.8 对这个选择给出更明确指引。

### 6.1 推荐方向 A：evidence-augmented endpoint verification

4.8 支持优先选择方向 A：

```text
trace-only ranker 只产生 top-k endpoint；
新增 evidence verifier 判断候选 endpoint 是否真的满足 instruction / object / dialog goal；
gate 再根据 trace + verifier margin 决定是否修改 final。
```

原因：

| 4.8 证据 | 对方向 A 的含义 |
| --- | --- |
| gate score 在 `val_unseen` 过度乐观 | gate 不能只看 trace confidence，需要 verifier margin |
| harmed/recovered trace feature 不清晰可分 | 需要 instruction / object / room / dialog evidence 补充 |
| harmed item 常被改到更远点 | verifier 应判别“这个 endpoint 是否真的对齐目标”，而不是只看 ranker score |
| ranker top1 在 `val_unseen` 掉点 | ranker top-k 可以保留，但最终 top1 需要 evidence rerank / verify |

最小可执行版本建议：

| 模块 | 最小实现 |
| --- | --- |
| ranker | 保留 final-bias decoupled ranker 输出 top-k |
| verifier features | instruction keyword/entity overlap、room/object/landmark proxy、candidate metadata |
| CVDN 特例 | dialog evidence / question-answer keyword overlap |
| gate | 使用 `trace gate_score + verifier margin + final-vs-best evidence gap` |
| 决策 | 只在 verifier 对 non-final endpoint 明显胜过 final 时修改 |

这个方向正好针对 4.8 的核心瓶颈：trace-only evidence 不够。

### 6.2 不优先方向 B：multi-trajectory consensus

multi-trajectory / multi-sampling consensus 仍是合理备选，但不是当前 P1 首选。

4.8 对方向 B 的启发是：

```text
如果多个 trajectory 独立支持同一 endpoint cluster，
确实可能降低 single-trajectory false positive。
```

但它的成本更高：

1. 需要重新生成多轨迹或多采样结果。
2. 需要 endpoint clustering / consensus 规则。
3. 仍然可能缺少 instruction-object verification。

因此方向 B 更适合作为方向 A 之后的增强，而不是马上并行启动。

### 6.3 不优先方向 C：SPL-preserving / efficiency rerank

4.7 和 4.8 都说明 SPL 有小正收益，但 SR recovery 被 harm 抵消。

因此方向 C 当前不适合作为主线：

```text
SPL-preserving rerank 可以作为 verifier 通过后的 tie-breaker，
但不能单独作为下一阶段主方法。
```

原因是：

1. 4.7 `dev_best` 的 weighted dSPL 为正，但 dSR 只有约 +0.039pp。
2. 4.8 显示 harmed item 常被改到更远点，说明效率信号不能保证安全。
3. 论文主贡献需要 SR recovery，而不是仅靠很小 SPL gain 支撑。

## 7. 下一阶段最低观察指标

如果进入 evidence-augmented endpoint verification，建议最小报告：

| 指标 | 目的 |
| --- | --- |
| verifier top1 / top-k success on `should_rerank` | 看 evidence 是否补足 ranker top1 漂移 |
| verifier margin calibration by split / dataset | 避免复刻 gate calibration drift |
| final-vs-best evidence gap on recovered / harmed | 看是否真正分开 recovery/harm |
| weighted `val_unseen` dSR / dSPL | 主指标 |
| recovery / harm / net recovered items | 与 4.7 直接对齐 |
| max dataset harm | 防止 SOON / CVDN 局部爆 harm |

继续线沿用 5.3，但应明确超过 4.7 `dev_best`：

```text
val_unseen weighted dSR >= +0.20pp
weighted harm <= 1pp
max dataset harm <= 1pp
CVDN / SOON dSR 不明显为负
net recovered items >= 20
```

当前 4.7 `dev_best` 只有净 +4 items。下一阶段如果不能显著超过这个基线，就不应扩展成主实验。

## 8. 可写入总结的口径

英文口径：

```text
The endpoint oracle gap is not safely recoverable from single-trajectory trace evidence alone.
Our gate generalization diagnostics show that both the gate calibration and the ranker top-1
preference degrade from dev to val_unseen, collapsing the recovery-harm separation observed on dev.
```

中文口径：

```text
阶段 4.8 说明，4.7 的 val_unseen 断点不是单纯阈值问题；
gate 的校准外推和 ranker 的 top1 preference 外推都不稳定。
因此下一阶段应引入 endpoint evidence verification，
而不是继续在 trace-only gate + ranker 框架内扫参数。
```
