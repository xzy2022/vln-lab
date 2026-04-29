# VLN Endpoint Completion Optimization 路线说明

本文档记录当前项目中围绕 SAME endpoint / STOP 错误修正的研究路线、已完成阶段的代码入口、实验产物和阶段性结论。核心目标不是直接做 full RL，而是先用诊断、无训练上界、规则 reranking 和离线 endpoint learning，判断 SAME 的可恢复失败是否能以低成本被修正。

当前主线可以命名为：

```text
Evidence-Adaptive Completion Optimization for VLN
```

更具体地说：

```text
用 episode-level diagnostics、endpoint preference 和 category-aware routing，
在不重训主导航器的情况下减少 SAME 的 recoverable failure。
```

## 1. 总体路线

当前建议按 5 个阶段推进：

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| 1. oracle gap 报告 | 统计 final success、oracle success、best oracle step、overshoot、stop-too-early | 已实现 |
| 2. 无训练 endpoint 上界 | 从 SAME 访问过的 viewpoint 中选择离目标最近的 endpoint，估计可恢复空间 | 已实现 |
| 3. heuristic reranker | 只用 SAME 自身 trace 特征做规则 endpoint rerank，测试无训练规则能否直接吃掉 gap | 已实现 |
| 4. 离线 endpoint learning 闭环 | 固定 candidate / label / pair / evaluator 协议，在 train/dev 上学习并冻结 gate、ranker、tau，一次性报告 val_unseen | 4.1 数据与 evaluator 已实现；4.2 gate baseline 下一步 |
| 5. NavNuances router | 根据 RR/VM/DC/LR/NU 等类别选择 SAME-only、rerank 或少量 LLM/VLM assist | 后续 |

阶段 1-3 的作用是回答：

1. SAME 是否存在 final STOP / endpoint 可恢复失败？
2. 如果允许在访问过的 endpoint 中选择，理论上界有多大？
3. 不训练模型、只靠规则能否实际 recover 一部分 oracle gap？

当前结论是：

```text
有可恢复空间；规则有信号但泛化不稳；第 4 步需要 learned gate + ranker。
```

## 2. 当前使用的 R2R trace 实验

第三步需要 `decision_trace`。当前用于 R2R 原型的实验为：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
```

该实验包含：

```text
eval_items.v3
prediction.decision_trace
fine_metrics/
oracle_gap_for_rl_research/
endpoint_heuristic_rerank/
```

注意：早期多任务实验 `0013_same_val_all_r2r_reverie_cvdn_soon_same_s0_v4` 的 `eval_items` 是 v2，没有 `decision_trace`，适合做 oracle gap 和 endpoint 上界，但不适合做 STOP heuristic / learned gate-ranker。

## 3. 阶段 1：oracle gap 报告

### 3.1 代码入口

```bash
python scripts/analysis/build_oracle_gap_report.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

当前 R2R trace 实验对应产物：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/oracle_gap_for_rl_research/
  manifest.json
  oracle_gap_items.csv
  oracle_gap_summary.csv
  oracle_gap_report.md
```

### 3.2 计算内容

脚本读取：

```text
experiment_outputs/<experiment_id>/eval_items/
experiment_outputs/<experiment_id>/fine_metrics/tables/fine_metrics_wide.csv
```

主要统计：

| 字段 | 含义 |
| --- | --- |
| `final_success` | SAME 原始 final endpoint 是否成功 |
| `oracle_success` | 轨迹中是否曾经进入成功区域 |
| `first_success_step` | 第一次成功的 step |
| `best_distance_step` | 距离目标最近的访问 step |
| `overshoot` | final 失败，但轨迹中曾经成功 |
| `stop_too_early_proxy` | final 失败、从未成功，且 final 已是最近点 |
| `never_reached` | final 失败、从未成功，且 final 不是最近点 |

### 3.3 当前结果

`0014` 的 R2R official scope：

| split | items | final SR | oracle SR | oracle gap | overshoot | stop-too-early proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val_train_seen | 1501 | 88.74 | 94.34 | 5.60 | 5.60 | 1.20 |
| val_unseen | 2349 | 76.29 | 84.80 | 8.51 | 8.51 | 3.36 |

阶段结论：

```text
R2R val_unseen 上存在 8.51pp 的 oracle gap。
这些 gap 在当前口径下全部表现为 overshoot / pass-but-not-stop：
轨迹到过成功区域，但 final endpoint 没停在成功区域。
```

这说明 endpoint / STOP correction 是值得继续做的，不是一个没有空间的问题。

## 4. 阶段 2：无训练 endpoint 上界

### 4.1 代码入口

阶段 2 已经合并在 oracle gap 报告脚本中：

```bash
python scripts/analysis/build_oracle_gap_report.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `nearest_endpoint_success` | 选择访问过的最近目标 endpoint 后是否成功 |
| `nearest_endpoint_spl` | 选择访问过的最近目标 endpoint 后的 SPL |
| `recovered_by_nearest_endpoint` | final 失败但 nearest endpoint 成功 |

这里的 nearest endpoint 是 oracle-style 上界，选择时使用 GT distance；它不能作为真实方法，只用于判断空间。

### 4.2 当前结果

`0014` 的 R2R official scope：

| split | final SR | nearest SR | final SPL | nearest SPL |
| --- | ---: | ---: | ---: | ---: |
| val_train_seen | 88.74 | 94.34 | 85.01 | 92.00 |
| val_unseen | 76.29 | 84.80 | 66.24 | 75.81 |

阶段结论：

```text
如果能从 SAME 已访问过的 endpoint 中选对终点，
R2R val_unseen 理论上可从 76.29 SR 提升到 84.80 SR。
```

这验证了第 4 步 offline endpoint learning 的目标是合理的：不需要重新规划整条轨迹，仅在访问过的 endpoint 中做选择，就存在明显可恢复空间。

## 5. 阶段 3：heuristic endpoint reranker

### 5.1 代码入口

```bash
python scripts/analysis/build_endpoint_heuristic_report.py \
  --experiment-dir experiment_outputs/<experiment_id>
```

当前 R2R trace 实验对应产物：

```text
experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5/endpoint_heuristic_rerank/
  manifest.json
  endpoint_heuristic_items.csv
  endpoint_heuristic_summary.csv
  endpoint_heuristic_report.md
```

如果实验目录由容器 root 用户生成，宿主机没有写权限，可临时指定输出目录：

```bash
python scripts/analysis/build_endpoint_heuristic_report.py \
  --experiment-dir experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5 \
  --output-dir /tmp/endpoint_heuristic_rerank_0014
```

### 5.2 规则约束

heuristic 选择 endpoint 时只能使用 SAME 自身输出：

```text
trajectory
decision_trace.steps[*].stop_prob
selected.prob
selected.selection_kind
candidate probability / margin
MoE router entropy
loop / revisit / backtrack
step position
```

不能用于选择：

```text
distance_to_nav_goal_by_step_m
final_success
oracle_success
gt_path
nav_goal_viewpoint
success_target_viewpoints
```

GT distance 和 success label 只允许在选择 endpoint 之后用于评估。

### 5.3 已实现 heuristic

当前脚本覆盖以下规则族：

| heuristic | 作用 |
| --- | --- |
| `final` | 原始 SAME final endpoint |
| `max_stop_prob` | 选全轨迹 STOP 概率最高点 |
| `max_stop_margin` | 选 STOP 相对 move margin 最高点 |
| `last_k_max_stop` | 只在最后 K 步中选 STOP 概率最高点 |
| `last_k_max_stop_margin` | last-k 的 margin 版本 |
| `first_stop_threshold` | 第一次 STOP 概率超过阈值就停 |
| `last_stop_threshold` | 最后一次 STOP 概率超过阈值的位置 |
| `last_high_stop_before_move` | 某步 STOP 高但模型继续走时，尝试回退 |
| `loop_guard` | 出现 revisit / loop / backtrack 时回退到 loop 附近 STOP 高点 |
| `conservative_rerank` | gate-like 的保守组合规则 |

### 5.4 当前结果

`0014` 的 R2R val_unseen official scope：

| heuristic | params | heur SR | delta SR | recovery | harm | changed | heur SPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `final` |  | 76.29 | +0.00 | 0.00 | 0.00 | 0.00 | 66.19 |
| `loop_guard` | `threshold=0.2,window=10` | 76.20 | -0.09 | 0.64 | 0.72 | 3.87 | 66.78 |
| `loop_guard` | `threshold=0.1,window=10` | 76.03 | -0.26 | 1.06 | 1.32 | 5.92 | 67.11 |
| `last_high_stop_before_move` | `threshold=0.2` | 75.05 | -1.23 | 2.00 | 3.24 | 21.80 | 67.15 |

`0014` 的 R2R val_train_seen official scope：

| heuristic | params | heur SR | delta SR | recovery | harm | changed | heur SPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_stop_threshold` | `threshold=0.2` | 89.47 | +0.73 | 2.13 | 1.40 | 21.39 | 87.12 |
| `last_high_stop_before_move` | `threshold=0.2` | 89.47 | +0.73 | 1.73 | 1.00 | 18.12 | 86.86 |
| `loop_guard` | `threshold=0.1,window=10` | 89.27 | +0.53 | 0.87 | 0.33 | 4.20 | 86.05 |

更细地看 val_unseen：

```text
loop_guard threshold=0.2:
  recover 15 个失败样本
  harm 17 个原本成功样本

last_high_stop_before_move threshold=0.2:
  recover 47 个失败样本
  harm 76 个原本成功样本
```

阶段结论：

```text
规则确实能 recover 一部分 overshoot，但 harm 通常更大。
val_train_seen 上简单规则有正收益；val_unseen 上收益不泛化。
```

这说明 SAME trace 中存在 endpoint completion 信号，但固定阈值无法可靠区分：

```text
什么时候应该改 final
什么时候必须保持 final
```

因此第 3 步不是最终方法，而是第 4 步 learned gate / ranker 的动机。

## 6. 阶段 4：离线 endpoint learning 闭环

阶段 4 统一命名为：

```text
离线 endpoint learning 闭环
```

旧称 `轻量 endpoint scorer / preference loss` 不再作为阶段名使用。它现在只是 4.3 / 4.4 中模型与 loss 设计的一部分。

阶段 4 的目标重新定义为：

```text
用固定协议构造 endpoint candidate group，
在 train/dev 上学习和冻结 gate、ranker、tau，
然后一次性报告 val_unseen，
验证 learned endpoint preference 是否比 rule-only heuristic 更稳。
```

核心边界：

```text
不重训 SAME 主导航器
不使用在线 RL rollout
不在 val_unseen 上调 threshold / tau
不把 4.1 smoke score 当成 learned method
```

### 6.1 统一子阶段

| 子阶段 | 目标 | 产物 | 是否训练 |
| --- | --- | --- | --- |
| 4.1 数据与评测闭环 | 固定 candidate / label / pair / tau-curve 协议 | dataset builder + evaluator | 否 |
| 4.2 gate-only baseline | 学“何时不要动 final” | gate baseline report | 只训 gate |
| 4.3 pairwise ranker 主实验 | 学候选 endpoint 排序 | ranker + inference report | 是 |
| 4.4 loss ablation | 比 CE / pairwise / DPO / GRPO | ablation table | 是 |

### 6.2 参数分类与冻结原则

协议参数在 4.1 后冻结：

```text
candidate = all trajectory steps
should_rerank = final_fail && oracle_success
pair types = success>fail, better_SPL>worse_SPL, final_success final>bad_nonfinal
target_scope = official
```

训练参数只在 train/dev 上确定：

```text
model type
hidden dim
learning rate
loss weights
class weights
pair sampling ratio
```

推理参数只在 dev 上确定：

```text
gate_threshold
tau
allow_change_final
top-k candidate restriction
```

重要原则：

```text
不要在 val_unseen 上看结果后再改 gate_threshold 或 tau。
否则第 4 步会变成调榜，而不是可信的研究实验。
```

当前 `0014` R2R 原型采用的离线划分是：

```text
train: val_train_seen 内部 hash split 80%
dev: val_train_seen 内部 hash split 20%
test/report: val_unseen，只跑一次冻结配置
```

更正式版本再补：

```text
train on R2R train
tune on val_seen
report on val_unseen
```

### 6.3 4.1 数据与评测闭环：数据协议

当前 4.1 的 dataset builder 已实现，代码入口：

```bash
bash scripts/setup/run_container.sh
conda activate test-v1
python scripts/analysis/build_endpoint_candidate_groups.py \
  --experiment-dir experiment_outputs/<experiment_id> \
  --target-scopes official
```

默认输出目录：

```text
experiment_outputs/<experiment_id>/endpoint_learning/
  candidate_groups/
    episode_groups.csv
    endpoint_candidates.csv
  preference_pairs/
    preference_pairs.csv
  eval_protocol/
    protocol.json
    eval_protocol_summary.csv
  manifest.json
```

每条 episode 构造一个 candidate group：

```text
episode e:
  candidates = step 0, step 1, ..., final step
```

当前协议保留 step-level candidate，不按 viewpoint 去重。理由是同一 viewpoint 在不同 step 出现时，path length、loop context、route context 和 STOP trace 不同，对 SPL 与 reranking 决策都有意义。

candidate row 保存 identifier、metadata 与可用特征：

```text
episode_id
candidate_step
viewpoint
has_decision_trace
decision_trace_index
is_route_intermediate
is_route_expanded_without_decision
route_step_offset
route_step_count
is_final
step_frac
path_length_m
stop_prob
stop_margin_prob
selected_prob
top1_top2_margin
moe_router_entropy
fuse_weight
is_revisit
is_loop_region
is_last_k
```

其中 `has_decision_trace` / `is_route_expanded_without_decision` 是后续训练必须使用的 mask 字段。

label / reward 字段可以保存，但训练和推理特征不能读：

```text
success_label
spl_at_candidate
distance_to_goal_m
reward
is_best_success_candidate
is_nearest_candidate
```

`should_rerank` 固定为：

```text
should_rerank = final_success == false AND oracle_success == true
```

也就是只把 pass-but-not-stop / overshoot 当成 gate 正例。`final_success`、`never_reached`、`stop_too_early_proxy` 都是 gate 负例。

pairwise preference pairs 固定三类：

```text
success candidate > failed candidate
higher-SPL success candidate > lower-SPL success candidate
final candidate > nonfinal failed candidate, when final_success == true
```

第三类非常重要，专门训练模型不要乱改原本成功的 final endpoint。

当前 `0014_same_val_r2r_eval_only_same_s0_v5` 的 R2R official 产物统计：

| source split | protocol split | episodes | candidates | pairs | should_rerank |
| --- | --- | ---: | ---: | ---: | ---: |
| `val_train_seen` | `train` | 1204 | 8156 | 19649 | 68 / 5.65% |
| `val_train_seen` | `dev` | 297 | 1985 | 3916 | 16 / 5.39% |
| `val_unseen` | `test` | 2349 | 18926 | 44962 | 200 / 8.51% |

当前 pair 构成：

| source split | protocol split | success>fail | better-SPL success>success | final-success final>bad-nonfinal |
| --- | --- | ---: | ---: | ---: |
| `val_train_seen` | `train` | 13334 | 2052 | 4263 |
| `val_train_seen` | `dev` | 2669 | 252 | 995 |
| `val_unseen` | `test` | 31345 | 4843 | 8774 |

关于 trace 特征缺失的提醒：

```text
val_unseen:
  candidates = 18926
  candidates with decision_trace = 16647 / 87.96%
  route_expanded_without_decision = 2279 / 12.04%
```

这不是命令没执行成功，也不是普通脏缺失。原因是当前协议把 `trajectory` 的所有 step 都作为 candidate，而 SAME 的 `decision_trace` 只记录真正做决策的 step；当一次决策的 `route_viewpoints` 展开成多个 trajectory step 时，route 中间点没有独立的 STOP / MoE / fuse decision 特征。

因此：

```text
trace 特征空值仍会存在；
但它们已被 has_decision_trace=false 和 is_route_expanded_without_decision=true 显式标注；
训练时不能把这些空值静默当 0，必须使用 mask / imputation。
```

如果未来想让 trace 特征缺失率变成 0%，需要改协议为 `candidate = decision_trace step`，但这会丢掉 route 中间 endpoint，和当前 `all trajectory steps as candidates` 的协议不同。当前建议保留现有协议，并在 4.2 / 4.3 的特征预处理中显式处理 missing mask。

### 6.3.1 4.1 数据与评测闭环：evaluator

已新增 endpoint score 评测入口：

```bash
python scripts/analysis/eval_endpoint_reranker.py \
  --experiment-dir experiment_outputs/<experiment_id> \
  --score-csv path/to/endpoint_scores.csv \
  --target-scope official \
  --split dev \
  --gate-thresholds 0.3,0.5,0.7 \
  --taus 0,0.05,0.1 \
  --allow-change-final true,false
```

最小 score CSV：

```text
candidate_id,candidate_score,gate_score
R2R:val_train_seen:official:r2r_3100_0:step_000,0.12,0.83
```

其中 `gate_score` 可选；缺失时默认用 `--default-gate-score 1.0`，即 gate 总是通过。`candidate_score` 也可以写成 `score`、`endpoint_score`、`logit` 或 `model_score`。如果不写 `candidate_id`，也支持 `episode_id + candidate_step`。

推理逻辑固定为：

```text
if gate_score < gate_threshold:
    choose final
else:
    best = argmax candidate_score
    if best_score <= final_score + tau:
        choose final
    else:
        choose best
```

输出位置默认在：

```text
experiment_outputs/<experiment_id>/endpoint_learning/eval_protocol/
  endpoint_learning_items.csv
  endpoint_learning_summary.csv
  tau_curve.csv
  recovery_harm_curve.csv
  endpoint_reranker_eval_manifest.json
```

核心指标：

```text
SR
SPL
delta_SR
gap_capture_rate
recovery_rate
harm_rate
changed_endpoint_rate
gate_precision
gate_recall
gate_auc
overshoot_recovery_rate
final_success_harm_rate
```

`--split dev` 会筛选 `protocol_split=dev`；`--split val_unseen` 会筛选原始数据 split。这样 train/dev 上可以调 `gate_threshold/tau`，`val_unseen` 只用于冻结配置的一次性报告。

当前这一步仍属于 4.1，而不是 4.2。它不训练 gate，也不学习 ranker；它只定义“给定一份 candidate/gate score，如何按冻结推理规则评估 endpoint 选择”的统一出口。后续 4.2 logistic gate 和 4.3 pairwise ranker 都应该把自己的分数写成同一份 score CSV，再交给这个 evaluator 产出 tau curve 和 recovery-harm curve。

用 `stop_prob` 临时生成的 smoke score 只用于验证闭环是否跑通，不代表 learned method。当前观察是：单独用 SAME 的 `stop_prob` 作为 `candidate_score`，dev 和 val_unseen 的 `delta_SR/gap_capture_rate/recovery_rate` 基本为 0，说明它不能直接解决 overshoot recovery；这正是继续做 learned gate 的动机。

### 6.4 4.2 gate-only baseline

当前状态：尚未实现训练脚本。4.2 要新增的是一个真正产出 `gate_score` 的 episode-level baseline，例如：

```text
scripts/analysis/train_endpoint_gate_baseline.py
```

它和 6.3.1 的关系是：

```text
4.2 gate-only baseline 负责学习 gate_score；
6.3.1 eval_endpoint_reranker.py 负责消费 gate_score 并评估 threshold/tau curve。
```

gate-only baseline 只回答：

```text
这个 episode 是否值得改 final？
```

gate label 固定为：

```text
1 = final_fail && oracle_success
0 = otherwise
```

episode-level features 可由 candidate group 聚合：

```text
final_stop_prob
max_stop_prob
max_stop_minus_final_stop
last_k_max_stop
loop_count
revisit_count
path_len
action_step_count
avg_moe_entropy
final_step_frac
```

第一版模型建议：

```text
LogisticRegression
```

原因是解释性强、调参少，适合先验证 gate 是否可学。RandomForest、XGBoost 或 small MLP 可以作为后续可选模型，但不作为第一版主口径。

gate-only inference 分两个 selector：

1. oracle-like selector：gate 通过后选择 nearest / best success candidate，只测 gate 上界，不能作为真实方法。
2. non-oracle selector：gate 通过后接 `loop_guard` 或 `last_k_max_stop`，和第 3 步规则对齐。

4.2 的成功标准不是 SR，而是：

```text
gate precision 高
harm_rate 可控
tau/gate threshold curve 上存在低 harm 区间
```

如果 gate 连 overshoot 都分不出来，4.3 就不该贸然做 ranker 主实验。

### 6.5 4.3 pairwise ranker 主实验

主实验建议结构：

```text
candidate_features -> candidate_score
episode_features -> gate_score
```

第一版模型：

```text
tabular MLP
2 hidden layers
hidden dim 64 or 128
dropout 0.1
```

loss：

```text
L = L_pairwise + lambda_gate * L_gate + lambda_final * L_final_stay
```

其中：

```text
L_pairwise = -log sigmoid(score_chosen - score_rejected)
L_gate = BCE(gate_score, should_rerank)
L_final_stay = final_success 时惩罚 nonfinal_score > final_score
```

推理固定为：

```text
if gate_score < gate_threshold:
    choose final
else:
    best = argmax candidate_score
    if best_score <= final_score + tau:
        choose final
    else:
        choose best
```

训练和调参只用 train/dev：

```text
train: val_train_seen 内部 hash split 80%
dev: val_train_seen 内部 hash split 20%
test/report: val_unseen，只跑一次冻结配置
```

报告必须包含 tau curve：

```text
SR
SPL
delta_SR
gap_capture_rate
recovery_rate
harm_rate
changed_endpoint_rate
overshoot_recovery_rate
final_success_harm_rate
```

### 6.6 4.4 loss ablation

建议在 pairwise 主实验闭环稳定后再做 loss ablation：

| loss | 说明 |
| --- | --- |
| CE | `BCE(candidate_score, success_label)` |
| Pairwise | `-log sigmoid(score_chosen - score_rejected)` |
| DPO-style | 用 SAME STOP 分数作为 reference policy |
| GRPO-style | episode 内 group-relative reward，不引入 critic |

DPO-style：

```text
ref_i = logit(stop_prob_i)
loss = -log sigmoid(beta * ((s_chosen - s_rejected) - (ref_chosen - ref_rejected)))
```

GRPO-style：

```text
R_i = success_i * (1 + alpha * SPL_i)
A_i = (R_i - mean(R)) / std(R)
pi_i = softmax(score_i)
loss = -sum_i A_i * log pi_i + KL(pi || pi_ref)
```

如果一个 episode 没有任何成功 candidate，则默认 final candidate 是 preferred / stay target，避免模型在 never-reached 样本上幻想式 rerank。

### 6.7 实施顺序

当前阶段按下面顺序推进：

1. 完成 `build_endpoint_candidate_groups.py`，生成 candidate 和 pair 数据。
2. 完成 `eval_endpoint_reranker.py`，支持从 score CSV 评估 tau curve。
3. 做 gate-only logistic baseline。
4. 做 pairwise MLP 主实验。
5. 最后补 CE / DPO / GRPO ablation。

第 4 步真正要拿到的不是某个 fancy loss 名字，而是一张可信的冻结协议表：rule-only 在 unseen 伤害更大，learned gate + ranker 在冻结阈值下能把 harm 压住，并吃掉一部分 oracle gap。

## 7. 阶段 5：NavNuances router

阶段 5 不应早于离线 endpoint learning 闭环。它的目标是把 completion optimization 和细粒度类别诊断结合起来：

```text
RR / VM：低成本 SAME-only
DC / LR / NU：高风险时启用 rich history、endpoint rerank 或少量 LLM/VLM assist
```

这里的关键不是每步调用 VLM，而是做 input-adaptive routing：

```text
先判断样本类别和风险
再决定是否增加计算预算
```

阶段 5 的前提是第 4 步已经有一个可控 harm 的 reranker，否则 router 很难判断何时启用 endpoint correction。

## 8. 当前阶段性总判断

阶段 1 和 2 给出的正信号：

```text
R2R val_unseen final SR = 76.29
R2R val_unseen nearest endpoint SR = 84.80
可恢复 oracle gap = 8.51pp
```

阶段 3 给出的负信号：

```text
固定 heuristic 在 val_unseen 上不能稳定提升 SR。
它能 recover overshoot，但也会 harm 原本成功的 final endpoint。
```

因此下一步不应继续堆规则，而应进入：

```text
gate + learned endpoint ranker + final-stay threshold / tau curve
```

第 4 步的最低成功标准：

```text
val_unseen delta SR > 0
harm_rate <= 1pp
SPL 不下降
```

更强标准：

```text
recover >= 10% oracle gap
R2R val_unseen SR 提升 >= 0.85pp
harm_rate <= 1pp
```

如果第 4 步能达到宽松标准，就可以扩展到多任务 trace 实验；如果能达到强标准，就可以把 pairwise / DPO / GRPO-style preference optimization 作为论文方法核心。
