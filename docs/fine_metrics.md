# SAME Fine Metrics 说明

本文档说明 SAME 实验中的 `fine_metrics` 后处理产物。它读取已经生成的 `eval_items` sidecar，不重新运行模型，也不修改 `third_party/SAME`，用于生成适合逐样本错误分析、表格筛选和跨指标聚合的细粒度指标。

当前实现入口为：

```bash
python scripts/analysis/build_same_fine_metrics.py \
  --experiment-dir experiment_outputs/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1 \
  --connectivity-dir data/same/simulator/connectivity
```

如果实验目录是容器内 root 用户生成的，宿主机可能没有写权限。此时应在默认 SAME 容器中执行同一命令：

```bash
bash scripts/setup/run_container.sh
conda activate test-v1
python scripts/analysis/build_same_fine_metrics.py \
  --experiment-dir experiment_outputs/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1 \
  --connectivity-dir data/same/simulator/connectivity
```

## 1. 产物结构

脚本默认把结果写入实验目录下的 `fine_metrics/`：

```text
experiment_outputs/<experiment_id>/
  fine_metrics/
    manifest.json
    jsonl/
      R2R_val_unseen_fine_metrics.jsonl
      REVERIE_val_unseen_fine_metrics.jsonl
      CVDN_val_unseen_fine_metrics.jsonl
      SOON_val_unseen_fine_metrics.jsonl
    tables/
      fine_metrics_wide.csv
      fine_metrics_long.csv
      fine_metrics_summary.json
```

各文件用途：

- `manifest.json`：记录 schema、输入目录、connectivity 目录、输出文件列表、metric group 列表和行数。
- `jsonl/*_fine_metrics.jsonl`：每个 dataset/split 一个 JSONL 文件，每行一个样本，是最完整的逐样本产物。
- `tables/fine_metrics_wide.csv`：一行一个样本，把所有指标摊平成列，适合 Excel、pandas、R、DuckDB 直接筛选。
- `tables/fine_metrics_long.csv`：一行一个“样本-指标”，适合做跨指标聚合、透视和绘图。
- `tables/fine_metrics_summary.json`：按 dataset/split 汇总 item 数和 success 计数，用于快速 sanity check 和对齐官方 `metrics.json`。

`0011` 当前 v2 产物行数为：

```text
CVDN_val_unseen_fine_metrics.jsonl      907
R2R_val_unseen_fine_metrics.jsonl       2349
REVERIE_val_unseen_fine_metrics.jsonl   3521
SOON_val_unseen_fine_metrics.jsonl      3390
fine_metrics_wide.csv                   10167 data rows
fine_metrics_long.csv                   552873 data rows
```

## 2. JSONL schema

当前 schema 为 `same_fine_metrics.v2`。每行结构如下：

```json
{
  "schema_version": "same_fine_metrics.v2",
  "experiment_id": "0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1",
  "dataset": "CVDN",
  "split": "val_unseen",
  "identity": {
    "internal_item_id": "cvdn_1_6",
    "saved_instr_id": "cvdn_1_6",
    "source_ids": {}
  },
  "common": {},
  "eval_end_goal": {},
  "eval_end_region": {},
  "eval_end_region_threshold": {},
  "official": {}
}
```

五个 metric group：

- `common`：所有数据集共有的路径与指令指标。
- `eval_end_goal`：以单一导航终点为目标的指标。
- `eval_end_region`：以目标 viewpoint 集合作为目标，success 必须真实命中集合。
- `eval_end_region_threshold`：目标集合与 region 完全一致，但 success 用到最近目标的距离 `< success_threshold_m`。
- `official`：按 SAME evaluator 官方口径生成的对齐字段，并透出官方特有指标。

R2R 没有 region 目标，因此 `eval_end_region` 和 `eval_end_region_threshold` 在 JSONL 中均为 `null`，wide CSV 对应列为空，long CSV 不展开这两个 group。

## 3. common 指标

| 字段 | 类型 | 含义 | 计算方式 |
| --- | --- | --- | --- |
| `action_step_count` | `int` | 模型做了多少次高层导航选择，不包含 stop | 优先读 `official_item_scores.raw_same.action_steps`；缺失时用 `len(prediction.pred_path_segments) - 1` |
| `move_step_count` | `int` | 实际展开后沿拓扑图移动了多少次 | flatten 后 `prediction.trajectory` 中相邻 viewpoint 发生变化的次数 |
| `instruction_token_count` | `int` | 指令 token 数 | `annotation.instruction_meta.encoding_len`，也就是实际 BERT WordPiece encoding 长度，通常包含 `[CLS]` 和 `[SEP]` |
| `path_length_m` | `float` | 实际走过总路径长度，单位米 | 优先读 `official_item_scores.canonical.actual_length_m`；缺失时回退到 cumulative 或 edge length 求和 |
| `path_edge_count` | `int` | 实际走过路径所包含的边数 | 当前等于 `move_step_count` |

易错点：

- `action_step_count` 当前统一为“不包含 stop”。CVDN raw 缺失时，用 pred path segment 数量回退。
- `move_step_count` 不是 `len(trajectory) - 1`。如果 trajectory 中相邻 viewpoint 相同，原地不动不计入 move。
- `path_edge_count` 当前按 flatten trajectory 中“相邻 viewpoint 是否变化”计数，不是 shortest path 内部展开后的边数之和。

## 4. endpoint 通用字段

`eval_end_goal`、`eval_end_region`、`eval_end_region_threshold` 都使用同一组 endpoint 字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `final_success` | `bool` | 最终是否成功，具体判定由 group 决定 |
| `oracle_success` | `bool` | 轨迹中是否曾经成功，具体判定由 group 决定 |
| `final_distance_to_goal_m` | `float` | 最终位置到目标点或目标集合最近点的加权最短路径距离 |
| `final_distance_to_goal_edges` | `int` | 上述加权最短路径包含的边数 |
| `path_length_ratio` | `float` | `common.path_length_m / shortest_path_length_m` |
| `oracle_path_length_m` | `float/null` | 如果第一次成功时就停止，已走路径长度 |
| `oracle_path_edge_count` | `int/null` | 如果第一次成功时就停止，已走 move 边数 |
| `oracle_path_length_ratio` | `float/null` | `oracle_path_length_m / shortest_path_length_m` |
| `shortest_path_length_m` | `float` | 起点到目标点或目标集合最近点的加权最短路径长度 |
| `shortest_path_edge_count` | `int` | 上述加权最短路径包含的边数 |

通用实现点：

- 成功阈值来自对应 `eval_context.json` 的 `run_context.success_threshold_m`，当前通常是 `3.0` 米。
- 图距离读取 `data/same/simulator/connectivity/*_connectivity.json`，用 pose 欧氏距离作为边权，再用 Dijkstra 计算。
- `shortest_path_edge_count` 是“加权最短路径”对应路径的边数，不是拓扑边数最少路径的边数。
- 如果 `shortest_path_length_m` 为 0，ratio 写 `null`，避免除零。

## 5. eval_end_goal

`eval_end_goal` 以单一导航终点作为目标：

- R2R：`annotation.nav_goal_viewpoint`
- CVDN：`annotation.nav_goal_viewpoint`
- REVERIE：`annotation.nav_goal_viewpoint`
- SOON：`annotation.gt_path[-1]`

success 判定：

- `final_success = final_distance_to_goal_m < success_threshold_m`
- `oracle_success = trajectory 中第一次到 goal 的距离 < success_threshold_m`

米制距离优先复用 `primitives.distance_to_nav_goal_by_step_m` 和 `canonical.shortest_path_length_m`。边数需要通过 connectivity graph 重新计算。

易错点：

- REVERIE 官方 success 用 object-visible viewpoint 集合；但 `eval_end_goal` 明确使用 `nav_goal_viewpoint`，所以不要直接复用 `canonical.final_success`。
- SOON 的 goal 口径使用 `gt_path[-1]`，这是 SAME 官方导航评估使用的 `goal_vp`。

## 6. eval_end_region

`eval_end_region` 以目标 viewpoint 集合作为目标：

- CVDN：`dataset_extras.cvdn.end_panos`
- REVERIE：`annotation.success_target_viewpoints`
- SOON：`dataset_extras.soon.bbox_viewpoints`
- R2R：无 region，写 `null`

success 判定：

- `final_success = final_viewpoint in targets`
- `oracle_success = trajectory 中曾经有 viewpoint in targets`

易错点：

- region success 是“命中集合”，不是 `< 3m`。只有当前或曾经的 viewpoint 本身在目标集合中，success 才为 true。
- 距离指标仍然是到目标集合最近点的图上加权最短距离；因此距离可以小于 3m，但 `eval_end_region.final_success` 仍为 false。
- `oracle_path_*` 取第一次命中集合的 step。如果从未命中集合，写 `null`。
- CVDN 的 `eval_end_goal` 使用 trusted-path 的 `nav_goal_viewpoint`；`eval_end_region` 使用原始可接受的 `end_panos` 集合。这两者可能不同。

## 7. eval_end_region_threshold

`eval_end_region_threshold` 的目标集合与 `eval_end_region` 完全一致，但 success 判定放宽为距离阈值：

- `final_success = final_distance_to_nearest_target < success_threshold_m`
- `oracle_success = trajectory 中第一次 distance_to_nearest_target < success_threshold_m`

距离优先复用 `eval_items.primitives` 中的集合最近距离：

- CVDN：`distance_to_nearest_end_pano_by_step_m`
- REVERIE：`distance_to_nearest_success_target_by_step_m`
- SOON：`distance_to_nearest_bbox_viewpoint_by_step_m`

这些 primitive 是“对目标集合全体取最近距离”，不是固定到集合中的某一个点。若 primitive 缺失，脚本会用 connectivity graph 对每个 trajectory viewpoint 到 targets 做最近距离回退。

易错点：

- `eval_end_region_threshold` 和 `eval_end_region` 只差 success 判定，不差目标集合。
- 因为 `< 3m` 更宽松，`region_threshold_*_successes` 通常应大于或等于 `region_*_successes`。
- `oracle_path_*` 取第一次满足距离阈值的 step，而不是第一次命中集合的 step。

## 8. official

`official` 用于和 SAME evaluator 的 `metrics.json` 对齐，同时保留官方特有字段。缺失字段写 `null`，CSV 中为空。

通用官方对齐字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `final_success` | `bool/null` | 官方 per-item success |
| `oracle_success` | `bool/null` | 官方 per-item oracle success |
| `final_distance_to_goal_m` | `float/null` | 官方 nav error / final distance |
| `oracle_distance_to_goal_m` | `float/null` | 官方 oracle error / min distance |
| `final_distance_to_goal_edges` | `int/null` | 最终位置到官方距离目标点的加权最短路径边数 |
| `path_length_m` | `float/null` | 官方 trajectory length |
| `path_length_ratio` | `float/null` | `path_length_m / shortest_path_length_m` |
| `oracle_path_length_m` | `float/null` | 按官方 oracle success 口径第一次成功时的路径长度 |
| `oracle_path_edge_count` | `int/null` | 按官方 oracle success 口径第一次成功时的 move 边数 |
| `oracle_path_length_ratio` | `float/null` | `oracle_path_length_m / shortest_path_length_m` |
| `shortest_path_length_m` | `float/null` | 官方 shortest path length |
| `shortest_path_edge_count` | `int/null` | 官方距离目标点对应的起点最短路径边数 |
| `spl` | `float/null` | 官方 SPL |

官方特有字段：

| 字段 | 主要来源 |
| --- | --- |
| `oracle_plan_success` | `raw_same.oracle_plan_errors < success_threshold_m`，主要用于 CVDN |
| `oracle_plan_distance_m` | `raw_same.oracle_plan_errors` |
| `dist_to_end_reduction_m` | `raw_same.dist_to_end_reductions` |
| `rgs` / `rgspl` | REVERIE grounding 指标 |
| `det_success` / `det_spl` | SOON detection 指标 |
| `goal_progress_m` | SOON goal progress |
| `heading_error` / `elevation_error` / `point_det_error` | SOON object direction / point detection 误差 |

官方口径差异：

- R2R：success 和 distance 都使用 `gt_path[-1]` / `nav_goal_viewpoint` 的 `< 3m` 导航口径。
- CVDN：官方 `sr/nav_error/spl/oracle_sr` 使用 `nav_goal_viewpoint`；`oracle_plan_success` 单独由 `oracle_plan_errors < threshold` 派生。
- REVERIE：官方 `success/oracle_success` 使用 `success_target_viewpoints` 命中集合；官方 `nav_error/oracle_error` 距离仍使用 `nav_goal_viewpoint`。
- SOON：官方 `success/oracle_success/nav_error/oracle_error` 使用 `goal_vp = gt_path[-1]`；检测相关字段从 raw SAME 透出。

## 9. CSV 表

### 9.1 fine_metrics_wide.csv

wide 表一行一个样本。列名用点号展开：

```text
experiment_id,dataset,split,internal_item_id,saved_instr_id,
common.action_step_count,
eval_end_goal.final_success,
eval_end_region.final_success,
eval_end_region_threshold.final_success,
official.final_success,
official.oracle_plan_success,
...
```

布尔值写为 `true/false`，空值写为空字符串。

### 9.2 fine_metrics_long.csv

long 表一行一个“样本-指标”：

```text
experiment_id,dataset,split,internal_item_id,metric_group,metric_name,value_num,value_bool,value_type
```

字段规则：

- `metric_group`：`common`、`eval_end_goal`、`eval_end_region`、`eval_end_region_threshold`、`official`。
- `metric_name`：指标名，例如 `action_step_count`、`final_success`、`oracle_plan_success`。
- `value_num`：数值型指标写这里。
- `value_bool`：布尔型指标写这里，取值 `true/false`。
- `value_type`：`num`、`bool`、`null` 或 `str`。

R2R 的 region / region threshold group 为 `null`，因此 long 表不会为 R2R 生成这两组指标行。

## 10. 代码实现关键点

核心实现位于 `scripts/analysis/build_same_fine_metrics.py`：

- `GROUP_METRICS` 是统一 group 注册表，wide、long、manifest 都从这里展开，避免写死三组指标。
- `discover_eval_item_sources()` 扫描 `eval_items/*_eval_context.json`，确定 dataset、split、success threshold 和配套 JSONL。
- `Graph.from_connectivity_file()` 读取 MatterSim connectivity JSON，用 `pose[3]`、`pose[7]`、`pose[11]` 计算边权，避免依赖 `networkx`。
- `Graph.shortest()` 用标准库 `heapq` 实现 Dijkstra，同时返回最短米数和该加权最短路径的边数。
- `build_common_metrics()` 生成共有指标，并处理 CVDN 缺少 raw `action_steps` 的 fallback。
- `build_goal_metrics()` 生成单点 goal 指标。
- `build_region_metrics()` 生成集合命中 region 指标。
- `build_region_threshold_metrics()` 生成集合距离阈值 region 指标，优先复用 primitives 中的集合最近距离。
- `build_official_metrics()` 生成与官方 `metrics.json` 对齐的指标，并透出 CVDN/REVERIE/SOON 的官方特有字段。
- `build_summary()` 汇总 goal、region、region threshold、official success 计数；CVDN 的 `official_oracle_plan_successes` 可用于对齐 `oracle_path_success_rate`。

当前测试位于 `tests/test_build_same_fine_metrics.py`，覆盖：

- v2 schema 和 manifest group 列表。
- 小型 connectivity graph 上的 common、goal、region、region threshold、official 指标。
- `action_step_count` 有 raw 时用 raw，缺失时用 `len(pred_path_segments) - 1`。
- “未命中 region target 但距离 target < 3m”的样本：region false，region threshold true。
- REVERIE official success 使用 region 命中，official distance 使用 nav goal。
- CVDN `official.oracle_plan_success` 由 `oracle_plan_errors < threshold` 派生。
- wide/long CSV 的列名、布尔值和空值写法。
- 对真实 `0011` 产物做只读 smoke test，确认行数、R2R region 为空、官方 SR/OSR 与 `metrics.json` 对齐、CVDN planner oracle 与 `oracle_path_success_rate` 对齐。

## 11. 与 eval_items 的关系

`eval_items` 是稳定的样本级溯源材料，`fine_metrics` 是面向当前分析问题的派生表。推荐流程是：

1. SAME 完整评估生成 `eval_items`。
2. 使用 `scripts/analysis/build_same_fine_metrics.py` 从 `eval_items` 派生 `fine_metrics`。
3. 后续错误分析、分桶、可视化优先读取 `fine_metrics/jsonl` 或 `fine_metrics/tables`。

当未来需要新增实验性字段时，优先扩展父项目后处理脚本；只有当 `eval_items` 缺少无法恢复的原始信息时，才考虑给 SAME 子项目增加 patch。
