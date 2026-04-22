# SAME Fine Metrics 说明

本文档说明 SAME 实验中的 `fine_metrics` 后处理产物。它读取已经生成的 `eval_items` sidecar，不重新运行模型，也不修改 `third_party/SAME`，用于生成更适合逐样本错误分析、表格筛选和跨指标聚合的细粒度指标。

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

- `manifest.json`：记录 schema、输入目录、connectivity 目录、输出文件列表和行数。
- `jsonl/*_fine_metrics.jsonl`：每个 dataset/split 一个 JSONL 文件，每行一个样本，是最完整的逐样本产物。
- `tables/fine_metrics_wide.csv`：一行一个样本，把三类指标摊平成列，适合 Excel、pandas、R、DuckDB 直接筛选。
- `tables/fine_metrics_long.csv`：一行一个“样本-指标”，适合做跨指标聚合、透视和绘图。
- `tables/fine_metrics_summary.json`：按 dataset/split 汇总 item 数和 success 计数，用于快速 sanity check。

`0011` 当前生成的行数为：

```text
CVDN_val_unseen_fine_metrics.jsonl     907
R2R_val_unseen_fine_metrics.jsonl      2349
REVERIE_val_unseen_fine_metrics.jsonl  3521
SOON_val_unseen_fine_metrics.jsonl     3390
fine_metrics_wide.csv                  10167 data rows
fine_metrics_long.csv                  230685 data rows
```

## 2. JSONL schema

每行结构如下：

```json
{
  "schema_version": "same_fine_metrics.v1",
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
  "eval_end_region": {}
}
```

顶层字段：

- `schema_version`：当前为 `same_fine_metrics.v1`。
- `experiment_id`：实验目录名。
- `dataset` / `split`：来自对应的 `eval_context.json`。
- `identity`：从 `eval_items.identity` 继承，保留内部唯一 ID、官方保存 ID 和原始 source IDs。
- `common`：所有数据集共有的路径与指令指标。
- `eval_end_goal`：以单一导航终点为目标的指标。
- `eval_end_region`：以目标 viewpoint 集合为目标的指标。R2R 没有 region 目标，因此该字段为 `null`。

## 3. common 指标

`common` 包含所有数据集共有的部分：

| 字段 | 类型 | 含义 | 计算方式 |
| --- | --- | --- | --- |
| `action_step_count` | `int` | 模型做了多少次高层导航选择，不包含 stop | 优先读 `official_item_scores.raw_same.action_steps`；缺失时用 `len(prediction.pred_path_segments) - 1` |
| `move_step_count` | `int` | 实际展开后沿拓扑图移动了多少次 | flatten 后 `prediction.trajectory` 中相邻 viewpoint 发生变化的次数 |
| `instruction_token_count` | `int` | 指令 token 数 | `annotation.instruction_meta.encoding_len`，也就是实际 BERT WordPiece encoding 长度，通常包含 `[CLS]` 和 `[SEP]` |
| `path_length_m` | `float` | 实际走过总路径长度，单位米 | 优先读 `official_item_scores.canonical.actual_length_m`；缺失时回退到 cumulative 或 edge length 求和 |
| `path_edge_count` | `int` | 实际走过路径所包含的边数 | 当前等于 `move_step_count` |

易错点：

- `action_step_count` 当前统一为“不包含 stop”。SAME 的 R2R/REVERIE/SOON raw `action_steps` 本身就是 `len(pred_path) - 1`；CVDN raw 缺失，因此用同一口径回退。
- `move_step_count` 不是 `len(trajectory) - 1`。如果 trajectory 中相邻 viewpoint 相同，原地不动不计入 move。
- `path_edge_count` 当前按“相邻 viewpoint 是否变化”计数，与 `move_step_count` 一致。它不是 shortest path 内部展开后的边数之和。

## 4. eval_end_goal 指标

`eval_end_goal` 以单一导航终点作为目标：

- R2R：`annotation.nav_goal_viewpoint`
- CVDN：`annotation.nav_goal_viewpoint`
- REVERIE：`annotation.nav_goal_viewpoint`
- SOON：`annotation.gt_path[-1]`，通常也等于 sidecar 中的 `nav_goal_viewpoint`

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `final_success` | `bool` | 最终位置到 goal 的图距离是否 `< success_threshold_m` |
| `oracle_success` | `bool` | 轨迹中是否曾有一步到 goal 的图距离 `< success_threshold_m` |
| `final_distance_to_goal_m` | `float` | 最终位置到 goal 的加权最短路径距离，单位米 |
| `final_distance_to_goal_edges` | `int` | 最终位置到 goal 的加权最短路径边数 |
| `path_length_ratio` | `float` | `common.path_length_m / shortest_path_length_m` |
| `oracle_path_length_m` | `float/null` | 如果第一次进入成功范围就停止，已走路径长度 |
| `oracle_path_edge_count` | `int/null` | 如果第一次进入成功范围就停止，已走 move 边数 |
| `oracle_path_length_ratio` | `float/null` | `oracle_path_length_m / shortest_path_length_m` |
| `shortest_path_length_m` | `float` | 起点到 goal 的加权最短路径长度 |
| `shortest_path_edge_count` | `int` | 起点到 goal 的加权最短路径边数 |

关键实现点：

- 成功阈值来自对应 `eval_context.json` 的 `run_context.success_threshold_m`，当前通常是 `3.0` 米。
- 米制距离优先复用 `primitives.distance_to_nav_goal_by_step_m` 和 `canonical.shortest_path_length_m`。
- 边数需要重新读取 `data/same/simulator/connectivity/*_connectivity.json`，按 SAME 一样的 pose 欧氏距离作为边权，然后跑 Dijkstra。
- `oracle_path_*` 使用第一次满足成功阈值的 step。如果从未成功，三个 oracle path 字段写 `null`。

易错点：

- REVERIE 的 SAME 官方 success 使用 object-visible viewpoint 集合；但这里的 `eval_end_goal` 明确使用 `nav_goal_viewpoint`，所以不能直接复用 `canonical.final_success`。
- `shortest_path_edge_count` 是“加权最短路径”对应路径的边数，不是拓扑边数最少路径的边数。
- 如果 `shortest_path_length_m` 为 0，ratio 写 `null`，避免除零。

## 5. eval_end_region 指标

`eval_end_region` 以目标 viewpoint 集合作为目标。R2R 没有这组指标，因此 JSONL 中写 `null`，wide CSV 中对应列为空，long CSV 中不展开 R2R 的 region 指标。

目标集合来源：

- CVDN：`dataset_extras.cvdn.end_panos`
- REVERIE：`annotation.success_target_viewpoints`
- SOON：`dataset_extras.soon.bbox_viewpoints`，对应 SAME 样本中的 `end_image_ids` / observation 里的 `gt_end_vps`

字段与 `eval_end_goal` 相同：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `final_success` | `bool` | 最终 viewpoint 是否命中目标集合 |
| `oracle_success` | `bool` | 轨迹中是否曾命中目标集合 |
| `final_distance_to_goal_m` | `float` | 最终位置到目标集合中最近 viewpoint 的加权最短路径距离 |
| `final_distance_to_goal_edges` | `int` | 最终位置到最近目标 viewpoint 的加权最短路径边数 |
| `path_length_ratio` | `float` | `common.path_length_m / shortest_path_length_m` |
| `oracle_path_length_m` | `float/null` | 如果第一次命中目标集合就停止，已走路径长度 |
| `oracle_path_edge_count` | `int/null` | 如果第一次命中目标集合就停止，已走 move 边数 |
| `oracle_path_length_ratio` | `float/null` | `oracle_path_length_m / shortest_path_length_m` |
| `shortest_path_length_m` | `float` | 起点到目标集合最近 viewpoint 的加权最短路径长度 |
| `shortest_path_edge_count` | `int` | 起点到目标集合最近 viewpoint 的加权最短路径边数 |

易错点：

- region success 是“命中集合”，不是 `< 3m`。也就是说，只有当前/曾经的 viewpoint 本身在目标集合中，`final_success/oracle_success` 才为 true。
- 距离指标仍然是到目标集合最近点的图上加权最短距离；因此距离可以小于 3m，但 success 仍为 false。
- `oracle_path_*` 取第一次命中集合的 step。如果从未命中集合，写 `null`。
- CVDN 的 `eval_end_goal` 使用 trusted-path 的 `nav_goal_viewpoint`；`eval_end_region` 使用原始可接受的 `end_panos` 集合。这两者可能不同。

## 6. CSV 表

### 6.1 fine_metrics_wide.csv

wide 表一行一个样本，适合筛选和人工检查。列名用点号展开：

```text
experiment_id,dataset,split,internal_item_id,saved_instr_id,
common.action_step_count,
common.move_step_count,
...
eval_end_goal.final_success,
eval_end_goal.oracle_success,
...
eval_end_region.final_success,
eval_end_region.oracle_success,
...
```

布尔值写为 `true/false`，空值写为空字符串。

### 6.2 fine_metrics_long.csv

long 表一行一个“样本-指标”，适合聚合和画图：

```text
experiment_id,dataset,split,internal_item_id,metric_group,metric_name,value_num,value_bool,value_type
```

字段规则：

- `metric_group`：`common`、`eval_end_goal` 或 `eval_end_region`。
- `metric_name`：指标名，例如 `action_step_count`、`final_success`。
- `value_num`：数值型指标写这里。
- `value_bool`：布尔型指标写这里，取值 `true/false`。
- `value_type`：`num`、`bool`、`null` 或 `str`。

R2R 的 `eval_end_region` 为 `null`，因此 long 表不会为 R2R 生成 region 指标行。

## 7. 代码实现关键点

核心实现位于 `scripts/analysis/build_same_fine_metrics.py`：

- `discover_eval_item_sources()` 扫描 `eval_items/*_eval_context.json`，确定 dataset、split、success threshold 和配套 JSONL。
- `Graph.from_connectivity_file()` 读取 MatterSim connectivity JSON，用 `pose[3]`、`pose[7]`、`pose[11]` 计算边权，避免依赖 `networkx`。
- `Graph.shortest()` 用标准库 `heapq` 实现 Dijkstra，同时返回最短米数和该加权最短路径的边数。
- `build_common_metrics()` 生成共有指标，并处理 CVDN 缺少 raw `action_steps` 的 fallback。
- `build_goal_metrics()` 生成单点 goal 指标。
- `build_region_metrics()` 生成 region 集合指标，并对 R2R 返回 `None`。
- `write_wide_csv()` 和 `write_long_csv()` 分别生成宽表和长表。

当前测试位于 `tests/test_build_same_fine_metrics.py`，覆盖：

- 小型 connectivity graph 上的 common、goal、region 指标。
- `action_step_count` 有 raw 时用 raw，缺失时用 `len(pred_path_segments) - 1`。
- region success 必须命中集合，不使用 3m 阈值。
- wide/long CSV 的列名、布尔值和空值写法。
- 对真实 `0011` 产物做只读 smoke test，确认行数和 R2R region 为空。

## 8. 与 eval_items 的关系

`eval_items` 是稳定的样本级溯源材料，`fine_metrics` 是面向当前分析问题的派生表。推荐流程是：

1. SAME 完整评估生成 `eval_items`。
2. 使用 `scripts/analysis/build_same_fine_metrics.py` 从 `eval_items` 派生 `fine_metrics`。
3. 后续错误分析、分桶、可视化优先读取 `fine_metrics/jsonl` 或 `fine_metrics/tables`。

当未来需要新增实验性字段时，优先扩展父项目后处理脚本；只有当 `eval_items` 缺少无法恢复的原始信息时，才考虑给 SAME 子项目增加 patch。
