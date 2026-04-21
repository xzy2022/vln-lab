# SAME Eval Items Sidecar 说明

本文档说明 SAME 实验中的 `eval_items` sidecar 文件。它的目标不是直接完成所有细粒度分析，而是在完整数据集评估时保存足够可靠的样本级溯源材料，之后可以在父项目中单独生成更自由的 `fine_metrics.jsonl`、错误分析表或可视化报告。

## 1. 总体架构

SAME 原始评估流程会产生两类信息：

1. 官方提交格式结果，例如 `results/R2R_val_unseen_results.json`。
2. split 级聚合指标，例如 `metrics.json` 和日志中的 `sr`、`spl`、`nav_error`。

这两类文件适合复现官方分数，但不适合做逐样本分析。原因是官方结果文件会为了提交格式压缩或改写字段，例如 R2R/REVERIE 会去掉 `instr_id` 前缀，SOON 会把 `instr_id` 压缩成整数，CVDN 会把 `instr_idx/inst_idx` 写成官方需要的形式。只靠这些结果文件，后续很难稳定反查原始 annotation、路径、目标语义和 per-item 评分。

`eval_items` sidecar 解决的是这个问题。它在 `dataset.save_json()` 改写结果之前写出，保留 SAME 内部唯一样本 ID、官方保存 ID、annotation 关键信息、预测轨迹、图距离中间量和 `eval_metrics()` 已经算出的 per-item 官方字段。

一次实验的输出结构如下：

```text
experiment_outputs/<experiment_id>/
  results/
    R2R_val_unseen_results.json
    REVERIE_val_unseen_results.json
    CVDN_val_unseen_results.json
    SOON_val_unseen_results.json
  eval_items/
    R2R_val_unseen_eval_context.json
    R2R_val_unseen_eval_items.jsonl
    REVERIE_val_unseen_eval_context.json
    REVERIE_val_unseen_eval_items.jsonl
    CVDN_val_unseen_eval_context.json
    CVDN_val_unseen_eval_items.jsonl
    SOON_val_unseen_eval_context.json
    SOON_val_unseen_eval_items.jsonl
```

每个 dataset/split 对应两个文件：

- `*_eval_context.json`：保存一次运行级上下文和 split 级汇总。
- `*_eval_items.jsonl`：每行一条被评估样本，保存样本级材料。

当前 sidecar 由 `patches/same/experimental/0004-eval-items-sidecars.patch` 引入。注意 `scripts/experiments/run_same.py` 默认只自动应用 `patches/same/base/*.patch`，因此 experimental patch 需要手动 apply，或后续移动到 base 后才会随运行器自动生效。

## 2. eval_context.json

`eval_context.json` 是 dataset/split 级别的上下文文件，不重复写入每一行样本。典型结构如下：

```json
{
  "schema_version": "eval_context.v1",
  "items_schema_version": "eval_items.v2",
  "run_context": {
    "framework": "SAME",
    "experiment_id": "0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1",
    "dataset": "CVDN",
    "split": "val_unseen",
    "simulation_env": "mattersim",
    "success_threshold_m": 3.0
  },
  "files": {
    "official_results": "CVDN_val_unseen_results.json",
    "eval_items": "CVDN_val_unseen_eval_items.jsonl"
  },
  "counts": {
    "predictions": 907,
    "items": 907,
    "unique_internal_item_ids": 907
  },
  "score_summary": {
    "nav_error": 12.94,
    "sr": 24.03,
    "spl": 16.98
  }
}
```

字段含义：

- `schema_version`：context 文件自身 schema 版本。
- `items_schema_version`：配套 JSONL 行 schema 版本。
- `run_context.framework`：评估框架，目前为 `SAME`。
- `run_context.experiment_id`：父项目分配的实验 ID。
- `run_context.dataset`：数据集名，取值为 `R2R`、`REVERIE`、`CVDN`、`SOON`。
- `run_context.split`：SAME split 名，例如 `val_seen`、`val_unseen`。
- `run_context.simulation_env`：用于距离计算的环境，目前主要是 `mattersim`。
- `run_context.success_threshold_m`：距离成功阈值，目前为 `3.0` 米。
- `files.official_results`：对应的 SAME 官方结果文件名。
- `files.eval_items`：对应的 JSONL 文件名。
- `counts.predictions`：本次参与保存的预测条数。
- `counts.items`：JSONL 行数。
- `counts.unique_internal_item_ids`：唯一内部样本 ID 数量。
- `score_summary`：`eval_metrics()` 返回的 split 级聚合指标原样保存。

## 3. eval_items.jsonl

`eval_items.jsonl` 每行是一条样本。它是 JSONL，而不是一个大的 JSON 数组，便于流式读取、按行抽样和后续增量处理。

顶层结构固定为：

```json
{
  "schema_version": "eval_items.v2",
  "identity": {},
  "annotation": {},
  "prediction": {},
  "primitives": {},
  "official_item_scores": {},
  "dataset_extras": {}
}
```

### 3.1 identity

`identity` 保存样本身份信息，分为 SAME/评估流程统一字段和数据集原始字段。

```json
{
  "internal_item_id": "cvdn_0_5",
  "saved_instr_id": "cvdn_0_5",
  "source_ids": {
    "path_id": null,
    "instr_idx": null,
    "inst_idx": 5,
    "obj_id": null,
    "raw_idx": 0,
    "sample_idx": 0
  },
  "same_ids": {
    "instr_id": "cvdn_0_5",
    "path_id": 5,
    "sample_idx": 0,
    "data_type": "cvdn"
  }
}
```

字段含义：

- `internal_item_id`：SAME 评估内部唯一 key，用于连接 prediction、annotation 和 `eval_metrics()` per-item 字段。这个字段在一个 JSONL 文件内必须唯一。
- `saved_instr_id`：官方结果文件中保存的 ID。它保持 SAME 官方保存逻辑的类型和形式。SOON 中它是压缩后的整数，允许重复。
- `source_ids`：尽量表示数据集原始身份字段，不把 SAME 临时制造的字段混进去。
- `same_ids`：SAME 为统一流程使用或临时制造的字段。例如 CVDN 中 SAME 会把 `inst_idx` 放进 `path_id`，这个 `path_id` 属于 SAME 内部字段，应放在 `same_ids`。

通用 `source_ids` 字段：

- `path_id`：原始数据集里的路径 ID。CVDN 没有原生 `path_id` 概念，因此为 `null`。
- `instr_idx`：原始指令编号。R2R/REVERIE/SOON 通常存在；CVDN 为 `null`。
- `inst_idx`：CVDN 原始样本 ID。其他数据集通常为 `null`。
- `obj_id`：REVERIE 对象 ID。其他数据集通常为 `null`。
- `raw_idx`：加载原始 annotation 时的原始样本序号。
- `sample_idx`：SAME 展开 instruction 后的样本序号。

### 3.2 annotation

`annotation` 保存和任务定义直接相关的信息。

```json
{
  "scan": "Z6MFQCViBuw",
  "instruction": "...",
  "instruction_meta": {
    "whitespace_token_count": 31,
    "encoding_len": 46
  },
  "start_viewpoint": "2008e72476f84104858e908beeac0193",
  "nav_goal_viewpoint": "59aed460238643f48f593e9b9ba1c17d",
  "success_target_viewpoints": ["59aed460238643f48f593e9b9ba1c17d"],
  "gt_path": ["2008e72476f84104858e908beeac0193", "..."]
}
```

字段含义：

- `scan`：Matterport3D scan id。
- `instruction`：SAME 实际使用的文本指令。
- `instruction_meta.whitespace_token_count`：按空白切分的粗略词数。
- `instruction_meta.encoding_len`：SAME 中 `instr_encoding` 的长度。
- `start_viewpoint`：GT path 起点。
- `nav_goal_viewpoint`：导航目标点。通常是 `gt_path[-1]`。
- `success_target_viewpoints`：按 SAME 官方 success 语义使用的成功目标集合。
- `gt_path`：SAME 评估时使用的有效 GT path。对 CVDN 来说，这可能已经经过 SAME 的 trusted-path 预处理。

### 3.3 prediction

`prediction` 保存模型输出路径，并同时保留 SAME 官方结果文件会写出的 trajectory 形态。

```json
{
  "pred_path_segments": [["vp_a"], ["vp_b"], ["vp_c"]],
  "trajectory": ["vp_a", "vp_b", "vp_c"],
  "official_saved_trajectory": [["vp_a", 0, 0], ["vp_b", 0, 0], ["vp_c", 0, 0]]
}
```

字段含义：

- `pred_path_segments`：SAME agent 原始输出的分段路径。
- `trajectory`：flatten 后的 viewpoint 序列，后续分析一般优先用它。
- `official_saved_trajectory`：与 `results/*_results.json` 中保存格式一致的 trajectory。R2R/REVERIE/CVDN 是 `[viewpoint, 0, 0]` 列表；SOON 是包含 `path/obj_heading/obj_elevation` 的对象列表。

SOON 示例：

```json
{
  "official_saved_trajectory": [
    {
      "path": [["vp_a", 0, 0], ["vp_b", 0, 0]],
      "obj_heading": [0],
      "obj_elevation": [0]
    }
  ]
}
```

### 3.4 primitives

`primitives` 只保存评估时容易拿到、但后续仅从官方结果文件很难恢复的中间量。它不是细粒度指标表，不保存 `length_ratio`、`repeated_view_count`、`final_minus_min_distance` 这类后续派生指标。

通用字段：

- `final_viewpoint`：预测路径最后一个 viewpoint。
- `nearest_viewpoint_to_nav_goal`：预测路径上离 `nav_goal_viewpoint` 最近的 viewpoint。
- `trajectory_edge_lengths_m`：预测轨迹相邻 viewpoint 间的最短路径距离，单位米。
- `trajectory_cumulative_lengths_m`：预测轨迹累计长度，长度应等于 `trajectory` 长度。
- `distance_to_nav_goal_by_step_m`：预测轨迹每一步到 `nav_goal_viewpoint` 的距离。
- `gt_path_edge_lengths_m`：GT path 相邻 viewpoint 间距离。
- `gt_path_length_m`：GT path 按边累加的长度。
- `shortest_start_to_nav_goal_distance_m`：从起点到 `nav_goal_viewpoint` 的图上加权最短路径距离，单位米。SAME 加载 Matterport3D connectivity graph 时，会给每条可通行边设置权重，权重是两个 viewpoint pose 坐标之间的欧氏距离；随后用 NetworkX 的 Dijkstra 算法预计算 `shortest_distances`。因此这里的“最短”指最小累计米数，不是拓扑图中经过边数最少。

数据集特有 primitive：

- REVERIE：`distance_to_nearest_success_target_by_step_m`，每一步到最近 object-visible viewpoint 的距离。
- CVDN：`distance_to_planner_goal_by_step_m`，每一步到 planner goal 的距离。
- CVDN：`distance_to_nearest_end_pano_by_step_m`，每一步到最近 `end_panos` 的距离。
- SOON：`distance_to_nearest_bbox_viewpoint_by_step_m`，每一步到最近 bbox viewpoint 的距离。

### 3.5 official_item_scores

`official_item_scores` 保存 SAME 已经算出的 per-item 官方字段，并额外给出少量跨数据集统一别名。

```json
{
  "raw_same": {
    "nav_errors": 7.31,
    "oracle_errors": 7.31,
    "trajectory_lengths": 16.78,
    "shortest_path_lengths": 21.98,
    "success": 0.0,
    "spl": 0.0
  },
  "canonical": {
    "final_distance_m": 7.31,
    "min_distance_along_trajectory_m": 7.31,
    "actual_length_m": 16.78,
    "shortest_path_length_m": 21.98,
    "final_success": false,
    "oracle_success": false
  }
}
```

`raw_same` 的原则是原样保存 `eval_metrics()` 已经生成的 per-item 字段，不改名，不丢字段。不同数据集字段名不同，这是 SAME 现状的一部分。

`canonical` 只做少量统一别名，便于下游脚本用同一套字段做基本筛选：

- `final_distance_m`
- `min_distance_along_trajectory_m`
- `actual_length_m`
- `shortest_path_length_m`
- `final_success`
- `oracle_success`

当前映射规则：

- R2R/REVERIE/SOON：`raw_same.nav_error -> canonical.final_distance_m`
- R2R/REVERIE/SOON：`raw_same.oracle_error -> canonical.min_distance_along_trajectory_m`
- CVDN：`raw_same.nav_errors -> canonical.final_distance_m`
- CVDN：`raw_same.oracle_errors -> canonical.min_distance_along_trajectory_m`
- 所有数据集：`raw_same.trajectory_lengths -> canonical.actual_length_m`
- 所有数据集：`raw_same.success -> canonical.final_success`
- `oracle_success` 如果 `raw_same` 已存在就直接使用；CVDN 中用 `oracle_errors < success_threshold_m` 补出。
- `shortest_path_length_m` 优先使用 `raw_same.shortest_path_lengths`，否则使用 `primitives.shortest_start_to_nav_goal_distance_m`。目前 CVDN 的 `raw_same` 会提供 `shortest_path_lengths`；R2R/REVERIE/SOON 通常不提供这个 raw 字段，所以 canonical 会回退到 primitive 中的图上加权最短距离。

### 3.6 dataset_extras

`dataset_extras` 保存数据集特有目标语义，避免把 dataset-specific 字段散落在顶层。

REVERIE：

```json
{
  "reverie": {
    "gt_obj_id": 215,
    "pred_obj_id": "215",
    "visible_viewpoints": ["vp_a", "vp_b"]
  }
}
```

CVDN：

```json
{
  "cvdn": {
    "end_panos": ["vp_a", "vp_b"],
    "planner_path": ["vp_s", "vp_t"],
    "planner_goal": "vp_t"
  }
}
```

SOON：

```json
{
  "soon": {
    "bbox_viewpoints": ["vp_a", "vp_b"],
    "obj_name": "chair",
    "pseudo_labels": {},
    "pred_obj_heading": null,
    "pred_obj_elevation": null
  }
}
```

## 4. 各数据集特殊处理

### R2R

R2R 是最标准的路径跟随任务。

- `internal_item_id`：`r2r_{path_id}_{instr_idx}`。
- `saved_instr_id`：去掉 `r2r_` 前缀，例如 `{path_id}_{instr_idx}`。
- `source_ids.path_id`：原始 `path_id`。
- `source_ids.instr_idx`：指令编号。
- `nav_goal_viewpoint`：`gt_path[-1]`。
- `success_target_viewpoints`：单元素列表 `[nav_goal_viewpoint]`。
- `raw_same` 主要包含 `nav_error`、`oracle_error`、`action_steps`、`trajectory_steps`、`trajectory_lengths`、`success`、`oracle_success`、`spl`、`DTW`、`nDTW`、`SDTW`、`CLS`。

### REVERIE

REVERIE 既有导航目标，也有对象 grounding 目标。导航距离仍然按 `gt_path[-1]` 计算，但 success 使用对象可见 viewpoint 集合。

- `internal_item_id`：通常为 `reverie_{path_id}_{obj_id}_{instr_idx}`。
- `saved_instr_id`：去掉 `reverie_` 前缀。
- `source_ids.path_id`：原始 `path_id`。
- `source_ids.obj_id`：目标对象 ID。
- `nav_goal_viewpoint`：`gt_path[-1]`，用于 `nav_error`。
- `success_target_viewpoints`：object-visible viewpoints。
- `dataset_extras.reverie.visible_viewpoints`：与 `success_target_viewpoints` 对齐。
- `primitives.distance_to_nearest_success_target_by_step_m`：每一步到最近 object-visible viewpoint 的距离。
- `raw_same` 除导航指标外，还包含 `rgs`、`rgspl`。

需要注意：REVERIE 的 `success` 不是简单的 `nav_error < 3m`，而是最终 viewpoint 是否能看到目标对象。因此下游做成功分析时应优先使用 `raw_same.success` 或 `canonical.final_success`，不要自己用 `final_distance_m < 3` 推断。

### CVDN

CVDN 的原始身份字段与 R2R/REVERIE 不同。它的原始样本 ID 是 `inst_idx`，SAME 当前内部为了复用流程把 `path_id` 设置为 `inst_idx`。sidecar 明确区分二者：

- `internal_item_id`：`cvdn_{sample_idx}_{inst_idx}`。
- `saved_instr_id`：当前 SAME 输出中保持同一个内部 ID。
- `source_ids.path_id`：`null`，因为 CVDN 原始数据没有这个原生概念。
- `source_ids.instr_idx`：`null`。
- `source_ids.inst_idx`：原始 CVDN `inst_idx`。
- `same_ids.path_id`：SAME 内部的 `ann["path_id"]`，语义上等于 `inst_idx`，但它是 SAME 临时制造的字段。
- `annotation.gt_path`：SAME 经过 CVDN trusted-path 预处理后的有效路径。
- `nav_goal_viewpoint`：有效 `gt_path[-1]`。
- `success_target_viewpoints`：`[nav_goal_viewpoint]`。
- `dataset_extras.cvdn.end_panos`：原始可接受 end panos。
- `dataset_extras.cvdn.planner_path`：原始 planner path。
- `dataset_extras.cvdn.planner_goal`：planner path 末端。
- `primitives.distance_to_planner_goal_by_step_m`：支持后续分析是否经过 planner goal。
- `primitives.distance_to_nearest_end_pano_by_step_m`：支持后续分析到 end pano 集合的距离变化。
- `raw_same` 包含 `nav_errors`、`oracle_errors`、`oracle_plan_errors`、`dist_to_end_reductions`、`trajectory_lengths`、`success`、`spl`、`shortest_path_lengths` 等。

需要注意：CVDN 的 `raw_same` 使用复数键名，例如 `nav_errors`、`oracle_errors`。`canonical` 会把这些映射成统一字段。

### SOON

SOON 的官方保存 ID 会被压缩，这会造成 `saved_instr_id` 重复。因此 SOON 的唯一身份必须依赖 `internal_item_id`。

- `internal_item_id`：`soon_{raw_idx}_{path_id}_{instr_idx}`。
- `saved_instr_id`：官方保存逻辑中的压缩 ID，通常为整数，例如 `int(path_id.split("-")[0])`。
- `source_ids.path_id`：原始 SOON `path_id`，可能形如 `321-2`。
- `source_ids.instr_idx`：指令编号。
- `source_ids.raw_idx`：原始 JSONL 行号。
- `nav_goal_viewpoint`：`gt_path[-1]`，用于导航 success。
- `success_target_viewpoints`：`[nav_goal_viewpoint]`。
- `dataset_extras.soon.bbox_viewpoints`：bbox 所在 viewpoint 集合。
- `dataset_extras.soon.pseudo_labels`：annotation 中的 pseudo label 映射。
- `dataset_extras.soon.pred_obj_heading/pred_obj_elevation`：模型预测的对象方向。如果模型没有输出，通常为 `null`。
- `prediction.official_saved_trajectory`：必须是 SOON 官方结构，即包含 `path/obj_heading/obj_elevation` 的对象列表。
- `primitives.distance_to_nearest_bbox_viewpoint_by_step_m`：每一步到最近 bbox viewpoint 的距离。
- `raw_same` 除导航指标外，还包含 `goal_progress`、`det_success`、`det_spl`，以及可选的 `heading_error`、`elevation_error`、`point_det_error`。

需要注意：当模型没有输出 object direction 时，`heading_error/elevation_error/point_det_error` 可以全为 `null`，`det_sr` 可能为 0。这表示检测相关输出缺失或失败，不代表 sidecar 字段生成失败。

## 5. 校验建议

生成 sidecar 后建议做以下检查：

1. 每个非 test dataset/split 都有一对 context/items 文件。
2. `eval_context.json.counts.items` 等于 JSONL 行数。
3. JSONL 行数等于 `results/*_results.json` 条数。
4. `identity.internal_item_id` 在单个 JSONL 文件内唯一。
5. SOON 允许 `identity.saved_instr_id` 重复，但不允许 `internal_item_id` 重复。
6. `prediction.trajectory[0] == annotation.start_viewpoint`。
7. `prediction.official_saved_trajectory` 与官方 results 文件中的 trajectory 一致。
8. 用 `canonical.final_distance_m` 聚合出的 `nav_error` 与 context 中 `score_summary.nav_error` 一致。
9. 用 `canonical.final_success` 聚合出的 `sr` 与 context 中 `score_summary.sr` 一致。
10. 用 `raw_same.spl` 聚合出的 `spl` 与 context 中 `score_summary.spl` 一致。

## 6. 后续细粒度分析

`eval_items.jsonl` 是后续分析的输入，不建议把所有实验性指标都塞进 sidecar。推荐流程是：

1. SAME 完整评估时生成 `eval_items` sidecar。
2. 父项目中新建独立分析脚本读取 sidecar。
3. 根据当前研究问题计算派生指标。
4. 输出到单独文件，例如：

```text
reports/item_metrics/<experiment_id>_fine_metrics.jsonl
```

适合放到后续 `fine_metrics.jsonl` 的派生指标包括：

- `length_ratio`
- `step_count`
- `repeated_view_count`
- `final_minus_min_distance`
- `instruction_length_bucket`
- 各类路径绕路、回退、停早、停晚分析标签

这样 sidecar 保持稳定和可溯源，细粒度指标可以随着问题定义不断迭代，不需要反复重跑 SAME 模型评估。
