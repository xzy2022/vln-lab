## 示例模板
| Patch 文件名                      | 目的                 | 类型           | 适用版本                | 是否默认应用 | 备注                      |
| ------------------------------ | ------------------ | ------------ | ------------------- | ------ | ----------------------- |
| 0001-eval-only-exit.patch      | 阻止 eval 后进入训练      | base         | SAME commit b74c57b | 是      | 保证 checkpoint eval-only |
| 0002-export-json-metrics.patch | 导出结构化 metrics.json | base         | SAME commit b74c57b | 是      | 用于统一 parser             |
| 0003-debug-temp.patch          | 临时 debug           | experimental | SAME commit b74c57b | 否      | 仅开发期间使用                 |

## 正文
| Patch 文件名                                | 目的                                      | 类型           | 适用版本                | 是否默认应用 | 备注                                                     |
| ---------------------------------------- | --------------------------------------- | ------------ | ------------------- | ------ | ------------------------------------------------------ |
| base/0001-eval-only-exit.patch           | 为 `run.py` 增加显式 `experiment.eval_only` | base         | SAME commit b74c57b | 是      | 跳过 `train_dataloaders`，执行 `val_one_epoch(0)` 后直接退出，不改模型与训练实现 |
| base/0002-console-stdout.patch           | 将控制台 `INFO` 日志从 `stderr` 改到 `stdout` | base         | SAME commit b74c57b | 是      | 仅调整 `logging.StreamHandler` 的目标流，便于实验归档区分 stdout/stderr |
| base/0003-cvdn-soon-path-metrics.patch   | 为 CVDN/SOON 增加路径质量指标                 | base         | SAME commit b74c57b | 是      | 追加 `nDTW`、`SDTW`、`CLS`，保留原有 SR/SPL 等指标                 |
| base/0004-eval-items-sidecars.patch      | 为评测输出增加 eval items 明细文件              | base         | SAME commit b74c57b | 是      | 在官方 results 旁输出 `eval_items`/`eval_context` sidecar，用于逐项误差分析 |
| base/0005-decision-trace.patch           | 为 DUET 导航导出逐步决策轨迹                    | base         | SAME commit b74c57b | 是      | 通过 `experiment.decision_trace`/`moe_trace` 开关写入 eval_items，不改变官方 results |

### 说明

- Patch 根目录是 `third_party/SAME/`。
- 默认 patch 放在 `patches/same/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method same`
  - `bash scripts/setup/revert_patches.sh --method same`
- 正式运行器 `scripts/experiments/run_same.py` 会默认应用并记录 `patches/same/base/*.patch`。仍在开发的补丁应放在 `patches/same/experimental/`，并在实验命令中显式声明：

```bash
python scripts/experiments/run_same.py \
  --config configs/same/val_unseen_r2r_reverie_cvdn_soon.yaml \
  --experimental-patch patches/same/experimental/0005-new-feature.patch
```

- `run.json.patch_set` 会记录 base patches 和 CLI 声明的 experimental patches；`run.json.patch_manifest` 会记录每个 patch 的类型和 `sha256`。
- `run.json.provenance.manual_worktree` 用于标记 `third_party/SAME/src` 是否存在未被声明 patch 覆盖的运行代码改动。普通 docs、reports、草稿文件的改动只会体现在 repo dirty 状态，不会算作 manual worktree。
- 如果希望正式实验遇到未声明的 SAME 运行代码改动时直接失败，使用：

```bash
python scripts/experiments/run_same.py \
  --config configs/same/val_unseen_r2r_reverie_cvdn_soon.yaml \
  --experimental-patch patches/same/experimental/0005-new-feature.patch \
  --fail-on-manual-worktree
```

- 应用后可在 YAML 中显式设置：
  - `experiment.eval_only: true`
  - `experiment.decision_trace: true`
  - `experiment.moe_trace: true`
- `base/0005-decision-trace.patch` 会把 `prediction.decision_trace` 写入 `eval_items/*_eval_items.jsonl`，并将 eval item 行 schema 从 `eval_items.v2` 升到 `eval_items.v3`。官方提交格式的 `results/*_results.json` 会在保存前移除 `decision_trace`，避免破坏 R2R/REVERIE/CVDN/SOON 原始提交格式。
- `decision_trace` 每步记录当前位置、fusion/fuse weight、stop 概率、被选动作、实际执行 viewpoint、stop 原因、图路由、全局图候选、局部候选和当前可见候选。候选项包含 logit/prob、距离、point id、heading/elevation 等后处理字段。
- `moe_trace` 只有在 `decision_trace` 开启时才会随 step 写出，内容是 global/local router logits 汇总后的专家平均概率、top experts、top expert probs 和 router entropy。
- 0001、0002 不触碰模型、loss、optimizer 或 trainer 细节；0003 仅补充 CVDN/SOON 的评测指标计算；0004 仅补充评测输出明细；0005 会为追踪 fusion 和 MoE 增加模型输出字段 `fuse_weights`、`global_router_logits`、`local_router_logits`，这些字段只用于诊断和 sidecar 后处理，不应作为训练目标或官方评测字段使用。
- trace 会明显增大 `eval_items` 体积。大规模正式跑分如果不需要逐步诊断，应保持 `experiment.decision_trace: false`。
