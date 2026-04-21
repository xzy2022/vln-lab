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

### 说明

- Patch 根目录是 `third_party/SAME/`。
- 默认 patch 放在 `patches/same/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method same`
  - `bash scripts/setup/revert_patches.sh --method same`
- 应用后可在 YAML 中显式设置：
  - `experiment.eval_only: true`
- 这些补丁不触碰模型、loss、optimizer 或 trainer 细节；0003 仅补充 CVDN/SOON 的评测指标计算。
