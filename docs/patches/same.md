## 示例模板
| Patch 文件名                      | 目的                 | 类型           | 适用版本                | 是否默认应用 | 备注                      |
| ------------------------------ | ------------------ | ------------ | ------------------- | ------ | ----------------------- |
| 0001-eval-only-exit.patch      | 阻止 eval 后进入训练      | base         | SAME commit b74c57b | 是      | 保证 checkpoint eval-only |
| 0002-export-json-metrics.patch | 导出结构化 metrics.json | base         | SAME commit b74c57b | 是      | 用于统一 parser             |
| 0003-debug-temp.patch          | 临时 debug           | experimental | SAME commit b74c57b | 否      | 仅开发期间使用                 |

## 正文
| Patch 文件名                                | 目的                                      | 类型           | 适用版本                | 是否默认应用 | 备注                                                     |
| ---------------------------------------- | --------------------------------------- | ------------ | ------------------- | ------ | ------------------------------------------------------ |
| experimental/0001-eval-only-exit.patch   | 为 `run.py` 增加显式 `experiment.eval_only` | experimental | SAME commit b74c57b | 否      | 跳过 `train_dataloaders`，执行 `val_one_epoch(0)` 后直接退出，不改模型与训练实现 |

### 说明

- Patch 根目录是 `third_party/SAME/`。
- 应用后可在 YAML 中显式设置：
  - `experiment.eval_only: true`
- 该补丁只修改入口调度与默认配置，不触碰模型、loss、optimizer 或 trainer 细节。
