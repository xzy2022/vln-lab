# Result Card

## Main metrics
| dataset | split | SR | SPL | NE | OSR | nDTW | SDTW | CLS | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| R2R | val_train_seen | 88.74 | 85.01 | 1.28 | 94.34 | 88.18 | 83.85 | 87.06 | oracle_error=0.62m; action_steps=5.44; steps=5.76; lengths=11.18m |
| R2R | val_unseen | 76.29 | 66.24 | 2.72 | 84.80 | 71.32 | 63.74 | 70.42 | oracle_error=1.40m; action_steps=6.12; steps=7.06; lengths=13.59m |
| REVERIE | val_train_seen | 83.74 | 79.20 | 0.84 | 93.50 | 92.26 | 79.87 | 90.29 | oracle_error=0.26m; rgs=0.00; rgspl=0.00; action_steps=5.47; steps=5.63; lengths=10.57m; object grounding 扩展字段未激活 |
| REVERIE | val_unseen | 45.84 | 35.85 | 5.23 | 54.64 | 48.36 | 33.45 | 50.88 | oracle_error=2.52m; rgs=0.00; rgspl=0.00; action_steps=7.34; steps=9.85; lengths=18.89m; object grounding 扩展字段未激活 |
| CVDN | val_unseen | 24.04 | 16.98 | 12.94 | 53.03 |  |  |  | oracle_path_success_rate=80.60; dist_to_end_reduction=6.76m; lengths=31.08m |
| SOON | val_unseen | 36.34 | 25.66 | 8.13 | 54.01 |  |  |  | oracle_error=4.52m; det_sr=0.00; det_spl=0.00; action_steps=12.68; steps=17.65; lengths=34.75m; detection 扩展字段未激活 |

## Comparison
| dataset | split | metric | paper_official | current | delta | note |
|---|---|---|---:|---:|---:|---|
| R2R | val_unseen | SR | 76.00 | 76.29 | +0.29 | Table 4 |
| R2R | val_unseen | SPL | 66.00 | 66.24 | +0.24 | Table 4 |
| REVERIE | val_unseen | SR | 46.40 | 45.84 | -0.56 | Table 4 |
| REVERIE | val_unseen | SPL | 36.10 | 35.85 | -0.25 | Table 4 |
| SOON | val_unseen | SR | 36.10 | 36.34 | +0.24 | Table 4 |
| SOON | val_unseen | SPL | 25.40 | 25.66 | +0.26 | Table 4 |
| CVDN | val_unseen | GP | 6.94 | 6.76 | -0.18 | 论文中的 `GP` 对应当前实现的 `dist_to_end_reduction`；论文表 split 记作 `val` |

## Resource
| phase | time | peak_mem | latency | note |
|---|---:|---:|---:|---|
| eval(all 6 splits) | 1327.47 |  | 0.113 s/sample | 11791 条结果; 包含模型加载、图构建与最短路预处理 |

## Interpretation
- 是否对齐论文：R2R、REVERIE、SOON 的 `val_unseen` SR/SPL 与论文 Table 4 基本对齐，偏差都在 ±0.56 以内；CVDN 的论文 `GP` 与当前实现的 `dist_to_end_reduction` 对齐后低 0.18。
- 最大差距：REVERIE `val_unseen` 的 SR 为 -0.56，是当前和论文差距最大的已知有效对照项。
- 口径澄清：REVERIE 的 `rgs / rgspl` 与 SOON 的 `det_sr / det_spl` 在本次运行中虽被 evaluator 写出，但当前配置是默认主线 `run.py -> BaseTrainer -> DUETAgent -> VLNBert`，且 `feature.enable_og=false`，runtime 不产出 `pred_objid / pred_obj_direction`。因此这些 0 应解释为“object grounding / detection 扩展字段未激活”，不是论文主结果复现失败，也不应直接参与论文对照。
- 下一步：补齐 `official_results.csv` 的 SAME 参考项后再生成更完整对比；若后续需要 REVERIE/SOON 的 grounding/detection 指标可用，应在当前主线下补最小 object prediction 输出链路，而不是仅根据这次结果把模型能力判定为失败。

## Notes
- `peak_mem` 留空，因为归档里只有运行前的 `nvidia-smi` 快照，没有峰值显存监控结果。
- Comparison 只填写论文里已有且与当前主线口径一致的官方参考项；其余空白并不表示实验没有产出，而是当前没有可靠对照基线，或该指标属于未激活的扩展字段。
