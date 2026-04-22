# Result Card

## Main metrics
| dataset | split | SR | SPL | NE | OSR | nDTW | SDTW | CLS | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| R2R | val_unseen | 76.29 | 66.24 | 2.72 | 84.80 | 71.32 | 63.74 | 70.42 | oracle_error=1.40m; action_steps=6.12; steps=7.06; lengths=13.59m; Oracle-SPL=79.27 |
| REVERIE | val_unseen | 45.84 | 35.85 | 5.23 | 54.64 | 48.36 | 33.45 | 50.88 | oracle_error=2.52m; rgs=0.00; rgspl=0.00; action_steps=7.34; steps=9.85; lengths=18.89m; Oracle-SPL=43.73; object grounding 扩展字段未激活 |
| CVDN | val_unseen | 24.04 | 16.98 | 12.94 | 53.03 | 30.49 | 15.63 | 40.00 | oracle_path_success_rate=80.60; dist_to_end_reduction=6.76m; lengths=31.08m; Oracle-SPL=47.41 |
| SOON | val_unseen | 36.34 | 25.66 | 8.13 | 54.01 | 36.81 | 21.52 | 42.42 | oracle_error=4.52m; det_sr=0.00; det_spl=0.00; action_steps=12.68; steps=17.65; lengths=34.75m; Oracle-SPL=42.67; detection 扩展字段未激活 |

## Fine-metric standards
| dataset | standard | items | SR | OSR | delta_SR_vs_official | delta_OSR_vs_official | note |
|---|---|---:|---:|---:|---:|---:|---|
| R2R | official | 2349 | 76.29 | 84.80 | +0.00 | +0.00 | 与主指标一致 |
| R2R | eval_end_goal | 2349 | 76.29 | 84.80 | +0.00 | +0.00 | R2R 无 region 标准 |
| REVERIE | official | 3521 | 45.84 | 54.64 | +0.00 | +0.00 | 与主指标一致 |
| REVERIE | eval_end_goal | 3521 | 56.66 | 66.66 | +10.82 | +12.01 | 以单一 goal endpoint 计算，宽于 official object-visible region |
| REVERIE | eval_end_region | 3521 | 45.84 | 54.64 | +0.00 | +0.00 | 与 official 导航成功口径一致 |
| REVERIE | eval_end_region_threshold | 3521 | 58.51 | 67.96 | +12.67 | +13.32 | region + 3m 阈值，明显更宽 |
| CVDN | official | 907 | 24.04 | 53.03 | +0.00 | +0.00 | 与主指标一致；official_oracle_plan=80.60 |
| CVDN | eval_end_goal | 907 | 24.04 | 53.03 | +0.00 | +0.00 | 与 official final/oracle endpoint 口径一致 |
| CVDN | eval_end_region | 907 | 23.37 | 40.90 | -0.66 | -12.13 | region 口径更严格 |
| CVDN | eval_end_region_threshold | 907 | 31.09 | 55.13 | +7.06 | +2.09 | region + 3m 阈值，final success 更宽 |
| SOON | official | 3390 | 36.34 | 54.01 | +0.00 | +0.00 | 与主指标一致 |
| SOON | eval_end_goal | 3390 | 36.34 | 54.01 | +0.00 | +0.00 | 与 official final/oracle endpoint 口径一致 |
| SOON | eval_end_region | 3390 | 24.75 | 40.18 | -11.59 | -13.83 | region 口径更严格 |
| SOON | eval_end_region_threshold | 3390 | 41.39 | 57.46 | +5.04 | +3.45 | region + 3m 阈值，略宽于 official |

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
| eval(all 4 val_unseen splits) | 1030.55 |  | 0.101 s/sample | 10167 条结果; 包含模型加载、图构建与最短路预处理 |
| fine_metrics + plots |  |  |  | 基于 `eval_items` 后处理生成，未记录单独耗时 |

## Interpretation
- 是否对齐论文：R2R、REVERIE、SOON 的 `val_unseen` SR/SPL 与论文 headline 结果基本对齐，偏差都在 ±0.56 以内；CVDN 的论文 `GP` 与当前实现的 `dist_to_end_reduction` 对齐后低 0.18。
- 最大差距：REVERIE `val_unseen` 的 SR 为 -0.56，是当前和论文差距最大的已知有效对照项。
- 口径澄清：REVERIE 的 `rgs / rgspl` 与 SOON 的 `det_sr / det_spl` 在本次运行中虽被 evaluator 写出，但当前配置是默认主线 `run.py -> BaseTrainer -> DUETAgent -> VLNBert`，且 `feature.enable_og=false`，runtime 不产出 `pred_objid / pred_obj_direction`。因此这些 0 应解释为“object grounding / detection 扩展字段未激活”，不是论文主结果复现失败，也不应直接参与论文对照。
- 细粒度价值：本次新增 `eval_items` sidecars 与 `fine_metrics`，覆盖 4 个 val_unseen split、10167 条样本；后处理摘要显示 Oracle-SPL 分别为 R2R=79.27、REVERIE=43.73、CVDN=47.41、SOON=42.67，并已生成按 instruction token、action step、move step、path length 分桶的 SR/OSR 曲线图。不同 endpoint 标准会显著改变 REVERIE/CVDN/SOON 的 SR/OSR，因此后续分析必须显式声明采用哪套口径。
- 下一步：把 `eval_items` / `fine_metrics` schema 固定到文档和测试中；若后续需要 REVERIE/SOON 的 grounding/detection 指标可用，应在当前主线下补最小 object prediction 输出链路，而不是仅根据这次结果把模型能力判定为失败。

## Notes
- `peak_mem` 留空，因为归档里只有运行前的 `nvidia-smi` 快照，没有峰值显存监控结果。
- Comparison 只填写论文里已有且与当前主线口径一致的官方参考项；其余空白并不表示实验没有产出，而是当前没有可靠对照基线，或该指标属于未激活/细粒度扩展字段。
- `fine_metrics` 对应文件在 `experiment_outputs/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1/fine_metrics/`，可视化摘要在 `reports/artifacts/plots/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1_fine_metric_plots_summary.md`。
- `Fine-metric standards` 由 `fine_metrics_summary.json` 的计数字段临时计算得到：`SR = final_successes / items * 100`，`OSR = oracle_successes / items * 100`，`delta_*` 是相对同一 dataset 的 `official` 标准的百分点差；该 summary 不包含非 official 标准的 SPL，因此这里只比较 SR/OSR。
