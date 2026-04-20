# Error Card

- error_id: 0005-og-metrics-not-activated
- date: 2026-04-20
- experiment_id: 0005_same_val_r2r_reverie_cvdn_soon_same_s0_v1
- stage: eval
- error_type: semantic_metric_mismatch_for_object_grounding_fields
- stacktrace_summary: 无真实 Python stacktrace。现象是 REVERIE 的 `rgs / rgspl` 与 SOON 的 `det_sr / det_spl` 在 checkpoint evaluation 中恒为 0，容易被误读为“论文结果复现失败”。
- does_it_invalidate_metrics: partial
- root_cause_guess: 默认执行链路是 `run.py -> BaseTrainer -> DUETAgent -> VLNBert`。该主线能够正常产出导航相关指标，但不会在当前配置 `agent.type=duet`、`feature.enable_og=false` 下生成 evaluator 所需的 `pred_objid / pred_obj_direction`。因此 REVERIE / SOON 的 object grounding / detection 扩展字段虽被 evaluator 暴露并写入结果，却并不代表当前 runtime 真正激活了对应预测能力；这些 0 更应解释为“扩展字段未激活”，而不是“模型 grounding 能力完全失败”。
- tried_solutions: 1. 对照 `DUETAgent`、`BaseAgent.get_results()` 与 `navillm_agent.py`，确认默认主线不会写回 object prediction 字段。2. 检查 `reverie.py`、`soon.py` 的 evaluator，确认它们在缺少对象预测时仍会输出 `rgs / rgspl / det_sr / det_spl`。3. 回看 SAME 论文主结果口径，确认论文主表并未要求用这些附加指标做 headline 对照。4. 复核 CVDN evaluator，确认论文中的 `GP` 对应当前实现的 `dist_to_end_reduction`，这是一组命名映射，不属于本错误。
- final_solution: 将本问题记为“评测扩展字段未激活”的语义澄清，而不是训练/推理 bug。工程上保留 `metrics.json / metrics_long.csv` 中的原始 evaluator 输出，不删除这些 0；同时在 `scripts/experiments/run_same.py` 中补充语义告警，明确提示在 `agent.type=duet` 且 `feature.enable_og=false` 时，这些指标不可直接作为论文主结果对照；另外为 official reference 校验补上 CVDN `GP <-> dist_to_end_reduction` 的别名映射，避免把同义指标误报为缺参考项。
- regression_test: `python -m unittest tests.test_run_same.RunSameTests.test_build_metric_semantic_warnings_flags_unsupported_grounding_metrics tests.test_run_same.RunSameTests.test_check_official_references_treats_cvdn_gp_as_dist_to_end_reduction_alias`
- lesson: 需要区分“论文主线导航指标”和“仓库扩展 runtime 才会激活的 object grounding / detection 指标”。evaluator 提供某个字段，不代表默认 agent 一定会产出该字段所需的预测；同样，论文表中的指标命名也可能与当前代码实现存在一层别名映射，不能机械按字符串逐项比对。
