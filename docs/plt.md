

使用本机的 `conda activate plots` 环境.

## SAME fine metrics 绘图

输入数据:

```text
experiment_outputs/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1/fine_metrics/tables/fine_metrics_wide.csv
```

运行:

```bash
conda activate plots
python scripts/plts/plot_same_fine_metric_diagnostics.py \
  --fine-metrics-dir experiment_outputs/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1/fine_metrics \
  --output-dir reports/artifacts/plots
```

输出:

- `*_sr_osr_by_instruction_token_count.png`
- `*_sr_osr_by_action_step_count.png`
- `*_sr_osr_by_move_step_count.png`
- `*_sr_osr_by_path_length_m.png`
- `*_sr_osr_spl_oracle_spl_bar.png`
- `*_fine_metric_curve_points.csv`
- `*_fine_metric_summary.csv`
- `*_fine_metric_plots_summary.md`
