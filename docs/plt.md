

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

多 split 实验默认会按 `dataset_split` 分组，不会把同一 dataset 的不同 split 混在一起:

```bash
conda activate plots
python scripts/plts/plot_same_fine_metric_diagnostics.py \
  --fine-metrics-dir experiment_outputs/0013_same_val_all_r2r_reverie_cvdn_soon_same_s0_v4/fine_metrics \
  --output-dir reports/artifacts/plots
```

如果只想画某一个 split，用 `--split` 过滤。此时默认按 dataset 分组:

```bash
python scripts/plts/plot_same_fine_metric_diagnostics.py \
  --fine-metrics-dir experiment_outputs/0013_same_val_all_r2r_reverie_cvdn_soon_same_s0_v4/fine_metrics \
  --output-dir reports/artifacts/plots \
  --split val_unseen
```

如需强制把多个 split 混合到 dataset 级别，可显式使用 `--group-by dataset`。

输出:

- `*_sr_osr_by_instruction_token_count.png`
- `*_sr_osr_by_action_step_count.png`
- `*_sr_osr_by_move_step_count.png`
- `*_sr_osr_by_path_length_m.png`
- `*_sr_osr_spl_oracle_spl_bar.png`
- `*_fine_metric_curve_points.csv`
- `*_fine_metric_summary.csv`
- `*_fine_metric_plots_summary.md`
