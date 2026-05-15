# L.01.06.VLN-MME Matrix 实验结果

## 目标 / 任务

为 `scripts/experiments/run_vlnmme_matrix.sh` 建立一份可持续追加的实验结果记录表。

矩阵脚本当前覆盖：

| 维度 | 取值 |
|---|---|
| Dataset | `val_unseen`, `navnuances` |
| Agent | `baseline_agent`, `mapgpt_agent` |
| Model | `r2r_qwen25vl_7b`, `r2r_qwen25vl_3b`, `r2r_qwen3vl_4b`, `r2r_internvl3_2b` |
| 总任务数 | 16 |

本文件记录 `experiment_outputs/vlnmme_matrix` 下 16 个矩阵任务的结果。`val_unseen` 使用 VLN-MME 标准 R2R evaluator 写入的 `valid.txt`；`navnuances` 使用 `scripts/eval/run_vlnmme_navnuances_eval.py` 导出 `submit_{DC,LR,RR,VM,NU}.json` 后，由 NavNuances evaluator 生成的 `navnuances_eval/results.json`。

## 指标口径

| Dataset | 记录指标 | 说明 |
|---|---|---|
| `val_unseen` | 标准 R2R evaluator 输出 | `SR`、`SPL`、`nDTW`、`SDTW`、`CLS` 等 aggregate 指标 |
| `navnuances` | 五类 skill evaluator 输出 | DC/LR/RR/VM 使用 `sr` 作为主指标；NU 使用 `path_SR` 作为主指标 |

NavNuances skill 缩写：

| Skill | 含义 | 主指标 |
|---|---|---|
| DC | Direction Change | `sr` |
| LR | Landmark Recognition | `sr` |
| RR | Room Recognition | `sr` |
| VM | Vertical Movement | `sr` |
| NU | Numerical Directional Region | `path_SR` |

## 方法性能对比

指标口径：DC/LR/RR/VM 使用 SR，NU 使用 path_SR，val_unseen 使用标准 R2R val_unseen SR。SAME / NavGPT4v / NavGPT-2 XL 来自 `L.01.04.Ability Atlas 构建.md`；VLN-MME matrix 来自本文件记录的 `experiment_outputs/vlnmme_matrix`。

| Method | Agent | Model | DC | NU | LR | RR | VM | val_unseen SR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SAME | - | - | 63.56 | 33.33 | 31.39 | 89.82 | 85.88 | 76.29 |
| NavGPT4v (0-shot, paper) | - | - | 92.68 | 39.13 | 62.87 | 56.25 | 13.64 | 41.30 |
| NavGPT-2 XL (ours) | - | - | 62.52 | 25.64 | 33.58 | 77.45 | 83.53 | 69.65 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen25vl_7b` | 62.00 | 11.54 | 59.85 | 72.36 | 32.94 | 24.27 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen25vl_7b` | 58.55 | 7.69 | 57.96 | 63.27 | 15.88 | 20.31 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen3vl_4b` | 73.58 | 20.51 | 61.17 | 68.73 | 37.65 | 28.82 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen3vl_4b` | 72.54 | 21.79 | 59.85 | 62.91 | 39.41 | 29.37 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen25vl_3b` | 54.92 | 20.51 | 49.49 | 68.00 | 30.00 | 16.65 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen25vl_3b` | 51.47 | 10.26 | 43.94 | 54.18 | 30.00 | 9.79 |
| VLN-MME matrix | `baseline_agent` | `r2r_internvl3_2b` | 58.20 | 7.69 | 46.72 | 46.91 | 17.06 | 8.85 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_internvl3_2b` | 51.64 | 7.69 | 37.52 | 54.18 | 16.47 | 7.79 |

NavGPT4v 官方 R2R aggregate 行同时报告 `R2R SR=41.30`、`R2R nDTW=54.78`、`R2R SPL=36.84`；上表的 val_unseen 只使用 R2R val_unseen SR。

## Matrix Run Tracker

| Dataset | Agent | Model | Status | Run / Alias | Result summary |
|---|---|---|---|---|---|
| `val_unseen` | `baseline_agent` | `r2r_qwen25vl_7b` | 已填 | `baseline_agent/r2r_qwen25vl_7b_s0` | SR 24.27, SPL 14.89, nDTW 34.20 |
| `val_unseen` | `baseline_agent` | `r2r_qwen25vl_3b` | 已填 | `baseline_agent/r2r_qwen25vl_3b_s0` | SR 16.65, SPL 8.15, nDTW 26.74 |
| `val_unseen` | `baseline_agent` | `r2r_qwen3vl_4b` | 已填 | `baseline_agent/r2r_qwen3vl_4b_s0` | SR 28.82, SPL 11.68, nDTW 27.90 |
| `val_unseen` | `baseline_agent` | `r2r_internvl3_2b` | 已填 | `baseline_agent/r2r_internvl3_2b_s0` | SR 8.85, SPL 3.36, nDTW 19.65 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_7b` | 已填 | `mapgpt_agent/r2r_qwen25vl_7b_s0` | SR 20.31, SPL 12.00, nDTW 31.88 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_3b` | 已填 | `mapgpt_agent/r2r_qwen25vl_3b_s0` | SR 9.79, SPL 5.71, nDTW 24.95 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_4b` | 已填 | `mapgpt_agent/r2r_qwen3vl_4b_s0` | SR 29.37, SPL 11.95, nDTW 29.48 |
| `val_unseen` | `mapgpt_agent` | `r2r_internvl3_2b` | 已填 | `mapgpt_agent/r2r_internvl3_2b_s0` | SR 7.79, SPL 2.88, nDTW 18.62 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | 已填 | `baseline_agent/r2r_qwen25vl_7b_s0` | DC 62.00, LR 59.85, RR 72.36, VM 32.94, NU 11.54 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | 已填 | `baseline_agent/r2r_qwen25vl_3b_s0` | DC 54.92, LR 49.49, RR 68.00, VM 30.00, NU 20.51 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | 已填 | `baseline_agent/r2r_qwen3vl_4b_s0` | DC 73.58, LR 61.17, RR 68.73, VM 37.65, NU 20.51 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | 已填 | `baseline_agent/r2r_internvl3_2b_s0` | DC 58.20, LR 46.72, RR 46.91, VM 17.06, NU 7.69 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | 已填 | `mapgpt_agent/r2r_qwen25vl_7b_s0` | DC 58.55, LR 57.96, RR 63.27, VM 15.88, NU 7.69 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | 已填 | `mapgpt_agent/r2r_qwen25vl_3b_s0` | DC 51.47, LR 43.94, RR 54.18, VM 30.00, NU 10.26 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | 已填 | `mapgpt_agent/r2r_qwen3vl_4b_s0` | DC 72.54, LR 59.85, RR 62.91, VM 39.41, NU 21.79 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | 已填 | `mapgpt_agent/r2r_internvl3_2b_s0` | DC 51.64, LR 37.52, RR 54.18, VM 16.47, NU 7.69 |

## Standard R2R Results

| Dataset | Agent | Model | Run / Alias | action_steps | steps | lengths | nav_error | oracle_error | SR | oracle_SR | SPL | nDTW | SDTW | CLS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | 8.38 | 8.38 | 16.74 | 7.29 | 4.68 | 24.27 | 40.32 | 14.89 | 34.20 | 15.81 | 33.57 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | 8.55 | 8.55 | 16.89 | 8.56 | 5.20 | 16.65 | 35.12 | 8.15 | 26.74 | 8.80 | 29.11 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | 11.63 | 11.63 | 23.14 | 6.54 | 3.76 | 28.82 | 49.47 | 11.68 | 27.90 | 13.98 | 26.58 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | 9.91 | 9.91 | 20.86 | 9.02 | 5.94 | 8.85 | 22.78 | 3.36 | 19.65 | 3.81 | 21.29 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | 7.98 | 7.98 | 16.35 | 7.48 | 5.14 | 20.31 | 33.38 | 12.00 | 31.88 | 12.74 | 31.20 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | 7.34 | 7.34 | 14.65 | 9.93 | 5.78 | 9.79 | 28.78 | 5.71 | 24.95 | 5.78 | 28.81 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | 11.21 | 11.21 | 22.49 | 6.42 | 3.98 | 29.37 | 47.94 | 11.95 | 29.48 | 14.63 | 26.44 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | 9.94 | 9.94 | 20.85 | 9.31 | 6.12 | 7.79 | 21.24 | 2.88 | 18.62 | 3.14 | 20.90 |

## NavNuances Skill Summary

| Dataset | Agent | Model | Run / Alias | Skill | 样本数 | 主指标 | 主指标值 | 补充指标 |
|---|---|---|---|---|---:|---|---:|---|
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | DC | 579 | `sr` | 62.00 | `pair_sr` 61.46; `sr_left` 83.85; `sr_right` 75.52; `sr_around` 27.18 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | LR | 685 | `sr` | 59.85 | `success_past` 41.27; `success_towards` 77.34 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | RR | 275 | `sr` | 72.36 | `success_exit` 68.24; `success_into` 79.05; `oracle_success` 85.45 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | VM | 170 | `sr` | 32.94 | `oracle_sr` 34.12; `spl` 31.28; `nDTW` 53.67; `sr_double_dir` 34.09 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | NU | 78 | `path_SR` | 11.54 | `nDTW` 9.36 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | DC | 579 | `sr` | 54.92 | `pair_sr` 52.08; `sr_left` 77.08; `sr_right` 73.44; `sr_around` 14.87 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | LR | 685 | `sr` | 49.49 | `success_past` 36.45; `success_towards` 61.76 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | RR | 275 | `sr` | 68.00 | `success_exit` 65.29; `success_into` 72.38; `oracle_success` 81.82 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | VM | 170 | `sr` | 30.00 | `oracle_sr` 55.88; `spl` 21.44; `nDTW` 44.25; `sr_double_dir` 22.73 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | NU | 78 | `path_SR` | 20.51 | `nDTW` 12.85 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | DC | 579 | `sr` | 73.58 | `pair_sr` 51.56; `sr_left` 87.50; `sr_right` 61.46; `sr_around` 71.79 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | LR | 685 | `sr` | 61.17 | `success_past` 42.17; `success_towards` 79.04 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | RR | 275 | `sr` | 68.73 | `success_exit` 57.65; `success_into` 86.67; `oracle_success` 82.18 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | VM | 170 | `sr` | 37.65 | `oracle_sr` 61.18; `spl` 26.74; `nDTW` 48.05; `sr_double_dir` 40.91 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | NU | 78 | `path_SR` | 20.51 | `nDTW` 6.58 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | DC | 579 | `sr` | 58.20 | `pair_sr` 53.65; `sr_left` 78.12; `sr_right` 68.23; `sr_around` 28.72 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | LR | 685 | `sr` | 46.72 | `success_past` 32.53; `success_towards` 60.06 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | RR | 275 | `sr` | 46.91 | `success_exit` 32.94; `success_into` 69.52; `oracle_success` 79.27 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | VM | 170 | `sr` | 17.06 | `oracle_sr` 30.00; `spl` 9.51; `nDTW` 32.86; `sr_double_dir` 20.45 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | NU | 78 | `path_SR` | 7.69 | `nDTW` 4.15 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | DC | 579 | `sr` | 58.55 | `pair_sr` 50.52; `sr_left` 80.73; `sr_right` 68.75; `sr_around` 26.67 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | LR | 685 | `sr` | 57.96 | `success_past` 30.72; `success_towards` 83.57 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | RR | 275 | `sr` | 63.27 | `success_exit` 52.94; `success_into` 80.00; `oracle_success` 81.45 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | VM | 170 | `sr` | 15.88 | `oracle_sr` 17.06; `spl` 15.50; `nDTW` 43.22; `sr_double_dir` 15.91 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | NU | 78 | `path_SR` | 7.69 | `nDTW` 6.67 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | DC | 579 | `sr` | 51.47 | `pair_sr` 45.83; `sr_left` 76.56; `sr_right` 66.67; `sr_around` 11.79 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | LR | 685 | `sr` | 43.94 | `success_past` 32.83; `success_towards` 54.39 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | RR | 275 | `sr` | 54.18 | `success_exit` 52.35; `success_into` 57.14; `oracle_success` 73.45 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | VM | 170 | `sr` | 30.00 | `oracle_sr` 57.65; `spl` 21.37; `nDTW` 42.89; `sr_double_dir` 27.27 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | NU | 78 | `path_SR` | 10.26 | `nDTW` 14.26 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | DC | 579 | `sr` | 72.54 | `pair_sr` 48.96; `sr_left` 88.54; `sr_right` 59.38; `sr_around` 69.74 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | LR | 685 | `sr` | 59.85 | `success_past` 43.37; `success_towards` 75.35 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | RR | 275 | `sr` | 62.91 | `success_exit` 50.00; `success_into` 83.81; `oracle_success` 81.09 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | VM | 170 | `sr` | 39.41 | `oracle_sr` 58.82; `spl` 20.70; `nDTW` 43.93; `sr_double_dir` 38.64 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | NU | 78 | `path_SR` | 21.79 | `nDTW` 7.58 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | DC | 579 | `sr` | 51.64 | `pair_sr` 39.06; `sr_left` 70.31; `sr_right` 57.29; `sr_around` 27.69 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | LR | 685 | `sr` | 37.52 | `success_past` 25.60; `success_towards` 48.73 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | RR | 275 | `sr` | 54.18 | `success_exit` 50.59; `success_into` 60.00; `oracle_success` 81.82 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | VM | 170 | `sr` | 16.47 | `oracle_sr` 37.65; `spl` 6.76; `nDTW` 28.60; `sr_double_dir` 18.18 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | NU | 78 | `path_SR` | 7.69 | `nDTW` 4.80 |

## NavNuances Detailed Metrics

| Agent | Model | Skill | Metrics |
|---|---|---|---|
| `baseline_agent` | `r2r_qwen25vl_7b` | DC | `sr` 62.0035; `sr_left` 83.8542; `sr_right` 75.5208; `sr_around` 27.1795; `pair_sr` 61.4583; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen25vl_7b` | LR | `sr` 59.8540; `success_towards` 77.3371; `success_past` 41.2651; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen25vl_7b` | RR | `sr` 72.3636; `oracle_success` 85.4545; `success_into` 79.0476; `oracle_success_into` 86.6667; `success_exit` 68.2353; `oracle_success_exit` 84.7059; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen25vl_7b` | VM | `sr` 32.9412; `oracle_sr` 34.1176; `spl` 31.2803; `nDTW` 53.6682; `sr_double_dir` 34.0909; `oracle_sr_double_dir` 34.0909; `spl_double_dir` 31.1999; `nDTW_double_dir` 48.1929; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen25vl_7b` | NU | `path_SR` 11.5385; `nDTW` 9.3594; `num_paths` 78 |
| `baseline_agent` | `r2r_qwen25vl_3b` | DC | `sr` 54.9223; `sr_left` 77.0833; `sr_right` 73.4375; `sr_around` 14.8718; `pair_sr` 52.0833; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen25vl_3b` | LR | `sr` 49.4891; `success_towards` 61.7564; `success_past` 36.4458; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen25vl_3b` | RR | `sr` 68.0000; `oracle_success` 81.8182; `success_into` 72.3810; `oracle_success_into` 90.4762; `success_exit` 65.2941; `oracle_success_exit` 76.4706; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen25vl_3b` | VM | `sr` 30.0000; `oracle_sr` 55.8824; `spl` 21.4428; `nDTW` 44.2516; `sr_double_dir` 22.7273; `oracle_sr_double_dir` 45.4545; `spl_double_dir` 18.4506; `nDTW_double_dir` 34.8702; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen25vl_3b` | NU | `path_SR` 20.5128; `nDTW` 12.8501; `num_paths` 78 |
| `baseline_agent` | `r2r_qwen3vl_4b` | DC | `sr` 73.5751; `sr_left` 87.5000; `sr_right` 61.4583; `sr_around` 71.7949; `pair_sr` 51.5625; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen3vl_4b` | LR | `sr` 61.1679; `success_towards` 79.0368; `success_past` 42.1687; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen3vl_4b` | RR | `sr` 68.7273; `oracle_success` 82.1818; `success_into` 86.6667; `oracle_success_into` 90.4762; `success_exit` 57.6471; `oracle_success_exit` 77.0588; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen3vl_4b` | VM | `sr` 37.6471; `oracle_sr` 61.1765; `spl` 26.7420; `nDTW` 48.0453; `sr_double_dir` 40.9091; `oracle_sr_double_dir` 56.8182; `spl_double_dir` 29.1324; `nDTW_double_dir` 40.3286; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen3vl_4b` | NU | `path_SR` 20.5128; `nDTW` 6.5824; `num_paths` 78 |
| `baseline_agent` | `r2r_internvl3_2b` | DC | `sr` 58.2038; `sr_left` 78.1250; `sr_right` 68.2292; `sr_around` 28.7179; `pair_sr` 53.6458; `num_paths` 579 |
| `baseline_agent` | `r2r_internvl3_2b` | LR | `sr` 46.7153; `success_towards` 60.0567; `success_past` 32.5301; `num_paths` 685 |
| `baseline_agent` | `r2r_internvl3_2b` | RR | `sr` 46.9091; `oracle_success` 79.2727; `success_into` 69.5238; `oracle_success_into` 89.5238; `success_exit` 32.9412; `oracle_success_exit` 72.9412; `num_paths` 275 |
| `baseline_agent` | `r2r_internvl3_2b` | VM | `sr` 17.0588; `oracle_sr` 30.0000; `spl` 9.5102; `nDTW` 32.8645; `sr_double_dir` 20.4545; `oracle_sr_double_dir` 31.8182; `spl_double_dir` 11.1851; `nDTW_double_dir` 26.5998; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_internvl3_2b` | NU | `path_SR` 7.6923; `nDTW` 4.1495; `num_paths` 78 |
| `mapgpt_agent` | `r2r_qwen25vl_7b` | DC | `sr` 58.5492; `sr_left` 80.7292; `sr_right` 68.7500; `sr_around` 26.6667; `pair_sr` 50.5208; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen25vl_7b` | LR | `sr` 57.9562; `success_towards` 83.5694; `success_past` 30.7229; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen25vl_7b` | RR | `sr` 63.2727; `oracle_success` 81.4545; `success_into` 80.0000; `oracle_success_into` 83.8095; `success_exit` 52.9412; `oracle_success_exit` 80.0000; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen25vl_7b` | VM | `sr` 15.8824; `oracle_sr` 17.0588; `spl` 15.5044; `nDTW` 43.2176; `sr_double_dir` 15.9091; `oracle_sr_double_dir` 15.9091; `spl_double_dir` 15.8272; `nDTW_double_dir` 42.4110; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen25vl_7b` | NU | `path_SR` 7.6923; `nDTW` 6.6696; `num_paths` 78 |
| `mapgpt_agent` | `r2r_qwen25vl_3b` | DC | `sr` 51.4680; `sr_left` 76.5625; `sr_right` 66.6667; `sr_around` 11.7949; `pair_sr` 45.8333; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen25vl_3b` | LR | `sr` 43.9416; `success_towards` 54.3909; `success_past` 32.8313; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen25vl_3b` | RR | `sr` 54.1818; `oracle_success` 73.4545; `success_into` 57.1429; `oracle_success_into` 88.5714; `success_exit` 52.3529; `oracle_success_exit` 64.1176; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen25vl_3b` | VM | `sr` 30.0000; `oracle_sr` 57.6471; `spl` 21.3652; `nDTW` 42.8928; `sr_double_dir` 27.2727; `oracle_sr_double_dir` 43.1818; `spl_double_dir` 20.3289; `nDTW_double_dir` 41.0936; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen25vl_3b` | NU | `path_SR` 10.2564; `nDTW` 14.2616; `num_paths` 78 |
| `mapgpt_agent` | `r2r_qwen3vl_4b` | DC | `sr` 72.5389; `sr_left` 88.5417; `sr_right` 59.3750; `sr_around` 69.7436; `pair_sr` 48.9583; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen3vl_4b` | LR | `sr` 59.8540; `success_towards` 75.3541; `success_past` 43.3735; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen3vl_4b` | RR | `sr` 62.9091; `oracle_success` 81.0909; `success_into` 83.8095; `oracle_success_into` 89.5238; `success_exit` 50.0000; `oracle_success_exit` 75.8824; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen3vl_4b` | VM | `sr` 39.4118; `oracle_sr` 58.8235; `spl` 20.6980; `nDTW` 43.9315; `sr_double_dir` 38.6364; `oracle_sr_double_dir` 54.5455; `spl_double_dir` 19.7647; `nDTW_double_dir` 36.7110; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen3vl_4b` | NU | `path_SR` 21.7949; `nDTW` 7.5793; `num_paths` 78 |
| `mapgpt_agent` | `r2r_internvl3_2b` | DC | `sr` 51.6408; `sr_left` 70.3125; `sr_right` 57.2917; `sr_around` 27.6923; `pair_sr` 39.0625; `num_paths` 579 |
| `mapgpt_agent` | `r2r_internvl3_2b` | LR | `sr` 37.5182; `success_towards` 48.7252; `success_past` 25.6024; `num_paths` 685 |
| `mapgpt_agent` | `r2r_internvl3_2b` | RR | `sr` 54.1818; `oracle_success` 81.8182; `success_into` 60.0000; `oracle_success_into` 84.7619; `success_exit` 50.5882; `oracle_success_exit` 80.0000; `num_paths` 275 |
| `mapgpt_agent` | `r2r_internvl3_2b` | VM | `sr` 16.4706; `oracle_sr` 37.6471; `spl` 6.7650; `nDTW` 28.6022; `sr_double_dir` 18.1818; `oracle_sr_double_dir` 36.3636; `spl_double_dir` 7.0993; `nDTW_double_dir` 20.4225; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_internvl3_2b` | NU | `path_SR` 7.6923; `nDTW` 4.7999; `num_paths` 78 |

## 来源记录

| 来源 | 内容 |
|---|---|
| `scripts/experiments/run_vlnmme_matrix.sh` | 矩阵维度、agent/model/dataset 组合、summary.tsv 口径 |
| `docs/vlnmme-navnuances.md` | `run_vlnmme_navnuances_eval.py` 导出并评估 NavNuances 的流程；最终指标以 `navnuances_eval/results.json` 为准 |
| `experiment_outputs/vlnmme_matrix/val_unseen/*/*_s0/*/*/valid.txt` | 8 个 `val_unseen` 矩阵任务的标准 R2R aggregate 指标 |
| `experiment_outputs/vlnmme_matrix/navnuances/*/*_s0/navnuances_eval/results.json` | 8 个 `navnuances` 矩阵任务的官方 NavNuances evaluator 指标 |
| `research/VLN research/local_traces/L.01.04.Ability Atlas 构建.md` | SAME / NavGPT4v / NavGPT-2 XL 横向对比基线 |

## 后续追加规则

1. 跑完一个矩阵任务后，先在 `Matrix Run Tracker` 中将对应行从 `待填` 改成 `已填` 或 `失败`。
2. `val_unseen` 任务追加到 `Standard R2R Results`，数据源为对应输出目录的 `valid.txt`。
3. `navnuances` 任务先按 `docs/vlnmme-navnuances.md` 运行 `scripts/eval/run_vlnmme_navnuances_eval.py`，再从 `navnuances_eval/results.json` 追加到 `NavNuances Skill Summary` 和 `NavNuances Detailed Metrics`。
4. 同一模型如果有多次 seed 或重跑，使用不同 `Run / Alias`，不要覆盖旧结果。
