# L.01.08.VLN-MME Matrix 实验结果

## 目标 / 任务

为 `scripts/experiments/run_vlnmme_matrix.sh` 建立一份可持续追加的实验结果记录表。

本文件当前纳入统计的完整矩阵结果覆盖：

| 维度 | 取值 |
|---|---|
| Dataset | `val_unseen`, `navnuances` |
| Agent | `baseline_agent`, `mapgpt_agent` |
| Model | `r2r_qwen25vl_7b`, `r2r_qwen25vl_3b`, `r2r_qwen3vl_4b`, `r2r_qwen3vl_8b_instruct`, `r2r_qwen3vl_8b_thinking`, `r2r_qwen35_4b`, `r2r_qwen35_9b`, `r2r_internvl3_2b` |
| 总任务数 | 32 |

本文件记录 `experiment_outputs/vlnmme_matrix` 下 32 个已完成矩阵任务的结果。`val_unseen` 使用 VLN-MME 标准 R2R evaluator 写入的 `valid.txt`；`navnuances` 使用 `scripts/eval/run_vlnmme_navnuances_eval.py` 导出 `submit_{DC,LR,RR,VM,NU}.json` 后，由 NavNuances evaluator 生成的 `navnuances_eval/results.json`。

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
| VLN-MME matrix | `baseline_agent` | `r2r_qwen35_9b` | 33.68 | 16.67 | 25.40 | 29.09 | 9.41 | 3.02 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen35_9b` | 31.61 | 17.95 | 30.80 | 31.27 | 7.65 | 3.28 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen35_4b` | 31.61 | 8.97 | 19.56 | 17.45 | 7.65 | 2.77 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen35_4b` | 34.72 | 11.54 | 30.07 | 38.91 | 11.76 | 4.85 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen25vl_3b` | 54.92 | 20.51 | 49.49 | 68.00 | 30.00 | 16.65 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen25vl_3b` | 51.47 | 10.26 | 43.94 | 54.18 | 30.00 | 9.79 |
| VLN-MME matrix | `baseline_agent` | `r2r_internvl3_2b` | 58.20 | 7.69 | 46.72 | 46.91 | 17.06 | 8.85 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_internvl3_2b` | 51.64 | 7.69 | 37.52 | 54.18 | 16.47 | 7.79 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | 76.86 | 17.95 | 65.26 | 64.00 | 39.41 | 31.03 |
| VLN-MME matrix | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | 36.44 | 14.10 | 37.66 | 51.64 | 19.41 | 6.98 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | 74.61 | 16.67 | 60.88 | 66.91 | 42.35 | 32.44 |
| VLN-MME matrix | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | 39.38 | 11.54 | 36.93 | 42.55 | 20.00 | 7.28 |

NavGPT4v 官方 R2R aggregate 行同时报告 `R2R SR=41.30`、`R2R nDTW=54.78`、`R2R SPL=36.84`；上表的 val_unseen 只使用 R2R val_unseen SR。

## Matrix Run Tracker

| Dataset | Agent | Model | Status | Run / Alias | Result summary |
|---|---|---|---|---|---|
| `val_unseen` | `baseline_agent` | `r2r_qwen25vl_7b` | 已填 | `baseline_agent/r2r_qwen25vl_7b_s0` | SR 24.27, SPL 14.89, nDTW 34.20 |
| `val_unseen` | `baseline_agent` | `r2r_qwen25vl_3b` | 已填 | `baseline_agent/r2r_qwen25vl_3b_s0` | SR 16.65, SPL 8.15, nDTW 26.74 |
| `val_unseen` | `baseline_agent` | `r2r_qwen3vl_4b` | 已填 | `baseline_agent/r2r_qwen3vl_4b_s0` | SR 28.82, SPL 11.68, nDTW 27.90 |
| `val_unseen` | `baseline_agent` | `r2r_qwen35_9b` | 已填 | `baseline_agent/r2r_qwen35_9b_s0` | SR 3.02, SPL 2.64, nDTW 27.03 |
| `val_unseen` | `baseline_agent` | `r2r_qwen35_4b` | 已填 | `baseline_agent/r2r_qwen35_4b_s0` | SR 2.77, SPL 2.55, nDTW 26.69 |
| `val_unseen` | `baseline_agent` | `r2r_internvl3_2b` | 已填 | `baseline_agent/r2r_internvl3_2b_s0` | SR 8.85, SPL 3.36, nDTW 19.65 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_7b` | 已填 | `mapgpt_agent/r2r_qwen25vl_7b_s0` | SR 20.31, SPL 12.00, nDTW 31.88 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_3b` | 已填 | `mapgpt_agent/r2r_qwen25vl_3b_s0` | SR 9.79, SPL 5.71, nDTW 24.95 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_4b` | 已填 | `mapgpt_agent/r2r_qwen3vl_4b_s0` | SR 29.37, SPL 11.95, nDTW 29.48 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen35_9b` | 已填 | `mapgpt_agent/r2r_qwen35_9b_s0` | SR 3.28, SPL 3.11, nDTW 28.98 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen35_4b` | 已填 | `mapgpt_agent/r2r_qwen35_4b_s0` | SR 4.85, SPL 4.41, nDTW 28.38 |
| `val_unseen` | `mapgpt_agent` | `r2r_internvl3_2b` | 已填 | `mapgpt_agent/r2r_internvl3_2b_s0` | SR 7.79, SPL 2.88, nDTW 18.62 |
| `val_unseen` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | 已填 | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | SR 31.03, SPL 21.82, nDTW 40.88 |
| `val_unseen` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | 已填 | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | SR 6.98, SPL 5.78, nDTW 29.59 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | 已填 | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | SR 32.44, SPL 18.69, nDTW 34.11 |
| `val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | 已填 | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | SR 7.28, SPL 5.97, nDTW 30.17 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_7b` | 已填 | `baseline_agent/r2r_qwen25vl_7b_s0` | DC 62.00, LR 59.85, RR 72.36, VM 32.94, NU 11.54 |
| `navnuances` | `baseline_agent` | `r2r_qwen25vl_3b` | 已填 | `baseline_agent/r2r_qwen25vl_3b_s0` | DC 54.92, LR 49.49, RR 68.00, VM 30.00, NU 20.51 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_4b` | 已填 | `baseline_agent/r2r_qwen3vl_4b_s0` | DC 73.58, LR 61.17, RR 68.73, VM 37.65, NU 20.51 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_9b` | 已填 | `baseline_agent/r2r_qwen35_9b_s0` | DC 33.68, LR 25.40, RR 29.09, VM 9.41, NU 16.67 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_4b` | 已填 | `baseline_agent/r2r_qwen35_4b_s0` | DC 31.61, LR 19.56, RR 17.45, VM 7.65, NU 8.97 |
| `navnuances` | `baseline_agent` | `r2r_internvl3_2b` | 已填 | `baseline_agent/r2r_internvl3_2b_s0` | DC 58.20, LR 46.72, RR 46.91, VM 17.06, NU 7.69 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_7b` | 已填 | `mapgpt_agent/r2r_qwen25vl_7b_s0` | DC 58.55, LR 57.96, RR 63.27, VM 15.88, NU 7.69 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen25vl_3b` | 已填 | `mapgpt_agent/r2r_qwen25vl_3b_s0` | DC 51.47, LR 43.94, RR 54.18, VM 30.00, NU 10.26 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_4b` | 已填 | `mapgpt_agent/r2r_qwen3vl_4b_s0` | DC 72.54, LR 59.85, RR 62.91, VM 39.41, NU 21.79 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_9b` | 已填 | `mapgpt_agent/r2r_qwen35_9b_s0` | DC 31.61, LR 30.80, RR 31.27, VM 7.65, NU 17.95 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_4b` | 已填 | `mapgpt_agent/r2r_qwen35_4b_s0` | DC 34.72, LR 30.07, RR 38.91, VM 11.76, NU 11.54 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | 已填 | `mapgpt_agent/r2r_internvl3_2b_s0` | DC 51.64, LR 37.52, RR 54.18, VM 16.47, NU 7.69 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | 已填 | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | DC 76.86, LR 65.26, RR 64.00, VM 39.41, NU 17.95 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | 已填 | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | DC 36.44, LR 37.66, RR 51.64, VM 19.41, NU 14.10 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | 已填 | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | DC 74.61, LR 60.88, RR 66.91, VM 42.35, NU 16.67 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | 已填 | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | DC 39.38, LR 36.93, RR 42.55, VM 20.00, NU 11.54 |

## Standard R2R Results

| Dataset | Agent | Model | Run / Alias | action_steps | steps | lengths | nav_error | oracle_error | SR | oracle_SR | SPL | nDTW | SDTW | CLS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen25vl_7b` | `baseline_agent/r2r_qwen25vl_7b_s0` | 8.38 | 8.38 | 16.74 | 7.29 | 4.68 | 24.27 | 40.32 | 14.89 | 34.20 | 15.81 | 33.57 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen25vl_3b` | `baseline_agent/r2r_qwen25vl_3b_s0` | 8.55 | 8.55 | 16.89 | 8.56 | 5.20 | 16.65 | 35.12 | 8.15 | 26.74 | 8.80 | 29.11 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen3vl_4b` | `baseline_agent/r2r_qwen3vl_4b_s0` | 11.63 | 11.63 | 23.14 | 6.54 | 3.76 | 28.82 | 49.47 | 11.68 | 27.90 | 13.98 | 26.58 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen35_9b` | `baseline_agent/r2r_qwen35_9b_s0` | 1.58 | 1.58 | 3.15 | 9.59 | 8.42 | 3.02 | 4.13 | 2.64 | 27.03 | 2.29 | 27.69 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen35_4b` | `baseline_agent/r2r_qwen35_4b_s0` | 1.26 | 1.26 | 2.52 | 9.53 | 8.58 | 2.77 | 3.96 | 2.55 | 26.69 | 2.15 | 26.60 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_internvl3_2b` | `baseline_agent/r2r_internvl3_2b_s0` | 9.91 | 9.91 | 20.86 | 9.02 | 5.94 | 8.85 | 22.78 | 3.36 | 19.65 | 3.81 | 21.29 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_7b` | `mapgpt_agent/r2r_qwen25vl_7b_s0` | 7.98 | 7.98 | 16.35 | 7.48 | 5.14 | 20.31 | 33.38 | 12.00 | 31.88 | 12.74 | 31.20 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen25vl_3b` | `mapgpt_agent/r2r_qwen25vl_3b_s0` | 7.34 | 7.34 | 14.65 | 9.93 | 5.78 | 9.79 | 28.78 | 5.71 | 24.95 | 5.78 | 28.81 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_4b` | `mapgpt_agent/r2r_qwen3vl_4b_s0` | 11.21 | 11.21 | 22.49 | 6.42 | 3.98 | 29.37 | 47.94 | 11.95 | 29.48 | 14.63 | 26.44 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen35_9b` | `mapgpt_agent/r2r_qwen35_9b_s0` | 1.48 | 1.48 | 2.90 | 9.40 | 8.32 | 3.28 | 3.79 | 3.11 | 28.98 | 2.60 | 30.26 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen35_4b` | `mapgpt_agent/r2r_qwen35_4b_s0` | 2.38 | 2.38 | 4.77 | 9.80 | 7.99 | 4.85 | 7.07 | 4.41 | 28.38 | 3.82 | 30.86 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | 9.94 | 9.94 | 20.85 | 9.31 | 6.12 | 7.79 | 21.24 | 2.88 | 18.62 | 3.14 | 20.90 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | 8.13 | 8.13 | 16.17 | 6.48 | 3.94 | 31.03 | 48.49 | 21.82 | 40.88 | 21.95 | 40.46 |
| `R2R.val_unseen` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | 3.14 | 3.14 | 6.25 | 9.28 | 7.46 | 6.98 | 11.41 | 5.78 | 29.59 | 5.23 | 31.09 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | 10.26 | 10.26 | 20.61 | 6.64 | 3.70 | 32.44 | 52.02 | 18.69 | 34.11 | 20.16 | 33.05 |
| `R2R.val_unseen` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | 3.49 | 3.49 | 7.08 | 9.23 | 7.23 | 7.28 | 12.35 | 5.97 | 30.17 | 5.47 | 32.24 |

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
| `navnuances` | `baseline_agent` | `r2r_qwen35_9b` | `baseline_agent/r2r_qwen35_9b_s0` | DC | 579 | `sr` | 33.68 | `pair_sr` 0.52; `sr_left` 81.25; `sr_right` 2.08; `sr_around` 17.95 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_9b` | `baseline_agent/r2r_qwen35_9b_s0` | LR | 685 | `sr` | 25.40 | `success_past` 22.59; `success_towards` 28.05 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_9b` | `baseline_agent/r2r_qwen35_9b_s0` | RR | 275 | `sr` | 29.09 | `success_exit` 37.06; `success_into` 16.19; `oracle_success` 34.55 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_9b` | `baseline_agent/r2r_qwen35_9b_s0` | VM | 170 | `sr` | 9.41 | `oracle_sr` 11.18; `spl` 8.75; `nDTW` 29.27; `sr_double_dir` 9.09 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_9b` | `baseline_agent/r2r_qwen35_9b_s0` | NU | 78 | `path_SR` | 16.67 | `nDTW` 20.70 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_4b` | `baseline_agent/r2r_qwen35_4b_s0` | DC | 579 | `sr` | 31.61 | `pair_sr` 1.04; `sr_left` 79.17; `sr_right` 1.04; `sr_around` 14.87 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_4b` | `baseline_agent/r2r_qwen35_4b_s0` | LR | 685 | `sr` | 19.56 | `success_past` 13.86; `success_towards` 24.93 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_4b` | `baseline_agent/r2r_qwen35_4b_s0` | RR | 275 | `sr` | 17.45 | `success_exit` 20.00; `success_into` 13.33; `oracle_success` 18.91 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_4b` | `baseline_agent/r2r_qwen35_4b_s0` | VM | 170 | `sr` | 7.65 | `oracle_sr` 7.65; `spl` 7.62; `nDTW` 28.33; `sr_double_dir` 11.36 |
| `navnuances` | `baseline_agent` | `r2r_qwen35_4b` | `baseline_agent/r2r_qwen35_4b_s0` | NU | 78 | `path_SR` | 8.97 | `nDTW` 11.04 |
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
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_9b` | `mapgpt_agent/r2r_qwen35_9b_s0` | DC | 579 | `sr` | 31.61 | `pair_sr` 5.21; `sr_left` 78.65; `sr_right` 5.21; `sr_around` 11.28 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_9b` | `mapgpt_agent/r2r_qwen35_9b_s0` | LR | 685 | `sr` | 30.80 | `success_past` 22.29; `success_towards` 38.81 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_9b` | `mapgpt_agent/r2r_qwen35_9b_s0` | RR | 275 | `sr` | 31.27 | `success_exit` 32.35; `success_into` 29.52; `oracle_success` 34.55 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_9b` | `mapgpt_agent/r2r_qwen35_9b_s0` | VM | 170 | `sr` | 7.65 | `oracle_sr` 7.65; `spl` 7.22; `nDTW` 33.87; `sr_double_dir` 6.82 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_9b` | `mapgpt_agent/r2r_qwen35_9b_s0` | NU | 78 | `path_SR` | 17.95 | `nDTW` 19.06 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_4b` | `mapgpt_agent/r2r_qwen35_4b_s0` | DC | 579 | `sr` | 34.72 | `pair_sr` 0.00; `sr_left` 85.94; `sr_right` 0.00; `sr_around` 18.46 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_4b` | `mapgpt_agent/r2r_qwen35_4b_s0` | LR | 685 | `sr` | 30.07 | `success_past` 20.18; `success_towards` 39.38 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_4b` | `mapgpt_agent/r2r_qwen35_4b_s0` | RR | 275 | `sr` | 38.91 | `success_exit` 44.12; `success_into` 30.48; `oracle_success` 43.27 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_4b` | `mapgpt_agent/r2r_qwen35_4b_s0` | VM | 170 | `sr` | 11.76 | `oracle_sr` 15.88; `spl` 11.08; `nDTW` 31.99; `sr_double_dir` 9.09 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen35_4b` | `mapgpt_agent/r2r_qwen35_4b_s0` | NU | 78 | `path_SR` | 11.54 | `nDTW` 12.64 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | DC | 579 | `sr` | 51.64 | `pair_sr` 39.06; `sr_left` 70.31; `sr_right` 57.29; `sr_around` 27.69 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | LR | 685 | `sr` | 37.52 | `success_past` 25.60; `success_towards` 48.73 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | RR | 275 | `sr` | 54.18 | `success_exit` 50.59; `success_into` 60.00; `oracle_success` 81.82 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | VM | 170 | `sr` | 16.47 | `oracle_sr` 37.65; `spl` 6.76; `nDTW` 28.60; `sr_double_dir` 18.18 |
| `navnuances` | `mapgpt_agent` | `r2r_internvl3_2b` | `mapgpt_agent/r2r_internvl3_2b_s0` | NU | 78 | `path_SR` | 7.69 | `nDTW` 4.80 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | DC | 579 | `sr` | 76.86 | `pair_sr` 42.19; `sr_left` 88.02; `sr_right` 53.12; `sr_around` 89.23 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | LR | 685 | `sr` | 65.26 | `success_past` 43.98; `success_towards` 85.27 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | RR | 275 | `sr` | 64.00 | `success_exit` 50.00; `success_into` 86.67; `oracle_success` 70.91 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | VM | 170 | `sr` | 39.41 | `oracle_sr` 42.94; `spl` 37.55; `nDTW` 58.31; `sr_double_dir` 38.64 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_instruct` | `baseline_agent/r2r_qwen3vl_8b_instruct_s0` | NU | 78 | `path_SR` | 17.95 | `nDTW` 13.87 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | DC | 579 | `sr` | 36.44 | `pair_sr` 11.46; `sr_left` 55.73; `sr_right` 27.60; `sr_around` 26.15 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | LR | 685 | `sr` | 37.66 | `success_past` 23.80; `success_towards` 50.71 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | RR | 275 | `sr` | 51.64 | `success_exit` 48.24; `success_into` 57.14; `oracle_success` 59.64 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | VM | 170 | `sr` | 19.41 | `oracle_sr` 30.00; `spl` 16.67; `nDTW` 42.05; `sr_double_dir` 15.91 |
| `navnuances` | `baseline_agent` | `r2r_qwen3vl_8b_thinking` | `baseline_agent/r2r_qwen3vl_8b_thinking_s0` | NU | 78 | `path_SR` | 14.10 | `nDTW` 13.48 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | DC | 579 | `sr` | 74.61 | `pair_sr` 45.83; `sr_left` 86.98; `sr_right` 56.25; `sr_around` 80.51 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | LR | 685 | `sr` | 60.88 | `success_past` 39.16; `success_towards` 81.30 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | RR | 275 | `sr` | 66.91 | `success_exit` 54.12; `success_into` 87.62; `oracle_success` 81.09 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | VM | 170 | `sr` | 42.35 | `oracle_sr` 54.12; `spl` 36.68; `nDTW` 54.63; `sr_double_dir` 34.09 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | `mapgpt_agent/r2r_qwen3vl_8b_instruct_s0` | NU | 78 | `path_SR` | 16.67 | `nDTW` 7.05 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | DC | 579 | `sr` | 39.38 | `pair_sr` 18.75; `sr_left` 73.44; `sr_right` 27.08; `sr_around` 17.95 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | LR | 685 | `sr` | 36.93 | `success_past` 25.30; `success_towards` 47.88 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | RR | 275 | `sr` | 42.55 | `success_exit` 42.94; `success_into` 41.90; `oracle_success` 53.45 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | VM | 170 | `sr` | 20.00 | `oracle_sr` 28.82; `spl` 17.67; `nDTW` 39.25; `sr_double_dir` 11.36 |
| `navnuances` | `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | `mapgpt_agent/r2r_qwen3vl_8b_thinking_s0` | NU | 78 | `path_SR` | 11.54 | `nDTW` 13.96 |

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
| `baseline_agent` | `r2r_qwen35_9b` | DC | `sr` 33.6788; `sr_left` 81.2500; `sr_right` 2.0833; `sr_around` 17.9487; `pair_sr` 0.5208; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen35_9b` | LR | `sr` 25.4015; `success_towards` 28.0453; `success_past` 22.5904; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen35_9b` | RR | `sr` 29.0909; `oracle_success` 34.5455; `success_into` 16.1905; `oracle_success_into` 23.8095; `success_exit` 37.0588; `oracle_success_exit` 41.1765; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen35_9b` | VM | `sr` 9.4118; `oracle_sr` 11.1765; `spl` 8.7521; `nDTW` 29.2704; `sr_double_dir` 9.0909; `oracle_sr_double_dir` 11.3636; `spl_double_dir` 9.0639; `nDTW_double_dir` 26.9137; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen35_9b` | NU | `path_SR` 16.6667; `nDTW` 20.7021; `num_paths` 78 |
| `baseline_agent` | `r2r_qwen35_4b` | DC | `sr` 31.6062; `sr_left` 79.1667; `sr_right` 1.0417; `sr_around` 14.8718; `pair_sr` 1.0417; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen35_4b` | LR | `sr` 19.5620; `success_towards` 24.9292; `success_past` 13.8554; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen35_4b` | RR | `sr` 17.4545; `oracle_success` 18.9091; `success_into` 13.3333; `oracle_success_into` 16.1905; `success_exit` 20.0000; `oracle_success_exit` 20.5882; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen35_4b` | VM | `sr` 7.6471; `oracle_sr` 7.6471; `spl` 7.6243; `nDTW` 28.3341; `sr_double_dir` 11.3636; `oracle_sr_double_dir` 11.3636; `spl_double_dir` 11.3636; `nDTW_double_dir` 31.3585; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen35_4b` | NU | `path_SR` 8.9744; `nDTW` 11.0417; `num_paths` 78 |
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
| `mapgpt_agent` | `r2r_qwen35_9b` | DC | `sr` 31.6062; `sr_left` 78.6458; `sr_right` 5.2083; `sr_around` 11.2821; `pair_sr` 5.2083; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen35_9b` | LR | `sr` 30.8029; `success_towards` 38.8102; `success_past` 22.2892; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen35_9b` | RR | `sr` 31.2727; `oracle_success` 34.5455; `success_into` 29.5238; `oracle_success_into` 35.2381; `success_exit` 32.3529; `oracle_success_exit` 34.1176; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen35_9b` | VM | `sr` 7.6471; `oracle_sr` 7.6471; `spl` 7.2166; `nDTW` 33.8729; `sr_double_dir` 6.8182; `oracle_sr_double_dir` 6.8182; `spl_double_dir` 5.6469; `nDTW_double_dir` 30.0928; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen35_9b` | NU | `path_SR` 17.9487; `nDTW` 19.0630; `num_paths` 78 |
| `mapgpt_agent` | `r2r_qwen35_4b` | DC | `sr` 34.7150; `sr_left` 85.9375; `sr_right` 0.0000; `sr_around` 18.4615; `pair_sr` 0.0000; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen35_4b` | LR | `sr` 30.0730; `success_towards` 39.3768; `success_past` 20.1807; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen35_4b` | RR | `sr` 38.9091; `oracle_success` 43.2727; `success_into` 30.4762; `oracle_success_into` 34.2857; `success_exit` 44.1176; `oracle_success_exit` 48.8235; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen35_4b` | VM | `sr` 11.7647; `oracle_sr` 15.8824; `spl` 11.0756; `nDTW` 31.9865; `sr_double_dir` 9.0909; `oracle_sr_double_dir` 11.3636; `spl_double_dir` 9.0909; `nDTW_double_dir` 27.7643; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen35_4b` | NU | `path_SR` 11.5385; `nDTW` 12.6444; `num_paths` 78 |
| `mapgpt_agent` | `r2r_internvl3_2b` | DC | `sr` 51.6408; `sr_left` 70.3125; `sr_right` 57.2917; `sr_around` 27.6923; `pair_sr` 39.0625; `num_paths` 579 |
| `mapgpt_agent` | `r2r_internvl3_2b` | LR | `sr` 37.5182; `success_towards` 48.7252; `success_past` 25.6024; `num_paths` 685 |
| `mapgpt_agent` | `r2r_internvl3_2b` | RR | `sr` 54.1818; `oracle_success` 81.8182; `success_into` 60.0000; `oracle_success_into` 84.7619; `success_exit` 50.5882; `oracle_success_exit` 80.0000; `num_paths` 275 |
| `mapgpt_agent` | `r2r_internvl3_2b` | VM | `sr` 16.4706; `oracle_sr` 37.6471; `spl` 6.7650; `nDTW` 28.6022; `sr_double_dir` 18.1818; `oracle_sr_double_dir` 36.3636; `spl_double_dir` 7.0993; `nDTW_double_dir` 20.4225; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_internvl3_2b` | NU | `path_SR` 7.6923; `nDTW` 4.7999; `num_paths` 78 |
| `baseline_agent` | `r2r_qwen3vl_8b_instruct` | DC | `sr` 76.8566; `sr_left` 88.0208; `sr_right` 53.1250; `sr_around` 89.2308; `pair_sr` 42.1875; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen3vl_8b_instruct` | LR | `sr` 65.2555; `success_towards` 85.2691; `success_past` 43.9759; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen3vl_8b_instruct` | RR | `sr` 64.0000; `oracle_success` 70.9091; `success_into` 86.6667; `oracle_success_into` 88.5714; `success_exit` 50.0000; `oracle_success_exit` 60.0000; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen3vl_8b_instruct` | VM | `sr` 39.4118; `oracle_sr` 42.9412; `spl` 37.5538; `nDTW` 58.3063; `sr_double_dir` 38.6364; `oracle_sr_double_dir` 40.9091; `spl_double_dir` 35.4559; `nDTW_double_dir` 52.7362; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen3vl_8b_instruct` | NU | `nDTW` 13.8681; `path_SR` 17.9487; `num_paths` 78 |
| `baseline_agent` | `r2r_qwen3vl_8b_thinking` | DC | `sr` 36.4421; `sr_left` 55.7292; `sr_right` 27.6042; `sr_around` 26.1538; `pair_sr` 11.4583; `num_paths` 579 |
| `baseline_agent` | `r2r_qwen3vl_8b_thinking` | LR | `sr` 37.6642; `success_towards` 50.7082; `success_past` 23.7952; `num_paths` 685 |
| `baseline_agent` | `r2r_qwen3vl_8b_thinking` | RR | `sr` 51.6364; `oracle_success` 59.6364; `success_into` 57.1429; `oracle_success_into` 63.8095; `success_exit` 48.2353; `oracle_success_exit` 57.0588; `num_paths` 275 |
| `baseline_agent` | `r2r_qwen3vl_8b_thinking` | VM | `sr` 19.4118; `oracle_sr` 30.0000; `spl` 16.6684; `nDTW` 42.0464; `sr_double_dir` 15.9091; `oracle_sr_double_dir` 20.4545; `spl_double_dir` 14.8575; `nDTW_double_dir` 33.5346; `num_paths_double` 44; `num_paths` 170 |
| `baseline_agent` | `r2r_qwen3vl_8b_thinking` | NU | `nDTW` 13.4784; `path_SR` 14.1026; `num_paths` 78 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | DC | `sr` 74.6114; `sr_left` 86.9792; `sr_right` 56.2500; `sr_around` 80.5128; `pair_sr` 45.8333; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | LR | `sr` 60.8759; `success_towards` 81.3031; `success_past` 39.1566; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | RR | `sr` 66.9091; `oracle_success` 81.0909; `success_into` 87.6190; `oracle_success_into` 92.3810; `success_exit` 54.1176; `oracle_success_exit` 74.1176; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | VM | `sr` 42.3529; `oracle_sr` 54.1176; `spl` 36.6790; `nDTW` 54.6312; `sr_double_dir` 34.0909; `oracle_sr_double_dir` 47.7273; `spl_double_dir` 29.0771; `nDTW_double_dir` 47.6560; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_instruct` | NU | `nDTW` 7.0525; `path_SR` 16.6667; `num_paths` 78 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | DC | `sr` 39.3782; `sr_left` 73.4375; `sr_right` 27.0833; `sr_around` 17.9487; `pair_sr` 18.7500; `num_paths` 579 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | LR | `sr` 36.9343; `success_towards` 47.8754; `success_past` 25.3012; `num_paths` 685 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | RR | `sr` 42.5455; `oracle_success` 53.4545; `success_into` 41.9048; `oracle_success_into` 58.0952; `success_exit` 42.9412; `oracle_success_exit` 50.5882; `num_paths` 275 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | VM | `sr` 20.0000; `oracle_sr` 28.8235; `spl` 17.6665; `nDTW` 39.2457; `sr_double_dir` 11.3636; `oracle_sr_double_dir` 22.7273; `spl_double_dir` 9.7663; `nDTW_double_dir` 28.4638; `num_paths_double` 44; `num_paths` 170 |
| `mapgpt_agent` | `r2r_qwen3vl_8b_thinking` | NU | `nDTW` 13.9598; `path_SR` 11.5385; `num_paths` 78 |

## 来源记录

| 来源 | 内容 |
|---|---|
| `scripts/experiments/run_vlnmme_matrix.sh` | 矩阵维度、agent/model/dataset 组合、summary.tsv 口径 |
| `docs/vlnmme-navnuances.md` | `run_vlnmme_navnuances_eval.py` 导出并评估 NavNuances 的流程；最终指标以 `navnuances_eval/results.json` 为准 |
| `experiment_outputs/vlnmme_matrix/val_unseen/*/*_s0/*/*/valid.txt` | 16 个 `val_unseen` 矩阵任务的标准 R2R aggregate 指标 |
| `experiment_outputs/vlnmme_matrix/navnuances/*/*_s0/navnuances_eval/results.json` | 16 个 `navnuances` 矩阵任务的官方 NavNuances evaluator 指标 |
| `research/VLN research/local_traces/L.01.04.Ability Atlas 构建.md` | SAME / NavGPT4v / NavGPT-2 XL 横向对比基线 |

## 后续追加规则

1. 跑完一个矩阵任务后，先在 `Matrix Run Tracker` 中将对应行从 `待填` 改成 `已填` 或 `失败`。
2. `val_unseen` 任务追加到 `Standard R2R Results`，数据源为对应输出目录的 `valid.txt`。
3. `navnuances` 任务先按 `docs/vlnmme-navnuances.md` 运行 `scripts/eval/run_vlnmme_navnuances_eval.py`，再从 `navnuances_eval/results.json` 追加到 `NavNuances Skill Summary` 和 `NavNuances Detailed Metrics`。
4. 同一模型如果有多次 seed 或重跑，使用不同 `Run / Alias`，不要覆盖旧结果。
