# VLN-MME via NavNuances

目标是让 VLN-MME 直接在 NavNuances 的 R2R 五类 split 上推理，并把结果转换成 NavNuances evaluator 的 `submit_*.json`。

这里不迁移 VLN-MME 的 `MP3D/`、`marked_obs/` 等运行时资源。VLN-MME runtime 仍由 `data/vlnmme` 挂载到 `third_party/VLN-MME/data`，而 NavNuances 样本单独安装到：

```text
<paths.mounts.datasets>/navnuances/vlnmme/R2R/
```

## 准备数据

先选定和当前容器挂载一致的全局路径配置。比如当前容器里数据在 `/workspace/vln-lab/external/datasets`，就使用：

```bash
PATHS_CONFIG=configs/global/local/paths.yaml
```

如果容器是用 lab profile 创建的，则使用：

```bash
PATHS_CONFIG=configs/global/lab/paths.yaml
```

先确保 SAME 风格的 NavNuances enc 文件已经存在：

```bash
python scripts/setup/prepare_navnuances_same_r2r.py \
  --paths-config "${PATHS_CONFIG}"
```

再安装一份给 VLN-MME：

```bash
python scripts/setup/prepare_navnuances_vlnmme_r2r.py \
  --paths-config "${PATHS_CONFIG}"
```

默认会写入：

```text
<paths.mounts.datasets>/navnuances/vlnmme/R2R/R2R_DC_enc.json
<paths.mounts.datasets>/navnuances/vlnmme/R2R/R2R_LR_enc.json
<paths.mounts.datasets>/navnuances/vlnmme/R2R/R2R_RR_enc.json
<paths.mounts.datasets>/navnuances/vlnmme/R2R/R2R_VM_enc.json
<paths.mounts.datasets>/navnuances/vlnmme/R2R/R2R_NU_enc.json
```

## 推理

```bash
CONFIG_PATH=/workspace/vln-lab/configs/vlnmme/r2r_internvl3_2b_navnuances.yaml \
bash /workspace/vln-lab/scripts/experiments/run_vlnmme_resume.sh
```

VLN-MME 原始结果会写到：

```text
experiment_outputs/vlnmme_internvl3_2b_navnuances_r2r_s0/
└── baseline_agent/
    └── internvl3_2b/
        ├── R2R.R2R_DC.json
        ├── R2R.R2R_LR.json
        ├── R2R.R2R_RR.json
        ├── R2R.R2R_VM.json
        └── R2R.R2R_NU.json
```

该运行器支持断点继续：每次启动都会先合并已经写出的 partial results，再只生成未完成样本的临时 split。中断、电源断开或容器退出后，重新执行同一条命令即可继续。临时文件放在：

```text
experiment_outputs/vlnmme_internvl3_2b_navnuances_r2r_s0/resume_work/
```

## 导出并评估

在 NavNuances evaluator 容器里运行：

```bash
python scripts/eval/run_vlnmme_navnuances_eval.py \
  --experiment-id vlnmme_internvl3_2b_navnuances_r2r_s0
```

输出：

```text
experiment_outputs/vlnmme_internvl3_2b_navnuances_r2r_s0/navnuances_submission/submit_{DC,LR,RR,VM,NU}.json
experiment_outputs/vlnmme_internvl3_2b_navnuances_r2r_s0/navnuances_eval/results.json
```

最终指标以 `navnuances_eval/results.json` 为准。

## 批量评估 `experiment_outputs/vlnmme_matrix`

`run_vlnmme_matrix.sh` 的两个 dataset 需要分开看：

- `experiment_outputs/vlnmme_matrix/val_unseen/...`：`run_vlnmme_resume.sh` 在 split 完成后会自动跑一次 `valid_from_file`，指标已经写进各实验目录下的 `valid.txt`。
- `experiment_outputs/vlnmme_matrix/navnuances/...`：需要再跑一次 `scripts/eval/run_vlnmme_navnuances_eval.py`，把 `R2R.R2R_{DC,LR,RR,VM,NU}.json` 导出成 `submit_*.json` 并调用 NavNuances evaluator。

批量跑 NavNuances evaluator：

```bash
for exp_dir in experiment_outputs/vlnmme_matrix/navnuances/*/*_s0; do
  [[ -d "${exp_dir}" ]] || continue

  if [[ -f "${exp_dir}/navnuances_eval/results.json" ]]; then
    echo "[skip] already evaluated: ${exp_dir}"
    continue
  fi

  complete=1
  for split in DC LR RR VM NU; do
    if ! compgen -G "${exp_dir}"'/*/*/R2R.R2R_'"${split}"'.json' > /dev/null; then
      complete=0
      break
    fi
  done

  if [[ "${complete}" -ne 1 ]]; then
    echo "[skip] incomplete NavNuances results: ${exp_dir}"
    continue
  fi

  python scripts/eval/run_vlnmme_navnuances_eval.py \
    --experiment-dir "${exp_dir}"
done
```

如果你现在只想处理这次 matrix 的部分模型，可以先限制目录范围。例如只看：

```bash
for exp_dir in \
  experiment_outputs/vlnmme_matrix/navnuances/*/r2r_qwen35_0_8b_s0 \
  experiment_outputs/vlnmme_matrix/navnuances/*/r2r_qwen35_4b_s0 \
  experiment_outputs/vlnmme_matrix/navnuances/*/r2r_qwen35_9b_s0 \
  experiment_outputs/vlnmme_matrix/navnuances/*/r2r_qwen3vl_8b_instruct_s0 \
  experiment_outputs/vlnmme_matrix/navnuances/*/r2r_qwen3vl_8b_thinking_s0
do
  [[ -d "${exp_dir}" ]] || continue
  python scripts/eval/run_vlnmme_navnuances_eval.py \
    --experiment-dir "${exp_dir}"
done
```

其中：

- `val_unseen` 结果看 `experiment_outputs/vlnmme_matrix/<dataset>/<agent>/<model>_s0/*/*/valid.txt`
- `navnuances` 结果看 `experiment_outputs/vlnmme_matrix/<dataset>/<agent>/<model>_s0/navnuances_eval/results.json`
