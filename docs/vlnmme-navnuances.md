# VLN-MME via NavNuances

目标是让 VLN-MME 直接在 NavNuances 的 R2R 五类 split 上推理，并把结果转换成 NavNuances evaluator 的 `submit_*.json`。

这里不迁移 VLN-MME 的 `MP3D/`、`marked_obs/` 等运行时资源。VLN-MME runtime 仍由 `data/vlnmme` 挂载到 `third_party/VLN-MME/data`，而 NavNuances 样本单独安装到：

```text
data/navnuances/vlnmme/R2R/
```

## 准备数据

先确保 SAME 风格的 NavNuances enc 文件已经存在：

```bash
python scripts/setup/prepare_navnuances_same_r2r.py
```

再安装一份给 VLN-MME：

```bash
python scripts/setup/prepare_navnuances_vlnmme_r2r.py
```

默认会写入：

```text
data/navnuances/vlnmme/R2R/R2R_DC_enc.json
data/navnuances/vlnmme/R2R/R2R_LR_enc.json
data/navnuances/vlnmme/R2R/R2R_RR_enc.json
data/navnuances/vlnmme/R2R/R2R_VM_enc.json
data/navnuances/vlnmme/R2R/R2R_NU_enc.json
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
