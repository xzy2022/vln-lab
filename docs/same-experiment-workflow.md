# SAME 实验流水线

本文档描述 `scripts/experiments/run_same.py` 的最短使用路径。

## 1. 进入运行环境

```bash
cd /workspace/vln-lab
bash scripts/setup/run_container.sh
conda activate test-v1
```

## 2. 执行实验

最常见的评估命令：

```bash
cd /workspace/vln-lab
python scripts/experiments/run_same.py --config configs/same/val_r2r_eval_only.yaml
```

如需覆盖 seed、checkpoint 或附加 OmegaConf 选项：

```bash
python scripts/experiments/run_same.py \
  --config configs/same/val_r2r_eval_only.yaml \
  --seed 0 \
  --checkpoint ../../../data/same/ckpt/SAME.pt \
  --tag smoke \
  --option training.workers=2
```

## 3. 产物位置

一次成功运行后，结果会直接写到：

```text
experiments/<experiment_id>/
```

其中包含：

1. 归档文件：`run.json`、`config_resolved.yaml`、`metrics.json`、`stdout.log`、`stderr.log`、`git_info.txt`、`data_manifest.txt`、`gpu_info.txt`、`patch.diff`
2. SAME 原始产物：`<experiment_id>.log`、`results/`、`ckpts/`、`tensorboard/`

## 4. 长期表

运行器会自动维护：

```text
reports/tables/runs.csv
reports/tables/metrics_long.csv
```

并只读校验：

```text
reports/tables/official_results.csv
```

如果某个 `(method, dataset, split, metric)` 没有官方参考项，warning 会写入 `stderr.log` 和 `run.json`。
