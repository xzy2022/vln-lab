## 示例模板
| Patch 文件名                                      | 目的                         | 类型   | 适用版本                     | 是否默认应用 | 备注                                      |
| ---------------------------------------------- | -------------------------- | ---- | ------------------------ | ------ | --------------------------------------- |
| 0001-parameterize-r2r-xl-validation.patch       | 参数化 R2R XL validation 脚本 | base | NavGPT-2 commit 1f535cb   | 是      | 支持外部数据、输出目录和运行参数覆盖 |

## 正文
| Patch 文件名                                                | 目的                         | 类型   | 适用版本                   | 是否默认应用 | 备注 |
| ---------------------------------------------------------- | -------------------------- | ---- | ---------------------- | ------ | ---- |
| base/0001-parameterize-r2r-xl-validation.patch             | 参数化 `val_r2r_xl.sh` 的路径和运行参数 | base | NavGPT-2 commit 1f535cb | 是      | 支持在 VLN lab 容器内通过环境变量指定数据、输出、checkpoint、batch size 和 CUDA 设备 |
| base/0002-eval-splits.patch                                | 参数化 validation split 列表 | base | NavGPT-2 commit 1f535cb | 是      | 支持通过 `--eval_splits DC LR RR VM NU` 跑 NavNuances 五类 split |

### 说明

- Patch 根目录是 `third_party/NavGPT-2/`。
- 默认 patch 放在 `patches/navgpt2/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method navgpt2`
  - `bash scripts/setup/revert_patches.sh --method navgpt2`
- `base/0001-parameterize-r2r-xl-validation.patch` 会让 `map_nav_src/scripts/val_r2r_xl.sh` 从脚本位置自动进入 `map_nav_src`，避免依赖调用时的当前目录。
- 该 patch 默认使用 `/workspace/vln-lab/data/navgpt2` 作为 `NAVGPT2_ASSET_ROOT`，并支持用环境变量覆盖 `DATA_ROOT`、`NAVGPT2_DATASETS`、`OUTPUT_ROOT`、`NAVGPT2_OUTPUT_ROOT`、`OUTPUT_DIR`、`QFORMER_CKPT`、`NAVGPT2_QFORMER_DIR`、`RESUME_FILE`、`BATCH_SIZE`、`NGPUS`、`SEED`、`CUDA_VISIBLE_DEVICES`、`PYTHON_BIN` 和 `EXTRA_ARGS`。
- `base/0002-eval-splits.patch` 会给 R2R evaluation 入口新增 `--eval_splits` 参数；应用 base patch 后可通过 `EXTRA_ARGS="--eval_splits DC LR RR VM NU"` 跑 NavNuances 五类 split。split 支持空格或逗号分隔。

- 应用 base patch 后，可用下面的命令把输出写入实验目录，并将 batch size 改为 1：

```bash
OUTPUT_DIR="/workspace/vln-lab/experiment_outputs/${RUN_ID}" \
BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash third_party/NavGPT-2/map_nav_src/scripts/val_r2r_xl.sh
```

- 如需验证 patch 栈，可在干净临时树里执行：

```bash
tmp=$(mktemp -d)
mkdir -p "$tmp/src"
git -C third_party/NavGPT-2 archive HEAD | tar -x -C "$tmp/src"
for patch in patches/navgpt2/base/*.patch; do
  git -C "$tmp/src" apply --check "$PWD/$patch"
  git -C "$tmp/src" apply "$PWD/$patch"
done
rm -rf "$tmp"
```
