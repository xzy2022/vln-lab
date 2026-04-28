## 示例模板
| Patch 文件名                                | 目的                    | 类型   | 适用版本                    | 是否默认应用 | 备注                         |
| ---------------------------------------- | --------------------- | ---- | ----------------------- | ------ | -------------------------- |
| 0001-parameterize-connectivity-dir.patch | 参数化 connectivity 目录 | base | NavNuances commit 57c0a58 | 是      | 便于使用仓库外 Matterport 连接图 |

## 正文
| Patch 文件名                                      | 目的                         | 类型   | 适用版本                    | 是否默认应用 | 备注                                                |
| ---------------------------------------------- | -------------------------- | ---- | ----------------------- | ------ | ------------------------------------------------- |
| base/0001-parameterize-connectivity-dir.patch   | 为 evaluator 增加 connectivity 目录参数 | base | NavNuances commit 57c0a58 | 是      | 支持通过 `--connectivity_dir` 指定 Matterport connectivity json 所在目录 |
| base/0002-skip-standard-eval.patch              | 允许按需跳过 Standard split          | base | NavNuances commit 57c0a58 | 是      | 增加 `--skip-standard`，只在缺少标准 R2R val_unseen 时跳过该项       |

### 说明

- Patch 根目录是 `third_party/navnuances/`。
- 默认 patch 放在 `patches/navnuances/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method navnuances`
  - `bash scripts/setup/revert_patches.sh --method navnuances`
- `base/0001-parameterize-connectivity-dir.patch` 会让 `evaluation/eval.py` 支持显式传入 `--connectivity_dir`。
- 当没有传入 `--scans_dir` 时，该 patch 会优先从 connectivity 目录下的 `scans.txt` 读取 scan 列表；若不存在 `scans.txt`，则根据 `*_connectivity.json` 文件名推断 scans。
- `base/0002-skip-standard-eval.patch` 会让 `evaluation/eval.py` 支持 `--skip-standard`。当 annotation/submission 中没有 `R2R_val_unseen.json` 与 `submit_val_unseen.json` 时，可以只评 DC/LR/RR/VM/NU；当这两个文件存在时，不传该参数即可同时评 Standard。
- 这些 patch 不改变 NavNuances 五类指标计算逻辑，只改变 connectivity graph 的发现/加载路径和 evaluator 入口的任务选择。
- 如需验证 patch 栈，可在干净临时树里执行：

```bash
tmp=$(mktemp -d)
mkdir -p "$tmp/src"
git -C third_party/navnuances archive HEAD | tar -x -C "$tmp/src"
for patch in patches/navnuances/base/*.patch; do
  git -C "$tmp/src" apply --check "$PWD/$patch"
  git -C "$tmp/src" apply "$PWD/$patch"
done
rm -rf "$tmp"
```
