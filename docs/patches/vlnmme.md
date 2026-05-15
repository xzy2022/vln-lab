## 示例模板
| Patch 文件名                                      | 目的                    | 类型   | 适用版本                 | 是否默认应用 | 备注                         |
| ---------------------------------------------- | --------------------- | ---- | -------------------- | ------ | -------------------------- |
| 0001-register-qwen-3b-and-qwen3-4b-models.patch | 注册额外 Qwen VL 模型入口 | base | VLN-MME commit 7d37bd6 | 是      | 支持 matrix 配置中的 3B/4B 模型 key |

## 正文
| Patch 文件名                                                | 目的                    | 类型   | 适用版本                 | 是否默认应用 | 备注 |
| ---------------------------------------------------------- | --------------------- | ---- | -------------------- | ------ | ---- |
| base/0001-register-qwen-3b-and-qwen3-4b-models.patch       | 注册 `qwen2_5_vl_3b` 和 `qwen3_vl_4b` 模型入口 | base | VLN-MME commit 7d37bd6 | 是      | 复用 Qwen2.5-VL 现有推理封装，并为 Qwen3-VL 4B 切换到 `AutoModelForImageTextToText` |

### 说明

- Patch 根目录是 `third_party/VLN-MME/`。
- 默认 patch 放在 `patches/vlnmme/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method vlnmme`
  - `bash scripts/setup/revert_patches.sh --method vlnmme`
- `base/0001-register-qwen-3b-and-qwen3-4b-models.patch` 会为 `src/models/__init__.py` 增加两个模型 key：
  - `qwen2_5_vl_3b`
  - `qwen3_vl_4b`
- 该 patch 还新增对应模型文件，使 `configs/vlnmme/matrix/**/r2r_qwen25vl_3b.yaml` 和 `configs/vlnmme/matrix/**/r2r_qwen3vl_4b.yaml` 能通过 VLN-MME 的 `get_models()` 加载。
- 该 patch 不改变 agent、环境、数据加载或评测指标逻辑，只补齐模型 registry 与 checkpoint 名称映射。
- 如需验证 patch 栈，可在干净临时树里执行：

```bash
tmp=$(mktemp -d)
mkdir -p "$tmp/src"
git -C third_party/VLN-MME archive HEAD | tar -x -C "$tmp/src"
for patch in patches/vlnmme/base/*.patch; do
  git -C "$tmp/src" apply --check "$PWD/$patch"
  git -C "$tmp/src" apply "$PWD/$patch"
done
rm -rf "$tmp"
```
