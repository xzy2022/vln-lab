## 正文
| Patch 文件名                                                | 目的                    | 类型   | 适用版本                 | 是否默认应用 | 备注 |
| ---------------------------------------------------------- | --------------------- | ---- | -------------------- | ------ | ---- |
| base/0001-register-qwen-model-wrappers.patch               | 注册额外 Qwen/Qwen-VL 模型入口 | base | VLN-MME commit 7d37bd6 | 是      | 复用 Qwen2.5-VL 现有推理封装，为 Qwen3-VL 切换到 `AutoModelForImageTextToText`，并补充 Qwen3.5 文本模型封装 |

### 说明

- Patch 根目录是 `third_party/VLN-MME/`。
- 默认 patch 放在 `patches/vlnmme/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method vlnmme`
  - `bash scripts/setup/revert_patches.sh --method vlnmme`
- `base/0001-register-qwen-model-wrappers.patch` 会为 `src/models/__init__.py` 增加这些模型 key：
  - `qwen2_5_vl_3b`
  - `qwen3_5_0_8b`
  - `qwen3_5_4b`
  - `qwen3_5_9b`
  - `qwen3_vl_4b`
  - `qwen3_vl_8b_instruct`
  - `qwen3_vl_8b_thinking`
- 该 patch 还新增对应模型文件，使 matrix 脚本中的 Qwen2.5-VL、Qwen3.5、Qwen3-VL 配置能通过 VLN-MME 的 `get_models()` 加载。
- `qwen3_5_0_8b`、`qwen3_5_4b` 和 `qwen3_5_9b` 是纯文本模型封装，在 VLN-MME agent 中会忽略图像观测，只使用 prompt 文本推理；Qwen3-VL 系列仍使用图像观测。
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
