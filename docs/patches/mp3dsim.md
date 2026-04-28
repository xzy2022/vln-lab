## 示例模板
| Patch 文件名                      | 目的                 | 类型           | 适用版本                | 是否默认应用 | 备注                      |
| ------------------------------ | ------------------ | ------------ | ------------------- | ------ | ----------------------- |
| 0001-opencv4-compat.patch      | 兼容 OpenCV 4 API    | base         | MP3DSim commit 589d091 | 是      | 保证当前 Docker 环境可编译 |

## 正文
| Patch 文件名                         | 目的                  | 类型   | 适用版本                 | 是否默认应用 | 备注                                           |
| --------------------------------- | ------------------- | ---- | -------------------- | ------ | -------------------------------------------- |
| base/0001-opencv4-compat.patch    | 兼容 OpenCV 3+/4 的常量命名 | base | MP3DSim commit 589d091 | 是      | 将旧宏 `CV_LOAD_IMAGE_ANYDEPTH`/`CV_L2` 替换为当前 OpenCV API |

### 说明

- Patch 根目录是 `third_party/Matterport3DSimulator/`。
- 默认 patch 放在 `patches/mp3dsim/base/` 下，可通过以下脚本统一管理：
  - `bash scripts/setup/apply_patches.sh --method mp3dsim`
  - `bash scripts/setup/revert_patches.sh --method mp3dsim`
- `base/0001-opencv4-compat.patch` 只处理 OpenCV API 兼容，不改变 MatterSim 的渲染、导航图或测试语义。
- 当前 `docker/mp3dsim/Dockerfile` 使用 OpenCV 4 环境；该 patch 用于避免旧版 OpenCV 宏在新环境中编译失败。
- 如需验证 patch 栈，可在干净临时树里执行：

```bash
tmp=$(mktemp -d)
mkdir -p "$tmp/src"
git -C third_party/Matterport3DSimulator archive HEAD | tar -x -C "$tmp/src"
for patch in patches/mp3dsim/base/*.patch; do
  git -C "$tmp/src" apply --check "$PWD/$patch"
  git -C "$tmp/src" apply "$PWD/$patch"
done
rm -rf "$tmp"
```
