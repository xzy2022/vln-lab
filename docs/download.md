根据目标样本 ID，例如 `r2r_5593_0`获取`scan`编号
```bash
python - <<'PY'
import json

eval_items = "experiment_outputs/0013_same_val_all_r2r_reverie_cvdn_soon_same_s0_v4/eval_items/R2R_val_seen_eval_items.jsonl"
target = "r2r_5593_0"

with open(eval_items, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        ident = item.get("identity", {})
        if ident.get("internal_item_id") == target or str(ident.get("saved_instr_id")) == target:
            print("scan =", item["annotation"]["scan"])
            print("trajectory_len =", len(item["prediction"]["trajectory"]))
            break
PY
```

下载对应`scan/`的数据.
```bash
export TARGET_SCAN=vyrNrziPKCB
python3 scripts/setup/download_mp.py \
  -o /home/xzy/datasets/mp3d-mini \
  --id "$TARGET_SCAN" \
  --type matterport_skybox_images
```

上面的下载脚本只会把 zip 下载到本地，不会自动解压。下载完成后需要继续准备：

```bash
export MP3D_DATA_DIR=/home/xzy/datasets/mp3d-mini/v1/scans

python scripts/setup/prepare_mp3d_skybox.py \
  --scans-dir "$MP3D_DATA_DIR" \
  --scan-id "$TARGET_SCAN"
```

说明：

- 这一步会先解压 `matterport_skybox_images.zip`
- 然后生成 MatterSim 运行和渲染测试需要的 `<PANO_ID>_skybox_small.jpg`

如果你要跑 `third_party/Matterport3DSimulator` 自带的 `RGB Image` 测试，单个 `TARGET_SCAN` 不够，还需要先下载 `src/test/rendertest_spec.json` 里列出的全部 20 个 scan。

`scripts/setup/download_mp.py` 当前只支持 `--id <单个_scan>` 或 `--id ALL`，不支持一次传多个 scan id，所以下载这 20 个测试 scan 时推荐直接循环现有脚本：

```bash
export MP_BASE_DIR=/home/xzy/datasets/mp3d-mini

python - <<'PY' | while read -r scan_id; do
import json
from pathlib import Path

root = json.loads(Path("third_party/Matterport3DSimulator/src/test/rendertest_spec.json").read_text())
seen = set()
for batch in root:
    for item in batch:
        scan_id = item["scanId"]
        if scan_id not in seen:
            seen.add(scan_id)
            print(scan_id)
PY
  python3 scripts/setup/download_mp.py \
    -o "$MP_BASE_DIR" \
    --id "$scan_id" \
    --type matterport_skybox_images \
    --assume_yes
done
```

下载完成后，再统一解压并生成 `*_skybox_small.jpg`：

```bash
export MP3D_DATA_DIR="$MP_BASE_DIR/v1/scans"

python scripts/setup/prepare_mp3d_skybox.py \
  --scans-dir "$MP3D_DATA_DIR" \
  --from-rendertest-spec third_party/Matterport3DSimulator/src/test/rendertest_spec.json
```
