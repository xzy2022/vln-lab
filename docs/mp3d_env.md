## MP3D Simulator 环境说明


### 1. 创建容器
```bash
docker build -f docker/mp3dsim/Dockerfile -t vln-lab-mp3dsim:cu128 .
docker rm -f vln-mp3dsim-cu128
bash ./scripts/setup/run_mp3dsim_container.sh
```

如果数据集路径不存在,按照提示和自己的情况设置环境变量`export MP3D_DATA_DIR=$HOME/datasets/mp3d-mini/v1/scans`

在创建容器时,下面代码等价
```bash
# 直接使用脚本创建容器
bash ./scripts/setup/run_mp3dsim_container.sh

# 手动创建容器
docker run -it \
  --name vln-mp3dsim-cu128 \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /home/xzy/projects/vln-lab:/workspace/vln-lab \
  -v /home/xzy/datasets/mp3d-mini/v1/scans:/workspace/vln-lab/third_party/Matterport3DSimulator/data/v1/scans \
  -w /workspace/vln-lab \
  vln-lab-mp3dsim:cu128 \
  bash
```



本文档记录 `third_party/Matterport3DSimulator` 在当前仓库中的推荐构建与运行方式，以及常见报错的排查方法。

```bash
export MP3D_DATA_DIR=$HOME/datasets/mp3d-mini/v1/scans
bash ./scripts/setup/run_mp3dsim_container.sh
```

注意：

- `MP3D_DATA_DIR` 必须指向 `v1/scans` 目录。
- 该目录里的每个 scan 需要是可直接读取的目录结构，不能只有 `matterport_skybox_images.zip`。
- 如果只有 zip，还需要解压并生成 `<PANO_ID>_skybox_small.jpg`，见下文“数据准备要求”。

## 2. 编译 Matterport3DSimulator

```bash
cd /workspace/vln-lab
conda activate mp3d-sim

# 这一步可以在本地机进行,只需确保子项目是新版即可.
git submodule update --init --recursive third_party/Matterport3DSimulator

cd third_party/Matterport3DSimulator
rm -rf build
mkdir -p build
cd build

conda activate mp3d-sim

cmake \
  -DEGL_RENDERING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DOpenCV_DIR="$OpenCV_DIR" \
  -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
  ..

make -j"$(nproc)"
```


## 3. 数据准备

在本地机下载测试需要使用的全部20个指定`scans`并解压.
```bash
export MP_BASE_DIR=/home/xzy/datasets/mp3d-mini

# 循环下载
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

# 解压
python scripts/setup/prepare_mp3d_skybox.py \
  --scans-dir /workspace/vln-lab/third_party/Matterport3DSimulator/data/v1/scans \
  --from-rendertest-spec third_party/Matterport3DSimulator/src/test/rendertest_spec.json
```

## 4. 运行测试

进入容器后：

```bash
cd third_party/Matterport3DSimulator
./build/tests exclude:[Rendering]
env -u DISPLAY ./build/tests "RGB Image"
```
