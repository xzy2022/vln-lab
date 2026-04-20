# SAME Docker 环境构建说明

本文档用于在当前 `vln-lab` 仓库内复现 `third_party/SAME` 的运行环境，目标机器为 RTX 5060，因此镜像固定使用 CUDA 12.8，并在容器内安装 PyTorch 2.7 的 `cu128` 轮子。

`SAME` 自身是 simulator-free 的，不需要额外安装 Matterport3D Simulator 或 Habitat。真正需要的是：

- `third_party/SAME` 子模块代码
- `third_party/SAME/requirements.txt` 中的 Python 依赖
- `download.py` 下载的数据和预训练权重

## 1. 构建镜像

先确认 `SAME` 子模块已经拉下来，然后从仓库根目录构建镜像。

```bash
VLN_LAB="/home/xzy/projects/vln-lab"

cd "$VLN_LAB"
git submodule update --init third_party/SAME
docker build -t vln-lab-same:cu128 -f docker/same/Dockerfile .
```

这里必须把仓库根目录 `.` 作为 build context，因为 Dockerfile 会直接复制 `third_party/SAME/requirements.txt`。如果在 `docker/same/` 目录下直接构建，Docker 无法拿到这个文件。

## 2. 启动或进入容器

项目里已经提供了针对当前仓库路径适配过的启动脚本。默认行为如下：

- 把当前仓库挂载到容器内的 `/workspace/vln-lab`
- 把宿主机的 Hugging Face 缓存挂载到 `/root/.cache/huggingface`
- 默认镜像名为 `vln-lab-same:cu128`
- 默认容器名为 `vln-same-cu128`

直接执行：

```bash
cd "$VLN_LAB"
bash scripts/setup/run_container.sh
```

如果你想自定义镜像名或容器名，可以在调用前覆盖环境变量：

```bash
IMAGE_NAME=vln-lab-same:cu128 CONTAINER_NAME=vln-same bash scripts/setup/run_container.sh
```

如果你不想使用脚本，等价的手工命令如下：

```bash
docker run -it \
  --name vln-same-cu128 \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/xzy/projects/vln-lab:/workspace/vln-lab \
  -v /home/xzy/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/vln-lab \
  vln-lab-same:cu128 \
  bash
```

## 3. 容器内检查环境

进入容器后，建议先回到仓库根目录再激活你实际使用的环境。下面示例使用 `test-v1`：

```bash
cd /workspace/vln-lab
conda activate test-v1
pip install -r envs/same/requirements-cu128-py310.txt
```

建议先确认 PyTorch 和 CUDA 版本符合预期：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

正常情况下应看到：

- PyTorch 主版本为 `2.7.0`
- CUDA 版本为 `12.8`
- `torch.cuda.is_available()` 返回 `True`

## 4. 下载 SAME 数据和预训练权重

这里建议使用仓库自带的通用下载脚本，把 SAME 相关资源统一放到项目根目录的 `data/same/` 下。因为整个仓库都挂载进了容器，所以下载结果会直接保留在宿主机上。

```bash
python scripts/setup/download_same_assets.py \
  --datasets R2R REVERIE CVDN SOON \
  --levels val test \
  --runtime mattersim \
  --models eval \
  --source hf-mirror
```

常见用法：

```bash
# 只下载 4 个任务的 val/test 标注
python scripts/setup/download_same_assets.py \
  --datasets R2R REVERIE CVDN SOON \
  --levels val test \
  --source hf-mirror

# 下载全部 SAME 标注（train / val / test / aug）
python scripts/setup/download_same_assets.py \
  --datasets all \
  --levels all \
  --source hf-mirror

# 下载全部标注 + 全部运行时资源 + 全部预训练/检查点
python scripts/setup/download_same_assets.py \
  --datasets all \
  --levels all \
  --runtime full \
  --models all \
  --source hf-mirror
```

## 5. 运行 SAME

如果你想让实验结果自动归档到 `experiment_outputs/` 并更新长期报表，推荐直接使用父项目入口：

```bash
cd /workspace/vln-lab
conda activate test-v1
# 只在 R2R 验证集评估
python scripts/experiments/run_same.py --config configs/same/val_r2r_eval_only.yaml

# 在4个数据集的验证集评估
python scripts/experiments/run_same.py --config configs/same/val_r2r_reverie_cvdn_soon.yaml
```

更完整的说明见 [docs/same-experiment-workflow.md](same-experiment-workflow.md)。

单卡先从最基本的配置开始：
```bash
cd /workspace/vln-lab/third_party/SAME/src
python run.py --config_dir configs/main_multi_q.yaml
```

如果要走分布式训练，再改用 `torchrun`：

```bash
cd /workspace/vln-lab/third_party/SAME/src
torchrun --nproc_per_node=1 --master_port=29500 run.py --config_dir configs/main_multi_q.yaml
```

训练输出默认会写到：

```text
third_party/SAME/src/output/
```

## 6. 与原始手工安装流程的对应关系

你原本准备执行的是：

```bash
conda create -n vln-same python=3.10 -y
conda activate vln-same
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

现在 Dockerfile 已经把这套流程固化成镜像构建步骤，只是做了两点更适合当前仓库的适配：

- `requirements.txt` 改为直接使用 `third_party/SAME/requirements.txt`
- `torchvision` 和 `torchaudio` 显式固定到和 `torch 2.7.0` 对应的版本，减少后续重新构建时的解析漂移
