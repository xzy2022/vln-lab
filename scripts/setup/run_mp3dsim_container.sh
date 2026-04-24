#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

IMAGE_NAME="${IMAGE_NAME:-vln-lab-mp3dsim:cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-vln-mp3dsim-cu128}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/vln-lab}"
MOUNT_TARGET="${MOUNT_TARGET:-/workspace/vln-lab/third_party/Matterport3DSimulator/data/v1/scans}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
DEFAULT_MP3D_DATA_DIR="${HOME}/datasets/mp3d-mini/v1/scans"

if [[ -z "${MP3D_DATA_DIR:-}" ]]; then
  if [[ -d "${DEFAULT_MP3D_DATA_DIR}" ]]; then
    MP3D_DATA_DIR="${DEFAULT_MP3D_DATA_DIR}"
    echo "未显式设置 MP3D_DATA_DIR，默认使用 ${MP3D_DATA_DIR}"
  else
    echo "ERROR: 默认数据目录不存在: ${DEFAULT_MP3D_DATA_DIR}"
    echo "请设置 MP3D_DATA_DIR，指向包含 scan 子目录的 v1/scans 目录。"
    echo "例如：export MP3D_DATA_DIR=\$HOME/datasets/mp3d-mini/v1/scans"
    exit 1
  fi
fi

if [[ ! -d "${MP3D_DATA_DIR}" ]]; then
  echo "ERROR: MP3D_DATA_DIR 不存在或不是目录: ${MP3D_DATA_DIR}"
  echo "请设置为包含 scan 子目录的 v1/scans 目录。"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    echo "启动已有容器 ${CONTAINER_NAME} ..."
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "进入容器 ${CONTAINER_NAME} ..."
  docker exec -it -w "${CONTAINER_WORKDIR}" "${CONTAINER_NAME}" bash
else
  echo "创建并启动容器 ${CONTAINER_NAME} ..."
  docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --network host \
    --ipc=host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}" \
    -e LD_LIBRARY_PATH="/opt/conda/envs/mp3d-sim/lib:/usr/local/cuda/lib64" \
    -v "${REPO_DIR}:${CONTAINER_WORKDIR}" \
    -v "${MP3D_DATA_DIR}:${MOUNT_TARGET}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    bash
fi
