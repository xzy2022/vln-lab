#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

IMAGE_NAME="${IMAGE_NAME:-vln-lab-same:cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-vln-same-cu128}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/vln-lab}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

mkdir -p "${HF_CACHE_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]; then
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
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -v "${REPO_DIR}:${CONTAINER_WORKDIR}" \
      -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
      -w "${CONTAINER_WORKDIR}" \
      "${IMAGE_NAME}" \
      bash
fi
