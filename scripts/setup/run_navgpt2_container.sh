#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

IMAGE_NAME="${IMAGE_NAME:-vln-lab-navgpt2:cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-vln-navgpt2-cu128}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/vln-lab}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
NAVGPT2_HOST_ROOT="${NAVGPT2_HOST_ROOT:-/data/E/navgpt-2}"
NAVGPT2_CONTAINER_ROOT="${NAVGPT2_CONTAINER_ROOT:-${CONTAINER_WORKDIR}/data/navgpt2}"
RECREATE_CONTAINER="${RECREATE_CONTAINER:-0}"

if [[ ! -d "${NAVGPT2_HOST_ROOT}" ]]; then
  echo "ERROR: NAVGPT2_HOST_ROOT 不存在: ${NAVGPT2_HOST_ROOT}" >&2
  echo "请设置为包含 datasets 和 map_nav_src 的目录，例如:" >&2
  echo "  export NAVGPT2_HOST_ROOT=/data/E/navgpt-2" >&2
  exit 1
fi

mkdir -p \
  "${HF_CACHE_DIR}" \
  "${REPO_DIR}/data/navgpt2"

DOCKER_ENV_ARGS=(
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
  -e "NAVGPT2_ASSET_ROOT=${NAVGPT2_CONTAINER_ROOT}"
  -e "NAVGPT2_DATASETS=${NAVGPT2_CONTAINER_ROOT}/datasets"
  -e "NAVGPT2_QFORMER_DIR=${NAVGPT2_CONTAINER_ROOT}/map_nav_src/models/lavis/output"
)

for env_name in HF_ENDPOINT HF_TOKEN HUGGINGFACE_HUB_TOKEN TRANSFORMERS_CACHE; do
  if [[ -n "${!env_name:-}" ]]; then
    DOCKER_ENV_ARGS+=(-e "${env_name}")
  fi
done

DOCKER_VOLUME_ARGS=(
  -v "${REPO_DIR}:${CONTAINER_WORKDIR}"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -v "${NAVGPT2_HOST_ROOT}:${NAVGPT2_CONTAINER_ROOT}"
)

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if [[ "${RECREATE_CONTAINER}" == "1" ]]; then
    echo "删除已有容器 ${CONTAINER_NAME} 以应用新的挂载 ..."
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
      docker stop "${CONTAINER_NAME}" >/dev/null
    fi
    docker rm "${CONTAINER_NAME}" >/dev/null
  else
    echo "容器 ${CONTAINER_NAME} 已存在；Docker 不会更新已有容器的挂载。"
    echo "若需要应用 data/navgpt2 新挂载，请运行:"
    echo "  RECREATE_CONTAINER=1 $0"
  fi
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
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_VOLUME_ARGS[@]}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    bash
fi
