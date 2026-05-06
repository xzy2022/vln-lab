#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-vln-lab-vlnmme:cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-vln-vlnmme-cu128}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/vln-lab}"
RECREATE_CONTAINER="${RECREATE_CONTAINER:-0}"

# Runtime choices. Override these from the host shell when needed, e.g.
# HF_ENDPOINT=https://huggingface.co scripts/setup/run_vlnmme_container.sh
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# data/ is a symlink to a large disk; resolve it so Docker mounts the real path.
DATA_HOST_DIR="$(realpath "${REPO_DIR}/data")"
DATA_CONTAINER_DIR="${DATA_CONTAINER_DIR:-${CONTAINER_WORKDIR}/data}"

HF_CACHE_DIR="${HF_CACHE_DIR:-${DATA_HOST_DIR}/.cache/huggingface/}"
HF_CONTAINER_CACHE_DIR="${HF_CONTAINER_CACHE_DIR:-${DATA_CONTAINER_DIR}/.cache/huggingface/}"

VLNMME_HOST_DATA_DIR="${VLNMME_HOST_DATA_DIR:-${DATA_HOST_DIR}/vlnmme}"
VLNMME_CONTAINER_DATA_DIR="${VLNMME_CONTAINER_DATA_DIR:-${DATA_CONTAINER_DIR}/vlnmme}"
VLNMME_UPSTREAM_DATA_DIR="${VLNMME_UPSTREAM_DATA_DIR:-${CONTAINER_WORKDIR}/third_party/VLN-MME/data}"
WHEELS_HOST_DIR="${WHEELS_HOST_DIR:-/home/xzy/data/wheels}"
WHEELS_CONTAINER_DIR="${WHEELS_CONTAINER_DIR:-/wheels}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

mkdir -p \
  "${HF_CACHE_DIR}" \
  "${VLNMME_HOST_DATA_DIR}" \
  "${WHEELS_HOST_DIR}" \
  "${REPO_DIR}/experiment_outputs"

DOCKER_ENV_ARGS=(
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
  -e "HF_ENDPOINT=${HF_ENDPOINT}"
  -e "HF_HOME=${HF_CONTAINER_CACHE_DIR}"
  -e "VLNMME_DATA_DIR=${VLNMME_CONTAINER_DATA_DIR}"
  -e "VLNMME_UPSTREAM_DATA_DIR=${VLNMME_UPSTREAM_DATA_DIR}"
)

for env_name in \
  HF_TOKEN \
  HUGGINGFACE_HUB_TOKEN \
  OPENAI_API_KEY \
  OPENAI_ORGANIZATION \
  OPENAI_PROJECT \
  OPENAI_BASE_URL \
  GOOGLE_API_KEY \
  GEMINI_API_KEY; do
  if [[ -n "${!env_name:-}" ]]; then
    DOCKER_ENV_ARGS+=(-e "${env_name}")
  fi
done

DOCKER_VOLUME_ARGS=(
  -v "${REPO_DIR}:${CONTAINER_WORKDIR}"
  -v "${DATA_HOST_DIR}:${DATA_CONTAINER_DIR}"
  -v "${HF_CACHE_DIR}:${HF_CONTAINER_CACHE_DIR}"
  -v "${VLNMME_HOST_DATA_DIR}:${VLNMME_CONTAINER_DATA_DIR}"
  -v "${VLNMME_HOST_DATA_DIR}:${VLNMME_UPSTREAM_DATA_DIR}"
  -v "${WHEELS_HOST_DIR}:${WHEELS_CONTAINER_DIR}"
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
    echo "本次进入 shell 会通过 docker exec -e 应用 HF_ENDPOINT 等环境变量。"
    echo "若需要应用新的挂载路径，请运行:"
    echo "  RECREATE_CONTAINER=1 $0"
  fi
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    echo "启动已有容器 ${CONTAINER_NAME} ..."
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "进入容器 ${CONTAINER_NAME} ..."
  docker exec -it \
    -w "${CONTAINER_WORKDIR}" \
    "${DOCKER_ENV_ARGS[@]}" \
    "${CONTAINER_NAME}" \
    bash
else
  echo "创建并启动容器 ${CONTAINER_NAME} ..."
  echo "HF endpoint: ${HF_ENDPOINT}"
  echo "HF cache: ${HF_CACHE_DIR} -> ${HF_CONTAINER_CACHE_DIR}"
  echo "VLN-MME data: ${VLNMME_HOST_DATA_DIR} -> ${VLNMME_CONTAINER_DATA_DIR}, ${VLNMME_UPSTREAM_DATA_DIR}"
  echo "Wheels: ${WHEELS_HOST_DIR} -> ${WHEELS_CONTAINER_DIR}"

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
