#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DEFAULT_IMAGE_NAME="vln-lab-navnuances:cu128"
FALLBACK_IMAGE_NAME="vln-lab-navnuances-eval:cu128"

if [[ -z "${IMAGE_NAME:-}" ]]; then
  if docker image inspect "${DEFAULT_IMAGE_NAME}" >/dev/null 2>&1; then
    IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
  elif docker image inspect "${FALLBACK_IMAGE_NAME}" >/dev/null 2>&1; then
    IMAGE_NAME="${FALLBACK_IMAGE_NAME}"
  else
    IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
  fi
fi

CONTAINER_NAME="${CONTAINER_NAME:-vln-navnuances-cu128}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/vln-lab}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"

DEFAULT_MP3D_DATA_DIR="${HOME}/datasets/mp3d-mini/v1/scans"
NAVGPT4V_SCANS_TARGET="${NAVGPT4V_SCANS_TARGET:-${CONTAINER_WORKDIR}/third_party/navnuances/baselines/navgpt4v/data/v1/scans}"
MATTERSIM_SCANS_TARGET="${MATTERSIM_SCANS_TARGET:-${CONTAINER_WORKDIR}/third_party/Matterport3DSimulator/data/v1/scans}"

mkdir -p "${HF_CACHE_DIR}"

DOCKER_ENV_ARGS=(
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
)

for env_name in OPENAI_API_KEY OPENAI_ORGANIZATION OPENAI_BASE_URL HF_ENDPOINT; do
  if [[ -n "${!env_name:-}" ]]; then
    DOCKER_ENV_ARGS+=(-e "${env_name}")
  fi
done

DOCKER_VOLUME_ARGS=(
  -v "${REPO_DIR}:${CONTAINER_WORKDIR}"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
)

if [[ -n "${NAVNUANCES_DATA_DIR:-}" ]]; then
  if [[ ! -d "${NAVNUANCES_DATA_DIR}" ]]; then
    echo "ERROR: NAVNUANCES_DATA_DIR 不存在或不是目录: ${NAVNUANCES_DATA_DIR}"
    exit 1
  fi
  DOCKER_VOLUME_ARGS+=(-v "${NAVNUANCES_DATA_DIR}:${CONTAINER_WORKDIR}/data/navnuances")
fi

if [[ -z "${MP3D_DATA_DIR:-}" && -d "${DEFAULT_MP3D_DATA_DIR}" ]]; then
  MP3D_DATA_DIR="${DEFAULT_MP3D_DATA_DIR}"
  echo "未显式设置 MP3D_DATA_DIR，默认使用 ${MP3D_DATA_DIR}"
fi

if [[ -n "${MP3D_DATA_DIR:-}" ]]; then
  if [[ ! -d "${MP3D_DATA_DIR}" ]]; then
    echo "ERROR: MP3D_DATA_DIR 不存在或不是目录: ${MP3D_DATA_DIR}"
    echo "请设置为包含 scan 子目录的 v1/scans 目录。"
    exit 1
  fi
  DOCKER_VOLUME_ARGS+=(
    -v "${MP3D_DATA_DIR}:${NAVGPT4V_SCANS_TARGET}"
    -v "${MP3D_DATA_DIR}:${MATTERSIM_SCANS_TARGET}"
  )
else
  echo "未挂载 MP3D scans。只跑 NavNuances evaluator 时不需要；如需 NavGPT4v/渲染，请设置 MP3D_DATA_DIR。"
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
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_VOLUME_ARGS[@]}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    bash
fi
