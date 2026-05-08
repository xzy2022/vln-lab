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
RECREATE_CONTAINER="${RECREATE_CONTAINER:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"

# data/ and experiment_outputs/ may be symlinks to a larger disk. Mount their
# resolved targets back over the repo paths so container path resolution works.
DATA_HOST_DIR="$(realpath -m "${DATA_HOST_DIR:-${REPO_DIR}/data}")"
EXPERIMENT_OUTPUTS_REPO_DIR="${REPO_DIR}/experiment_outputs"
if [[ -n "${EXPERIMENT_OUTPUTS_HOST_DIR:-}" ]]; then
  EXPERIMENT_OUTPUTS_HOST_DIR="$(realpath -m "${EXPERIMENT_OUTPUTS_HOST_DIR}")"
elif [[ -L "${EXPERIMENT_OUTPUTS_REPO_DIR}" ]]; then
  EXPERIMENT_OUTPUTS_HOST_DIR="$(realpath "${EXPERIMENT_OUTPUTS_REPO_DIR}")"
else
  EXPERIMENT_OUTPUTS_HOST_DIR="${EXPERIMENT_OUTPUTS_REPO_DIR}"
fi

DEFAULT_MP3D_DATA_DIR="${HOME}/datasets/mp3d-mini/v1/scans"
NAVGPT4V_SCANS_TARGET="${NAVGPT4V_SCANS_TARGET:-${CONTAINER_WORKDIR}/third_party/navnuances/baselines/navgpt4v/data/v1/scans}"
MATTERSIM_SCANS_TARGET="${MATTERSIM_SCANS_TARGET:-${CONTAINER_WORKDIR}/third_party/Matterport3DSimulator/data/v1/scans}"

mkdir -p "${HF_CACHE_DIR}" "${DATA_HOST_DIR}" "${EXPERIMENT_OUTPUTS_HOST_DIR}"

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
  -v "${DATA_HOST_DIR}:${CONTAINER_WORKDIR}/data"
  -v "${EXPERIMENT_OUTPUTS_HOST_DIR}:${CONTAINER_WORKDIR}/experiment_outputs"
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

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

container_is_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1")" == "true" ]]
}

print_container_summary() {
  echo "Repo: ${REPO_DIR} -> ${CONTAINER_WORKDIR}"
  echo "Data: ${DATA_HOST_DIR} -> ${CONTAINER_WORKDIR}/data"
  echo "Experiment outputs: ${EXPERIMENT_OUTPUTS_HOST_DIR} -> ${CONTAINER_WORKDIR}/experiment_outputs"
  echo "HF cache: ${HF_CACHE_DIR} -> /root/.cache/huggingface"
  if [[ -n "${NAVNUANCES_DATA_DIR:-}" ]]; then
    echo "NavNuances data override: ${NAVNUANCES_DATA_DIR} -> ${CONTAINER_WORKDIR}/data/navnuances"
  fi
  if [[ -n "${MP3D_DATA_DIR:-}" ]]; then
    echo "MP3D scans: ${MP3D_DATA_DIR} -> ${NAVGPT4V_SCANS_TARGET}, ${MATTERSIM_SCANS_TARGET}"
  fi
}

if container_exists "${CONTAINER_NAME}"; then
  case "${RECREATE_CONTAINER}" in
    1)
      echo "删除已有容器 ${CONTAINER_NAME} 以应用新的挂载 ..."
      if container_is_running "${CONTAINER_NAME}"; then
        docker stop "${CONTAINER_NAME}" >/dev/null
      fi
      docker rm "${CONTAINER_NAME}" >/dev/null
      ;;
    0)
      echo "容器 ${CONTAINER_NAME} 已存在；Docker 不会更新已有容器的挂载。"
      echo "本次进入 shell 会通过 docker exec -e 应用运行时环境变量。"
      echo "若需要应用新的挂载路径，请运行:"
      echo "  RECREATE_CONTAINER=1 $0"
      ;;
    *)
      echo "ERROR: RECREATE_CONTAINER 只能设置为 0 或 1。"
      exit 1
      ;;
  esac
fi

if container_exists "${CONTAINER_NAME}"; then
  if ! container_is_running "${CONTAINER_NAME}"; then
    echo "启动已有容器 ${CONTAINER_NAME} ..."
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "进入容器 ${CONTAINER_NAME} ..."
  docker exec -it -w "${CONTAINER_WORKDIR}" "${DOCKER_ENV_ARGS[@]}" "${CONTAINER_NAME}" bash
else
  echo "创建并启动容器 ${CONTAINER_NAME} ..."
  print_container_summary
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
