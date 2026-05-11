#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-vln-lab-vlnmme:cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-vln-vlnmme-cu128}"
RECREATE_CONTAINER="${RECREATE_CONTAINER:-0}"

# 容器内路径属于镜像和仓库的固定约定，不从外部覆盖。
readonly CONTAINER_WORKDIR="/workspace/vln-lab"
readonly DATA_CONTAINER_DIR="${CONTAINER_WORKDIR}/data"
readonly HF_CONTAINER_CACHE_DIR="${DATA_CONTAINER_DIR}/.cache/huggingface/"
readonly VLNMME_CONTAINER_DATA_DIR="${DATA_CONTAINER_DIR}/vlnmme"
readonly VLNMME_UPSTREAM_DATA_DIR="${CONTAINER_WORKDIR}/third_party/VLN-MME/data"
readonly EXPERIMENT_OUTPUTS_CONTAINER_DIR="${CONTAINER_WORKDIR}/experiment_outputs"
readonly WHEELS_CONTAINER_DIR="/wheels"

# 运行时选项：默认使用固定的 Hugging Face 国内镜像；如需官方源可设置 USE_HF_MIRROR=0。
readonly HF_MIRROR_ENDPOINT="https://hf-mirror.com"
USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
case "${USE_HF_MIRROR}" in
  1)
    HF_ENDPOINT="${HF_MIRROR_ENDPOINT}"
    ;;
  0)
    HF_ENDPOINT=""
    ;;
  *)
    echo "ERROR: USE_HF_MIRROR 只能设置为 1 或 0。" >&2
    exit 1
    ;;
esac
# data/ 可能是指向大容量磁盘的软链接；也可通过 DATA_HOST_DIR 覆盖为其他机器的真实数据路径。
DATA_HOST_DIR="$(realpath -m "${DATA_HOST_DIR:-${REPO_DIR}/data}")"

DEFAULT_HF_CACHE_DIR="$(realpath -m "${DATA_HOST_DIR}/.cache/huggingface")"
HF_CACHE_DIR="$(realpath -m "${HF_CACHE_DIR:-${DEFAULT_HF_CACHE_DIR}}")"

VLNMME_HOST_DATA_DIR="${VLNMME_HOST_DATA_DIR:-${DATA_HOST_DIR}/vlnmme}"
EXPERIMENT_OUTPUTS_REPO_DIR="${REPO_DIR}/experiment_outputs"
if [[ -L "${EXPERIMENT_OUTPUTS_REPO_DIR}" ]]; then
  EXPERIMENT_OUTPUTS_HOST_DIR="$(realpath "${EXPERIMENT_OUTPUTS_REPO_DIR}")"
else
  EXPERIMENT_OUTPUTS_HOST_DIR="${EXPERIMENT_OUTPUTS_REPO_DIR}"
fi
WHEELS_HOST_DIR="${WHEELS_HOST_DIR:-${DATA_HOST_DIR}/wheels}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

mkdir -p \
  "${DATA_HOST_DIR}" \
  "${HF_CACHE_DIR}" \
  "${VLNMME_HOST_DATA_DIR}" \
  "${WHEELS_HOST_DIR}" \
  "${EXPERIMENT_OUTPUTS_HOST_DIR}"

# 传入容器的环境变量。
DOCKER_ENV_ARGS=(
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
  -e "HF_HOME=${HF_CONTAINER_CACHE_DIR}"
  -e "VLNMME_DATA_DIR=${VLNMME_CONTAINER_DATA_DIR}"
  -e "VLNMME_UPSTREAM_DATA_DIR=${VLNMME_UPSTREAM_DATA_DIR}"
)

if [[ -n "${HF_ENDPOINT}" ]]; then
  DOCKER_ENV_ARGS+=(-e "HF_ENDPOINT=${HF_ENDPOINT}")
fi

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

# 宿主机目录到容器目录的挂载关系。
DOCKER_VOLUME_ARGS=(
  -v "${REPO_DIR}:${CONTAINER_WORKDIR}"
  -v "${DATA_HOST_DIR}:${DATA_CONTAINER_DIR}"
  -v "${VLNMME_HOST_DATA_DIR}:${VLNMME_CONTAINER_DATA_DIR}"
  -v "${VLNMME_HOST_DATA_DIR}:${VLNMME_UPSTREAM_DATA_DIR}"
  -v "${EXPERIMENT_OUTPUTS_HOST_DIR}:${EXPERIMENT_OUTPUTS_CONTAINER_DIR}"
  -v "${WHEELS_HOST_DIR}:${WHEELS_CONTAINER_DIR}"
)

# 默认 HF cache 位于 data/ 下，已经随 DATA_HOST_DIR 挂载进容器。
# 只有用户显式把 HF_CACHE_DIR 指到其他位置时，才增加更具体的 HF cache 挂载。
if [[ "${HF_CACHE_DIR}" != "${DEFAULT_HF_CACHE_DIR}" ]]; then
  DOCKER_VOLUME_ARGS+=(-v "${HF_CACHE_DIR}:${HF_CONTAINER_CACHE_DIR}")
fi

# 以下函数封装容器复用、重建和创建逻辑。
container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

container_is_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1")" == "true" ]]
}

remove_existing_container_if_requested() {
  if ! container_exists "${CONTAINER_NAME}"; then
    return
  fi

  if [[ "${RECREATE_CONTAINER}" == "1" ]]; then
    echo "删除已有容器 ${CONTAINER_NAME} 以应用新的挂载 ..."
    if container_is_running "${CONTAINER_NAME}"; then
      docker stop "${CONTAINER_NAME}" >/dev/null
    fi
    docker rm "${CONTAINER_NAME}" >/dev/null
    return
  fi

  echo "容器 ${CONTAINER_NAME} 已存在；Docker 不会更新已有容器的挂载。"
  echo "本次进入 shell 会通过 docker exec -e 应用运行时环境变量。"
  echo "若需要应用新的挂载路径，请运行:"
  echo "  RECREATE_CONTAINER=1 $0"
}

enter_existing_container() {
  if ! container_is_running "${CONTAINER_NAME}"; then
    echo "启动已有容器 ${CONTAINER_NAME} ..."
    docker start "${CONTAINER_NAME}" >/dev/null
  fi

  echo "进入容器 ${CONTAINER_NAME} ..."
  docker exec -it \
    -w "${CONTAINER_WORKDIR}" \
    "${DOCKER_ENV_ARGS[@]}" \
    "${CONTAINER_NAME}" \
    bash
}

print_container_summary() {
  echo "HF endpoint: ${HF_ENDPOINT:-<default>}"
  if [[ "${HF_CACHE_DIR}" == "${DEFAULT_HF_CACHE_DIR}" ]]; then
    echo "HF cache: ${HF_CACHE_DIR} -> ${HF_CONTAINER_CACHE_DIR} (via data mount)"
  else
    echo "HF cache: ${HF_CACHE_DIR} -> ${HF_CONTAINER_CACHE_DIR} (custom mount)"
  fi
  echo "Data root: ${DATA_HOST_DIR} -> ${DATA_CONTAINER_DIR}"
  echo "VLN-MME data: ${VLNMME_HOST_DATA_DIR} -> ${VLNMME_CONTAINER_DATA_DIR}, ${VLNMME_UPSTREAM_DATA_DIR}"
  echo "Experiment outputs: ${EXPERIMENT_OUTPUTS_HOST_DIR} -> ${EXPERIMENT_OUTPUTS_CONTAINER_DIR}"
  echo "Wheels: ${WHEELS_HOST_DIR} -> ${WHEELS_CONTAINER_DIR}"
}

create_container() {
  echo "创建并启动容器 ${CONTAINER_NAME} ..."
  print_container_summary

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --network host \
    --ipc=host \
    --restart unless-stopped \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_VOLUME_ARGS[@]}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    bash -lc "sleep infinity"

  enter_existing_container
}

run_container() {
  remove_existing_container_if_requested

  if container_exists "${CONTAINER_NAME}"; then
    enter_existing_container
  else
    create_container
  fi
}

run_container
