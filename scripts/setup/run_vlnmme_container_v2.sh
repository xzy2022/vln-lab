#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${REPO_DIR}/scripts/utils"

# shellcheck source=scripts/utils/docker_container_lifecycle.sh
source "${UTILS_DIR}/docker_container_lifecycle.sh"
# shellcheck source=scripts/utils/docker_deeplearning_defaults.sh
source "${UTILS_DIR}/docker_deeplearning_defaults.sh"
# shellcheck source=scripts/utils/docker_mounts_from_paths_config.sh
source "${UTILS_DIR}/docker_mounts_from_paths_config.sh"
# shellcheck source=scripts/utils/docker_run_args.sh
source "${UTILS_DIR}/docker_run_args.sh"

CONFIG_DIR=""
PATHS_CONFIG=""
PARSER="${UTILS_DIR}/parse_paths_mounts.py"

IMAGE_NAME=""
CONTAINER_NAME=""
RECREATE=0
CONTAINER_WORKDIR=""
DOCKER_ARGS=()
DOCKER_EXEC_ARGS=()
DOCKER_VOLUME_ARGS=()
MOUNT_SUMMARY=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup/run_vlnmme_container_v2.sh --image IMAGE --name CONTAINER --config-dir DIR [--recreate]

Required:
  --image IMAGE       Docker image to run, e.g. vln-lab-vlnmme:cu128
  --name CONTAINER    Docker container name, e.g. vln-vlnmme-cu128
  --config-dir DIR    Config directory containing paths.yaml, e.g. configs/global/local or configs/global/lab

Options:
  --recreate          Stop and remove an existing container before creating it
  -h, --help          Show this help
EOF
}

resolve_config_dir() {
  if [[ "${CONFIG_DIR}" != /* ]]; then
    CONFIG_DIR="${REPO_DIR}/${CONFIG_DIR}"
  fi

  PATHS_CONFIG="${CONFIG_DIR}/paths.yaml"

  if [[ ! -f "${PATHS_CONFIG}" ]]; then
    echo "ERROR: 配置文件不存在: ${PATHS_CONFIG}" >&2
    exit 1
  fi
}

print_existing_container_notice() {
  echo "容器 ${CONTAINER_NAME} 已存在；Docker 不会更新已有容器的挂载。"
  echo "本次进入 shell 会通过 docker exec -e 应用运行时环境变量。"
  echo "若需要应用新的挂载路径，请运行:"
  echo "  bash scripts/setup/run_vlnmme_container_v2.sh --image ${IMAGE_NAME} --name ${CONTAINER_NAME} --config-dir ${CONFIG_DIR} --recreate"
}

print_container_summary() {
  echo "Image: ${IMAGE_NAME}"
  echo "Container: ${CONTAINER_NAME}"
  echo "Workdir: ${REPO_DIR} -> ${CONTAINER_WORKDIR}"
  echo "Config: ${PATHS_CONFIG}"
  echo "Volumes:"
  echo "  repo: ${REPO_DIR} -> ${CONTAINER_WORKDIR}:rw"
  for line in "${MOUNT_SUMMARY[@]}"; do
    echo "  ${line}"
  done
}

create_container() {
  echo "创建并启动容器 ${CONTAINER_NAME} ..."
  print_container_summary

  docker run -d \
    --name "${CONTAINER_NAME}" \
    "${DOCKER_ARGS[@]}" \
    "${DOCKER_VOLUME_ARGS[@]}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    bash -lc "sleep infinity"
}

parse_basic_container_args IMAGE_NAME CONTAINER_NAME RECREATE CONFIG_DIR usage "$@"
resolve_config_dir
append_docker_mounts_from_paths_config \
  "${PATHS_CONFIG}" \
  "${REPO_DIR}" \
  "${PARSER}" \
  CONTAINER_WORKDIR \
  DOCKER_VOLUME_ARGS \
  MOUNT_SUMMARY

build_deeplearning_docker_args DOCKER_ARGS
DOCKER_EXEC_ARGS=(
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
)

DOCKER_VOLUME_ARGS=(
  -v "${REPO_DIR}:${CONTAINER_WORKDIR}:rw"
  "${DOCKER_VOLUME_ARGS[@]}"
)

if [[ "${RECREATE}" == "1" ]]; then
  echo "按 --recreate 请求删除已有容器 ${CONTAINER_NAME} ..."
  remove_container "${CONTAINER_NAME}"
fi

if container_exists "${CONTAINER_NAME}"; then
  print_existing_container_notice
  enter_container "${CONTAINER_NAME}" "${CONTAINER_WORKDIR}" "${DOCKER_EXEC_ARGS[@]}"
else
  create_container
  enter_container "${CONTAINER_NAME}" "${CONTAINER_WORKDIR}" "${DOCKER_EXEC_ARGS[@]}"
fi
