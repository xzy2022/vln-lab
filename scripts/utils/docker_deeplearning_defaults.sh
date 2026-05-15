#!/usr/bin/env bash

build_deeplearning_docker_args() {
  local -n out_args="$1"

  local docker_network="${DOCKER_NETWORK:-host}"
  local docker_shm_size="${DOCKER_SHM_SIZE:-16g}"
  local nvidia_driver_capabilities="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

  out_args+=(
    --gpus all
    --network "${docker_network}"
    --ipc=host
    --restart unless-stopped
    --shm-size="${docker_shm_size}"
    --ulimit memlock=-1
    --ulimit stack=67108864
    -e "NVIDIA_DRIVER_CAPABILITIES=${nvidia_driver_capabilities}"
  )
}
