#!/usr/bin/env bash

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

container_is_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1")" == "true" ]]
}

remove_container() {
  local container_name="$1"

  if ! container_exists "${container_name}"; then
    return
  fi

  if container_is_running "${container_name}"; then
    docker stop "${container_name}" >/dev/null
  fi
  docker rm "${container_name}" >/dev/null
}

enter_container() {
  local container_name="$1"
  local workdir="$2"
  shift 2

  if ! container_is_running "${container_name}"; then
    echo "启动已有容器 ${container_name} ..."
    docker start "${container_name}" >/dev/null
  fi

  echo "进入容器 ${container_name} ..."
  docker exec -it \
    -w "${workdir}" \
    "$@" \
    "${container_name}" \
    bash
}
