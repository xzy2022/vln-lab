#!/usr/bin/env bash

container_exists() {
  local name="$1"

  docker container inspect "${name}" >/dev/null 2>&1
}

container_running() {
  local name="$1"

  [[ "$(docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || true)" == "true" ]]
}

remove_container() {
  local name="$1"

  if container_exists "${name}"; then
    docker rm -f "${name}" >/dev/null
  fi
}

enter_container() {
  local name="$1"
  local workdir="$2"
  shift 2

  if ! container_running "${name}"; then
    echo "启动已有容器 ${name} ..."
    docker start "${name}" >/dev/null
  fi

  echo "进入容器 ${name} ..."
  docker exec -it "$@" -w "${workdir}" "${name}" bash
}
