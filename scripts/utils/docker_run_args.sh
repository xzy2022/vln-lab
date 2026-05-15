#!/usr/bin/env bash

require_option_value() {
  local option="$1"
  local value="${2-}"

  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "ERROR: ${option} 需要一个非空参数。" >&2
    return 1
  fi
}

parse_basic_container_args() {
  local image_var="$1"
  local name_var="$2"
  local recreate_var="$3"
  local config_dir_var="$4"
  local usage_func="$5"
  shift 5

  local -n out_image="${image_var}"
  local -n out_name="${name_var}"
  local -n out_recreate="${recreate_var}"
  local -n out_config_dir="${config_dir_var}"

  out_image=""
  out_name=""
  out_recreate=0
  out_config_dir=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --image)
        if ! require_option_value "$1" "${2-}"; then
          "${usage_func}" >&2
          return 1
        fi
        out_image="$2"
        shift 2
        ;;
      --name)
        if ! require_option_value "$1" "${2-}"; then
          "${usage_func}" >&2
          return 1
        fi
        out_name="$2"
        shift 2
        ;;
      --recreate)
        out_recreate=1
        shift
        ;;
      --config-dir)
        if ! require_option_value "$1" "${2-}"; then
          "${usage_func}" >&2
          return 1
        fi
        out_config_dir="$2"
        shift 2
        ;;
      -h|--help)
        "${usage_func}"
        exit 0
        ;;
      *)
        echo "ERROR: 未知参数: $1" >&2
        "${usage_func}" >&2
        return 1
        ;;
    esac
  done

  if [[ -z "${out_image}" ]]; then
    echo "ERROR: 缺少必填参数 --image。" >&2
    "${usage_func}" >&2
    return 1
  fi

  if [[ -z "${out_name}" ]]; then
    echo "ERROR: 缺少必填参数 --name。" >&2
    "${usage_func}" >&2
    return 1
  fi

  if [[ -z "${out_config_dir}" ]]; then
    echo "ERROR: 缺少必填参数 --config-dir。" >&2
    "${usage_func}" >&2
    return 1
  fi
}
