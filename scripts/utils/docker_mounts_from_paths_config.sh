#!/usr/bin/env bash

load_mounts_from_paths_config() {
  local config="$1"
  local repo_root="$2"
  local parser="$3"
  local python_bin="${PYTHON_BIN:-python3}"

  "${python_bin}" "${parser}" \
    --config "${config}" \
    --repo-root "${repo_root}"
}

append_docker_mounts_from_paths_config() {
  local config="$1"
  local repo_root="$2"
  local parser="$3"
  local workdir_var="$4"
  local volume_args_var="$5"
  local summary_var="$6"

  local -n out_workdir="${workdir_var}"
  local -n out_volume_args="${volume_args_var}"
  local -n out_summary="${summary_var}"

  local parser_output
  local row_type field1 field2 field3 field4

  parser_output="$(
    load_mounts_from_paths_config "${config}" "${repo_root}" "${parser}"
  )"

  while IFS=$'\t' read -r row_type field1 field2 field3 field4; do
    case "${row_type}" in
      project_root)
        out_workdir="${field1}"
        ;;
      mount)
        local key="${field1}"
        local host_path="${field2}"
        local container_path="${field3}"
        local mode="${field4}"

        if [[ ! -e "${host_path}" ]]; then
          echo "ERROR: paths.mounts.${key}.host 不存在: ${host_path}" >&2
          echo "请修正 ${config}，确保该 host 路径在宿主机存在。" >&2
          return 1
        fi

        out_volume_args+=(-v "${host_path}:${container_path}:${mode}")
        out_summary+=("paths.mounts.${key}: ${host_path} -> ${container_path}:${mode}")
        ;;
      "")
        ;;
      *)
        echo "ERROR: 挂载解析器输出了未知行类型: ${row_type}" >&2
        return 1
        ;;
    esac
  done <<< "${parser_output}"

  if [[ -z "${out_workdir}" ]]; then
    echo "ERROR: ${config} 缺少 project_root。" >&2
    return 1
  fi
}
