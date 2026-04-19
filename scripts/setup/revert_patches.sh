#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PATCHES_DIR="${REPO_DIR}/patches"
THIRD_PARTY_DIR="${REPO_DIR}/third_party"

DRY_RUN=0
declare -a METHODS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/setup/revert_patches.sh [--method <name>]... [--dry-run]

Revert all patch files under patches/<name>/base/.

Options:
  --method <name>  Revert only the selected subproject. Can be repeated or use
                   a comma-separated list such as --method same,vln-duet.
  --dry-run        Print the actions without executing git apply --reverse.
  -h, --help       Show this help message.

If --method is omitted, the script reverts base patches for every subproject
that has patch files under patches/*/base/.
EOF
}

canonicalize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

append_methods() {
    local raw="$1"
    local token
    local -a tokens=()
    IFS=',' read -r -a tokens <<< "${raw}"
    for token in "${tokens[@]}"; do
        if [[ -n "${token}" ]]; then
            METHODS+=("${token}")
        fi
    done
}

collect_default_methods() {
    local method_dir
    local -a patches=()

    shopt -s nullglob
    for method_dir in "${PATCHES_DIR}"/*; do
        [[ -d "${method_dir}/base" ]] || continue
        patches=("${method_dir}/base"/*.patch)
        (( ${#patches[@]} > 0 )) || continue
        METHODS+=("$(basename "${method_dir}")")
    done
    shopt -u nullglob
}

resolve_target_dir() {
    local method="$1"
    local normalized_method
    local candidate
    local candidate_name

    normalized_method=$(canonicalize_name "${method}")
    for candidate in "${THIRD_PARTY_DIR}"/*; do
        [[ -d "${candidate}" ]] || continue
        candidate_name=$(basename "${candidate}")
        if [[ "$(canonicalize_name "${candidate_name}")" == "${normalized_method}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "无法为方法 ${method} 找到对应的 third_party 目录。" >&2
    return 1
}

revert_patch_file() {
    local target_dir="$1"
    local patch_file="$2"
    local rel_patch

    rel_patch=${patch_file#"${REPO_DIR}/"}

    if git -C "${target_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        if (( DRY_RUN )); then
            echo "[dry-run] revert ${rel_patch} -> ${target_dir}"
        else
            git -C "${target_dir}" apply --reverse "${patch_file}"
            echo "Reverted ${rel_patch} -> ${target_dir}"
        fi
        return 0
    fi

    if git -C "${target_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
        echo "Skipped ${rel_patch}: patch is not applied in ${target_dir}"
        return 0
    fi

    echo "无法回退 patch: ${rel_patch} (target: ${target_dir})" >&2
    return 1
}

while (($# > 0)); do
    case "$1" in
        --method)
            shift
            [[ $# -gt 0 ]] || { echo "--method 需要一个参数。" >&2; exit 1; }
            append_methods "$1"
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ ! -d "${PATCHES_DIR}" ]]; then
    echo "未找到 patches 目录: ${PATCHES_DIR}" >&2
    exit 1
fi

if [[ ! -d "${THIRD_PARTY_DIR}" ]]; then
    echo "未找到 third_party 目录: ${THIRD_PARTY_DIR}" >&2
    exit 1
fi

if (( ${#METHODS[@]} == 0 )); then
    collect_default_methods
fi

if (( ${#METHODS[@]} == 0 )); then
    echo "没有发现可回退的 base patches。" >&2
    exit 1
fi

for method in "${METHODS[@]}"; do
    patch_dir="${PATCHES_DIR}/${method}/base"
    if [[ ! -d "${patch_dir}" ]]; then
        echo "未找到 patch 目录: ${patch_dir}" >&2
        exit 1
    fi

    shopt -s nullglob
    patch_files=("${patch_dir}"/*.patch)
    shopt -u nullglob
    if (( ${#patch_files[@]} == 0 )); then
        echo "目录下没有 patch 文件: ${patch_dir}" >&2
        exit 1
    fi

    target_dir=$(resolve_target_dir "${method}")
    for patch_file in "${patch_files[@]}"; do
        revert_patch_file "${target_dir}" "${patch_file}"
    done
done
