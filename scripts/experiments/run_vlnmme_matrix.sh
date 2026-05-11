#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

GPUS="${VLNMME_BATCH_GPUS:-0,1,2,3,4,5,6,7}"
RAW_ITEMS_PER_RUN="${VLNMME_BATCH_RAW_ITEMS_PER_RUN:-0}"
LOG_ROOT="${VLNMME_MATRIX_LOG_ROOT:-${REPO_DIR}/experiment_outputs/vlnmme_matrix_logs}"
RUN_ID="${VLNMME_MATRIX_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
SUMMARY_PATH="${RUN_LOG_DIR}/summary.tsv"

DATASETS=(
  "val_unseen"
  "navnuances"
)

AGENTS=(
  "baseline_agent"
  "mapgpt_agent"
)

MODELS=(
  "r2r_qwen25vl_7b"
  "r2r_qwen25vl_3b"
  "r2r_qwen3vl_4b"
  "r2r_internvl3_2b"
)

has_arg() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${name}" || "${arg}" == "${name}="* ]]; then
      return 0
    fi
  done
  return 1
}

format_duration() {
  local seconds="$1"
  printf '%02d:%02d:%02d' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

task_log_path() {
  local dataset="$1"
  local agent="$2"
  local model="$3"
  printf '%s/%s/%s/%s.log' "${RUN_LOG_DIR}" "${dataset}" "${agent}" "${model}"
}

if has_arg "--config" "$@"; then
  echo "[matrix] error: --config is managed by this matrix script; edit DATASETS/AGENTS/MODELS in ${BASH_SOURCE[0]} to change the experiment list." >&2
  exit 2
fi

default_args=()
if ! has_arg "--gpus" "$@"; then
  default_args+=(--gpus "${GPUS}")
fi
if ! has_arg "--raw-items-per-run" "$@"; then
  default_args+=(--raw-items-per-run "${RAW_ITEMS_PER_RUN}")
fi

mkdir -p "${RUN_LOG_DIR}"
printf 'index\ttotal\tdataset\tagent\tmodel\tstatus\tstart_time\tend_time\telapsed_seconds\telapsed_hms\tconfig\tlog\n' > "${SUMMARY_PATH}"

total=$(( ${#DATASETS[@]} * ${#AGENTS[@]} * ${#MODELS[@]} ))
index=0
matrix_start_epoch="$(date +%s)"
matrix_start_iso="$(date -Is)"

echo "[matrix] run id: ${RUN_ID}"
echo "[matrix] logs: ${RUN_LOG_DIR}"
echo "[matrix] summary: ${SUMMARY_PATH}"
echo "[matrix] total tasks: ${total}"
echo "[matrix] default args: ${default_args[*]:-(none)}"

for dataset in "${DATASETS[@]}"; do
  for agent in "${AGENTS[@]}"; do
    for model in "${MODELS[@]}"; do
      index=$((index + 1))
      config_path="${REPO_DIR}/configs/vlnmme/matrix/${dataset}/${agent}/${model}.yaml"
      log_path="$(task_log_path "${dataset}" "${agent}" "${model}")"
      mkdir -p "$(dirname "${log_path}")"

      if [[ ! -f "${config_path}" ]]; then
        echo "[matrix] missing config (${index}/${total}): ${config_path}" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${index}" "${total}" "${dataset}" "${agent}" "${model}" "missing_config" \
          "" "" "0" "00:00:00" "${config_path}" "${log_path}" >> "${SUMMARY_PATH}"
        exit 2
      fi

      task_start_epoch="$(date +%s)"
      task_start_iso="$(date -Is)"
      echo
      echo "[matrix] (${index}/${total}) start dataset=${dataset} agent=${agent} model=${model}"
      echo "[matrix] config: ${config_path}"
      echo "[matrix] log: ${log_path}"

      status="ok"
      set +e
      {
        echo "[matrix] start_time=${task_start_iso}"
        echo "[matrix] index=${index}/${total}"
        echo "[matrix] dataset=${dataset}"
        echo "[matrix] agent=${agent}"
        echo "[matrix] model=${model}"
        echo "[matrix] config=${config_path}"
        echo "[matrix] args=${default_args[*]} $*"
        CONFIG_PATH="${config_path}" \
          bash "${SCRIPT_DIR}/run_vlnmme_resume.sh" \
            "${default_args[@]}" \
            "$@"
      } 2>&1 | tee "${log_path}"
      task_rc="${PIPESTATUS[0]}"
      set -e

      task_end_epoch="$(date +%s)"
      task_end_iso="$(date -Is)"
      elapsed=$((task_end_epoch - task_start_epoch))
      elapsed_hms="$(format_duration "${elapsed}")"

      if [[ "${task_rc}" -ne 0 ]]; then
        status="failed:${task_rc}"
      fi

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${index}" "${total}" "${dataset}" "${agent}" "${model}" "${status}" \
        "${task_start_iso}" "${task_end_iso}" "${elapsed}" "${elapsed_hms}" \
        "${config_path}" "${log_path}" >> "${SUMMARY_PATH}"

      echo "[matrix] (${index}/${total}) ${status} elapsed=${elapsed_hms} dataset=${dataset} agent=${agent} model=${model}"

      if [[ "${task_rc}" -ne 0 ]]; then
        echo "[matrix] stopped after failure; rerun the same command to resume remaining work." >&2
        exit "${task_rc}"
      fi
    done
  done
done

matrix_end_epoch="$(date +%s)"
matrix_end_iso="$(date -Is)"
matrix_elapsed=$((matrix_end_epoch - matrix_start_epoch))

echo
echo "[matrix] all tasks finished"
echo "[matrix] started: ${matrix_start_iso}"
echo "[matrix] ended: ${matrix_end_iso}"
echo "[matrix] total elapsed: $(format_duration "${matrix_elapsed}")"
echo "[matrix] summary: ${SUMMARY_PATH}"
