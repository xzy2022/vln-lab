#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VLNMME_SRC_DIR="${VLNMME_SRC_DIR:-${REPO_DIR}/third_party/VLN-MME/src}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/vlnmme/r2r_internvl3_2b_tiny.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_ITEMS="${SMOKE_ITEMS:-3}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

export TRANSFORMERS_VERBOSITY

if [[ ! -d "${VLNMME_SRC_DIR}" ]]; then
  echo "Missing VLN-MME src dir: ${VLNMME_SRC_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing VLN-MME smoke config: ${CONFIG_PATH}" >&2
  exit 1
fi

read_config_value() {
  local key_path="$1"
  "${PYTHON_BIN}" - "${CONFIG_PATH}" "${key_path}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

value = cfg
for key in sys.argv[2].split("."):
    value = value[key]

if isinstance(value, list):
    print("\n".join(str(item) for item in value))
else:
    print(value)
PY
}

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(read_config_value "experiment.output_dir")"
fi

DATA_DIR="$(read_config_value "experiment.data_dir")"
R2R_SPLITS="$(read_config_value "task.eval_splits.R2R")"

if grep -qx "val_unseen_subset_smoke" <<< "${R2R_SPLITS}"; then
  if [[ -z "${SMOKE_SOURCE_DATA:-}" ]]; then
    for candidate in \
      "${REPO_DIR}/data/vlnmme/R2R/val_unseen_subset_enc.json" \
      "${REPO_DIR}/third_party/VLN-MME/data/R2R/val_unseen_subset_enc.json"; do
      if [[ -f "${candidate}" ]]; then
        SMOKE_SOURCE_DATA="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${SMOKE_SOURCE_DATA:-}" || ! -f "${SMOKE_SOURCE_DATA}" ]]; then
    echo "Missing source data for tiny smoke split: ${SMOKE_SOURCE_DATA:-<searched default candidates>}" >&2
    exit 1
  fi

  mkdir -p "${DATA_DIR}/R2R"
  "${PYTHON_BIN}" - "${SMOKE_SOURCE_DATA}" "${DATA_DIR}/R2R/val_unseen_subset_smoke_enc.json" "${SMOKE_ITEMS}" <<'PY'
import sys
import json

source_path, target_path, count_arg = sys.argv[1:4]
count = int(count_arg)

with open(source_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(target_path, "w", encoding="utf-8") as f:
    json.dump(data[:count], f, indent=2)

print(f"Wrote tiny smoke split: {target_path} ({min(count, len(data))}/{len(data)} source items)")
PY
fi

mkdir -p "${OUTPUT_DIR}"

cd "${VLNMME_SRC_DIR}"
"${PYTHON_BIN}" main.py \
  --config_dir "${CONFIG_PATH}" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"
