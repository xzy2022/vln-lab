#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/vlnmme/r2r_internvl3_2b_tiny.yaml}"
VLNMME_SRC_DIR="${VLNMME_SRC_DIR:-${REPO_DIR}/third_party/VLN-MME/src}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
VLNMME_HF_OFFLINE="${VLNMME_HF_OFFLINE:-1}"

export CONFIG_PATH
export TRANSFORMERS_VERBOSITY
export VLNMME_SRC_DIR
export VLNMME_HF_OFFLINE

"${PYTHON_BIN}" "${REPO_DIR}/scripts/experiments/run_vlnmme_resume.py" \
  --config "${CONFIG_PATH}" \
  --vlnmme-src "${VLNMME_SRC_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  "$@"
