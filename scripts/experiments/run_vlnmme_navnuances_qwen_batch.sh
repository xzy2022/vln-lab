#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[batch] run_vlnmme_navnuances_qwen_batch.sh is deprecated; forwarding to run_vlnmme_matrix.sh" >&2
exec bash "${SCRIPT_DIR}/run_vlnmme_matrix.sh" "$@"
