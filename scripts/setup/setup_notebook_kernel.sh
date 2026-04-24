#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-test-v1}"
DISPLAY_NAME="${2:-Python (${ENV_NAME})}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda init script not found: ${CONDA_SH}" >&2
    exit 1
fi

source "${CONDA_SH}"

if ! conda env list | awk 'NR > 2 {print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env '${ENV_NAME}' not found; skipping notebook kernel setup." >&2
    exit 0
fi

conda activate "${ENV_NAME}"

if ! python -m pip show ipykernel >/dev/null 2>&1; then
    python -m pip install ipykernel
fi

python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${DISPLAY_NAME}"

echo "Registered Jupyter kernel '${DISPLAY_NAME}' from conda env '${ENV_NAME}'."
