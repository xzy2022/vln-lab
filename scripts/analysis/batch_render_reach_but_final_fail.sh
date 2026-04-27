#!/usr/bin/env bash
set -euo pipefail

# Build reach-but-final-fail case tables and optionally render item replays.
#
# Default target:
#   experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5
#
# Common usage:
#   # Only export candidate CSVs and selected item ids.
#   RENDER=0 bash scripts/analysis/batch_render_reach_but_final_fail.sh
#
#   # Export and render the selected 20 cases. Missing scans are downloaded.
#   bash scripts/analysis/batch_render_reach_but_final_fail.sh
#
# Useful knobs:
#   COUNT=20
#   SPLIT=val_unseen
#   LOCAL_SCANS_ONLY=1        # only select cases whose scan exists locally
#   RENDER=0                  # skip rendering
#   FORCE_FINE_METRICS=1      # rebuild fine_metrics artifact
#   CONTAINER_NAME=vln-mp3dsim-cu128

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_DIR}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiment_outputs/0014_same_val_r2r_eval_only_same_s0_v5}"
EXPERIMENT_ID=$(basename "${EXPERIMENT_DIR}")
CONNECTIVITY_DIR="${CONNECTIVITY_DIR:-data/same/simulator/connectivity}"
FINE_OUTPUT_DIR="${FINE_OUTPUT_DIR:-reports/artifacts/fine_metrics/${EXPERIMENT_ID}}"
CASE_OUTPUT_DIR="${CASE_OUTPUT_DIR:-reports/artifacts/reach_but_final_fail/${EXPERIMENT_ID}}"
COUNT="${COUNT:-20}"
SPLIT="${SPLIT:-val_unseen}"
LOCAL_SCANS_ONLY="${LOCAL_SCANS_ONLY:-0}"
MP3D_SCAN_ROOT="${MP3D_SCAN_ROOT:-${HOME}/datasets/mp3d-mini/v1/scans}"
RENDER="${RENDER:-1}"
DOWNLOAD_MISSING_SCAN="${DOWNLOAD_MISSING_SCAN:-1}"
FORCE_FINE_METRICS="${FORCE_FINE_METRICS:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-vln-mp3dsim-cu128}"
CONDA_ENV="${CONDA_ENV:-mp3d-sim}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/vln-lab}"

if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
  echo "ERROR: experiment dir not found: ${EXPERIMENT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${EXPERIMENT_DIR}/eval_items" ]]; then
  echo "ERROR: missing eval_items dir: ${EXPERIMENT_DIR}/eval_items" >&2
  exit 1
fi

if [[ ! -d "${CONNECTIVITY_DIR}" ]]; then
  echo "ERROR: connectivity dir not found: ${CONNECTIVITY_DIR}" >&2
  exit 1
fi

FINE_WIDE="${FINE_OUTPUT_DIR}/tables/fine_metrics_wide.csv"
if [[ "${FORCE_FINE_METRICS}" == "1" || ! -f "${FINE_WIDE}" ]]; then
  echo "==> Building fine_metrics artifact: ${FINE_OUTPUT_DIR}"
  python scripts/analysis/build_same_fine_metrics.py \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --connectivity-dir "${CONNECTIVITY_DIR}" \
    --output-dir "${FINE_OUTPUT_DIR}"
else
  echo "==> Reusing fine_metrics artifact: ${FINE_OUTPUT_DIR}"
fi

mkdir -p "${CASE_OUTPUT_DIR}"

echo "==> Exporting reach-but-final-fail tables and selecting ${COUNT} cases"
export EXPERIMENT_DIR
export FINE_WIDE
export CASE_OUTPUT_DIR
export COUNT
export SPLIT
export LOCAL_SCANS_ONLY
export MP3D_SCAN_ROOT
python - <<'PY'
import csv
import json
import os
from pathlib import Path

experiment_dir = Path(os.environ["EXPERIMENT_DIR"])
experiment_id = experiment_dir.name
fine_wide = Path(os.environ["FINE_WIDE"])
case_output_dir = Path(os.environ["CASE_OUTPUT_DIR"])
count = int(os.environ["COUNT"])
split_filter = os.environ["SPLIT"]
local_scans_only = os.environ["LOCAL_SCANS_ONLY"] == "1"
mp3d_scan_root = Path(os.environ["MP3D_SCAN_ROOT"]).expanduser()

eval_by_internal = {}
saved_to_internal = {}
for path in sorted((experiment_dir / "eval_items").glob("*_eval_items.jsonl")):
    dataset_split = path.name.removesuffix("_eval_items.jsonl")
    dataset, split = dataset_split.split("_", 1)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        identity = row["identity"]
        internal = str(identity["internal_item_id"])
        saved = str(identity["saved_instr_id"])
        eval_by_internal[internal] = row
        saved_to_internal[(dataset, split, saved)] = internal

fields = [
    "source",
    "dataset",
    "split",
    "internal_item_id",
    "saved_instr_id",
    "scan",
    "instruction",
    "final_distance_m",
    "oracle_distance_m",
    "path_length_m",
    "shortest_path_length_m",
    "action_step_count",
    "move_step_count",
    "spl",
]

fine_rows = []
with fine_wide.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        final_success = row["official.final_success"].lower() == "true"
        oracle_success = row["official.oracle_success"].lower() == "true"
        if not oracle_success or final_success:
            continue
        item = eval_by_internal[row["internal_item_id"]]
        annotation = item["annotation"]
        fine_rows.append(
            {
                "source": "fine_metrics",
                "dataset": row["dataset"],
                "split": row["split"],
                "internal_item_id": row["internal_item_id"],
                "saved_instr_id": row["saved_instr_id"],
                "scan": annotation.get("scan", ""),
                "instruction": annotation.get("instruction", ""),
                "final_distance_m": row["official.final_distance_to_goal_m"],
                "oracle_distance_m": row["official.oracle_distance_to_goal_m"],
                "path_length_m": row["official.path_length_m"],
                "shortest_path_length_m": row["official.shortest_path_length_m"],
                "action_step_count": row["common.action_step_count"],
                "move_step_count": row["common.move_step_count"],
                "spl": row["official.spl"],
            }
        )

main_rows = []
for path in sorted((experiment_dir / "results").glob("*.json")):
    dataset_split = path.name.removesuffix("_results.json")
    dataset, split = dataset_split.split("_", 1)
    data = json.load(path.open(encoding="utf-8"))
    for result in data:
        if not bool(result.get("oracle_success")) or bool(result.get("success")):
            continue
        saved = str(result.get("instr_id"))
        internal = saved_to_internal.get((dataset, split, saved), "")
        item = eval_by_internal.get(internal, {})
        annotation = item.get("annotation", {})
        main_rows.append(
            {
                "source": "main_results",
                "dataset": dataset,
                "split": split,
                "internal_item_id": internal,
                "saved_instr_id": saved,
                "scan": annotation.get("scan", ""),
                "instruction": annotation.get("instruction", ""),
                "final_distance_m": result.get("nav_error", ""),
                "oracle_distance_m": result.get("oracle_error", ""),
                "path_length_m": result.get("trajectory_lengths", ""),
                "shortest_path_length_m": "",
                "action_step_count": result.get("action_steps", ""),
                "move_step_count": result.get("trajectory_steps", ""),
                "spl": result.get("spl", ""),
            }
        )

def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

write_csv(case_output_dir / "fine_metrics_reach_but_final_fail.csv", fine_rows, fields)
write_csv(case_output_dir / "main_results_reach_but_final_fail.csv", main_rows, fields)

selected_pool = [row for row in fine_rows if row["split"] == split_filter]
if local_scans_only:
    selected_pool = [
        row for row in selected_pool
        if (mp3d_scan_root / str(row["scan"])).is_dir()
    ]

selected = sorted(
    selected_pool,
    key=lambda row: (-float(row["final_distance_m"]), str(row["internal_item_id"])),
)[:count]

selected_fields = ["rank"] + fields + ["replay_dir"]
selected_name = f"selected_{count}_{split_filter}"
if local_scans_only:
    selected_name += "_local_scans"
selected_name += "_top_final_distance"

selected_csv = case_output_dir / f"{selected_name}.csv"
selected_txt = case_output_dir / f"{selected_name}_item_ids.txt"
with selected_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=selected_fields)
    writer.writeheader()
    for index, row in enumerate(selected, start=1):
        output = dict(row)
        output["rank"] = index
        output["replay_dir"] = str(experiment_dir / "item_replays" / str(row["internal_item_id"]))
        writer.writerow(output)
selected_txt.write_text(
    "\n".join(str(row["internal_item_id"]) for row in selected) + ("\n" if selected else ""),
    encoding="utf-8",
)

summary = {
    "experiment_id": experiment_id,
    "split": split_filter,
    "count_requested": count,
    "local_scans_only": local_scans_only,
    "fine_metrics_counts": {
        split: sum(1 for row in fine_rows if row["split"] == split)
        for split in sorted({row["split"] for row in fine_rows})
    },
    "main_results_counts": {
        split: sum(1 for row in main_rows if row["split"] == split)
        for split in sorted({row["split"] for row in main_rows})
    },
    "selected_count": len(selected),
    "selected_csv": str(selected_csv),
    "selected_item_ids": str(selected_txt),
}
(case_output_dir / "reach_but_final_fail_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

SELECTED_NAME="selected_${COUNT}_${SPLIT}"
if [[ "${LOCAL_SCANS_ONLY}" == "1" ]]; then
  SELECTED_NAME="${SELECTED_NAME}_local_scans"
fi
SELECTED_NAME="${SELECTED_NAME}_top_final_distance"
SELECTED_IDS="${CASE_OUTPUT_DIR}/${SELECTED_NAME}_item_ids.txt"
SELECTED_CSV="${CASE_OUTPUT_DIR}/${SELECTED_NAME}.csv"

echo "==> Selected CSV: ${SELECTED_CSV}"
echo "==> Selected ids: ${SELECTED_IDS}"

if [[ "${RENDER}" != "1" ]]; then
  echo "==> RENDER=${RENDER}; skip rendering."
  exit 0
fi

if [[ ! -s "${SELECTED_IDS}" ]]; then
  echo "ERROR: selected item id file is empty: ${SELECTED_IDS}" >&2
  exit 1
fi

DOWNLOAD_FLAG=""
if [[ "${DOWNLOAD_MISSING_SCAN}" == "1" ]]; then
  DOWNLOAD_FLAG="--download-missing-scan"
fi

render_selected_cases() {
  local selected_ids="$1"
  local experiment_dir="$2"
  local split="$3"
  local conda_env="$4"
  local download_flag="$5"

  SELECTED_IDS="${selected_ids}" \
  EXPERIMENT_DIR="${experiment_dir}" \
  SPLIT="${split}" \
  CONDA_ENV="${conda_env}" \
  DOWNLOAD_FLAG="${download_flag}" \
  bash -lc '
    set -euo pipefail
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
      source /opt/conda/etc/profile.d/conda.sh
      conda activate "${CONDA_ENV}"
    fi
    export PYTHONPATH=/workspace/vln-lab/third_party/Matterport3DSimulator/build:${PYTHONPATH:-}
    while read -r item_id; do
      [[ -n "${item_id}" ]] || continue
      echo "=== rendering ${item_id} ==="
      python scripts/analysis/render_same_eval_item.py "${item_id}" \
        --experiment-dir "${EXPERIMENT_DIR}" \
        --dataset R2R \
        --split "${SPLIT}" \
        ${DOWNLOAD_FLAG}
    done < "${SELECTED_IDS}"
  '
}

echo "==> Rendering selected cases into ${EXPERIMENT_DIR}/item_replays"
if command -v docker >/dev/null 2>&1; then
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "ERROR: Docker container not found: ${CONTAINER_NAME}" >&2
    echo "Run scripts/setup/run_mp3dsim_container.sh once to create it, then rerun this script." >&2
    exit 1
  fi

  echo "==> Starting container ${CONTAINER_NAME}"
  docker start "${CONTAINER_NAME}" >/dev/null
  docker exec -w "${CONTAINER_WORKDIR}" \
    -e SELECTED_IDS="${SELECTED_IDS}" \
    -e EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
    -e SPLIT="${SPLIT}" \
    -e CONDA_ENV="${CONDA_ENV}" \
    -e DOWNLOAD_FLAG="${DOWNLOAD_FLAG}" \
    "${CONTAINER_NAME}" \
    bash -lc "$(declare -f render_selected_cases); render_selected_cases \"\${SELECTED_IDS}\" \"\${EXPERIMENT_DIR}\" \"\${SPLIT}\" \"\${CONDA_ENV}\" \"\${DOWNLOAD_FLAG}\""
else
  echo "==> docker not found; assuming this script is already running inside the mp3dsim container."
  render_selected_cases "${SELECTED_IDS}" "${EXPERIMENT_DIR}" "${SPLIT}" "${CONDA_ENV}" "${DOWNLOAD_FLAG}"
fi

echo "==> Done."
