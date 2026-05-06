#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_OUTPUTS_DIR = REPO_ROOT / "experiment_outputs"
DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "data" / "navnuances" / "annotations" / "NavNuances"
DEFAULT_CONNECTIVITY_DIR = REPO_ROOT / "data" / "same" / "simulator" / "connectivity"
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "third_party" / "navnuances" / "evaluation" / "eval.py"
DEFAULT_SPLITS = ("DC", "LR", "RR", "VM", "NU")
STANDARD_SPLIT = "val_unseen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export VLN-MME R2R predictions and run the NavNuances evaluator."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--experiment-dir",
        type=Path,
        help="VLN-MME output directory under experiment_outputs/.",
    )
    source_group.add_argument(
        "--experiment-id",
        help="VLN-MME experiment id under experiment_outputs/<experiment-id>/.",
    )
    parser.add_argument(
        "--experiment-outputs-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUTS_DIR,
        help="Root directory used with --experiment-id.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help=(
            "Directory containing VLN-MME result files such as R2R.R2R_DC.json. "
            "Default: auto-detect below <experiment-dir>."
        ),
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        help="Output directory for submit_*.json files. Default: <experiment-dir>/navnuances_submission.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Evaluator output directory. Default: <experiment-dir>/navnuances_eval.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help="Directory containing original NavNuances R2R_*.json annotations.",
    )
    parser.add_argument(
        "--connectivity-dir",
        type=Path,
        default=DEFAULT_CONNECTIVITY_DIR,
        help="Matterport connectivity directory used by NavNuances evaluator.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=DEFAULT_EVAL_SCRIPT,
        help="Path to third_party/navnuances/evaluation/eval.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for NavNuances evaluator. Default uses the current Python.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split suffixes to export. Default: DC LR RR VM NU.",
    )
    parser.add_argument(
        "--standard",
        choices=("include", "skip"),
        default="skip",
        help="Whether to include standard R2R_val_unseen. Default skips it.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Only export submit_*.json files; do not run the evaluator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned paths and evaluator command without writing or executing.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def normalize_split(split: str) -> str:
    split = Path(split.strip()).name
    if split.lower().endswith(".json"):
        split = split[:-5]
    if split.lower().startswith("submit_"):
        split = split[7:]
    if split.lower().startswith("r2r_"):
        split = split[4:]
    if split.lower() in {"standard", STANDARD_SPLIT}:
        return STANDARD_SPLIT
    return split.upper()


def submission_filename(split: str) -> str:
    return f"submit_{normalize_split(split)}.json"


def result_candidates(split: str) -> list[str]:
    normalized = normalize_split(split)
    if normalized == STANDARD_SPLIT:
        return ["R2R.val_unseen.json", "R2R.R2R_val_unseen.json"]
    return [f"R2R.R2R_{normalized}.json", f"R2R.{normalized}.json"]


def load_json_list(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{repo_rel(path)} must contain a JSON list")
    return data


def normalize_vlnmme_instr_id(value: Any) -> str:
    instr_id = str(value)
    if not instr_id.startswith("r2r_"):
        return instr_id

    instr_id = instr_id[len("r2r_") :]
    path_id, sep, sample_idx = instr_id.rpartition("_")
    if sep and sample_idx.isdigit():
        return path_id
    return instr_id


def normalize_trajectory(trajectory: Any) -> list[Any]:
    if not isinstance(trajectory, list):
        raise ValueError("trajectory must be a list")

    normalized = []
    for step in trajectory:
        if isinstance(step, str):
            normalized.append([step])
        elif isinstance(step, list):
            normalized.append(step)
        else:
            raise ValueError(f"unsupported trajectory step: {step!r}")
    return normalized


def normalize_result_item(item: dict[str, Any]) -> dict[str, Any]:
    if "instr_id" not in item:
        raise ValueError(f"result item is missing instr_id: {item!r}")
    if "trajectory" not in item:
        raise ValueError(f"result item is missing trajectory: {item!r}")

    return {
        "instr_id": normalize_vlnmme_instr_id(item["instr_id"]),
        "trajectory": normalize_trajectory(item["trajectory"]),
    }


def resolve_experiment_dir(
    *,
    experiment_dir: Path | None,
    experiment_id: str | None,
    experiment_outputs_dir: Path,
) -> Path:
    if experiment_dir is not None:
        return experiment_dir.resolve()
    if experiment_id is None:
        raise ValueError("Either experiment_dir or experiment_id is required")
    return (experiment_outputs_dir / experiment_id).resolve()


def find_result_dir(experiment_dir: Path, result_dir: Path | None, splits: list[str]) -> Path:
    if result_dir is not None:
        return result_dir.resolve()

    candidates = [experiment_dir / "baseline_agent" / "internvl3_2b"]
    candidates.extend(path for path in sorted(experiment_dir.glob("*/*")) if path.is_dir())

    needed = result_candidates(splits[0])
    for candidate in candidates:
        if any((candidate / filename).exists() for filename in needed):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not auto-detect VLN-MME result dir below "
        f"{repo_rel(experiment_dir)}. Pass --result-dir explicitly."
    )


def resolve_result_file(result_dir: Path, split: str) -> Path:
    for filename in result_candidates(split):
        path = result_dir / filename
        if path.exists():
            return path
    expected = ", ".join(result_candidates(split))
    raise FileNotFoundError(f"Missing VLN-MME result file for {split}: expected one of {expected}")


def export_submissions(
    *,
    result_dir: Path,
    submission_dir: Path,
    splits: list[str],
    dry_run: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not dry_run:
        submission_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        input_path = resolve_result_file(result_dir, split)
        output_path = submission_dir / submission_filename(split)
        data = load_json_list(input_path)
        submission = [normalize_result_item(item) for item in data]
        counts[normalize_split(split)] = len(submission)

        if dry_run:
            print(
                f"Would write {repo_rel(output_path)}: {len(submission)} trajectories "
                f"from {repo_rel(input_path)}",
                flush=True,
            )
            continue

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(submission, handle, ensure_ascii=False)
        print(
            f"Wrote {repo_rel(output_path)}: {len(submission)} trajectories "
            f"from {repo_rel(input_path)}",
            flush=True,
        )
    return counts


def build_evaluator_command(
    *,
    python_bin: str,
    eval_script: Path,
    annotation_root: Path,
    submission_dir: Path,
    out_dir: Path,
    connectivity_dir: Path,
    include_standard: bool,
) -> list[str]:
    command = [
        python_bin,
        str(eval_script),
        "--annotation_root",
        str(annotation_root),
        "--submission_root",
        str(submission_dir),
        "--out_root",
        str(out_dir),
        "--connectivity_dir",
        str(connectivity_dir),
    ]
    if not include_standard:
        command.append("--skip-standard")
    return command


def main() -> None:
    args = parse_args()
    experiment_dir = resolve_experiment_dir(
        experiment_dir=args.experiment_dir,
        experiment_id=args.experiment_id,
        experiment_outputs_dir=args.experiment_outputs_dir,
    )

    splits = [normalize_split(split) for split in args.splits]
    if args.standard == "include" and STANDARD_SPLIT not in splits:
        splits.append(STANDARD_SPLIT)
    elif args.standard == "skip":
        splits = [split for split in splits if split != STANDARD_SPLIT]

    result_dir = find_result_dir(experiment_dir, args.result_dir, splits)
    submission_dir = (args.submission_dir or experiment_dir / "navnuances_submission").resolve()
    out_dir = (args.out_dir or experiment_dir / "navnuances_eval").resolve()

    print(f"experiment_dir: {repo_rel(experiment_dir)}", flush=True)
    print(f"result_dir: {repo_rel(result_dir)}", flush=True)
    print(f"submission_dir: {repo_rel(submission_dir)}", flush=True)
    print(f"out_dir: {repo_rel(out_dir)}", flush=True)
    print(f"splits: {' '.join(splits)}", flush=True)

    export_submissions(
        result_dir=result_dir,
        submission_dir=submission_dir,
        splits=splits,
        dry_run=args.dry_run,
    )

    command = build_evaluator_command(
        python_bin=args.python,
        eval_script=args.eval_script.resolve(),
        annotation_root=args.annotation_root.resolve(),
        submission_dir=submission_dir,
        out_dir=out_dir,
        connectivity_dir=args.connectivity_dir.resolve(),
        include_standard=args.standard == "include",
    )
    print("Evaluator command:", shlex.join(command), flush=True)

    if args.no_eval or args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
