#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
        description=(
            "Copy NavGPT-2 predictions to NavNuances submission format and run "
            "the NavNuances evaluator."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--experiment-dir",
        type=Path,
        help="NavGPT-2 output directory containing preds/submit_*.json.",
    )
    source_group.add_argument(
        "--experiment-id",
        help="NavGPT-2 experiment id under experiment_outputs/<experiment-id>/.",
    )
    parser.add_argument(
        "--experiment-outputs-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUTS_DIR,
        help="Root directory used with --experiment-id.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        help="Directory containing NavGPT-2 submit_*.json files. Default: <experiment-dir>/preds.",
    )
    parser.add_argument(
        "--pred-prefix",
        default="submit",
        help="Prediction file prefix. Use 'detail' for --detailed_output runs.",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        help="Output directory for NavNuances submit_*.json files.",
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
        default=None,
        help=(
            "Split suffixes to evaluate. Default: DC LR RR VM NU. The stock "
            "NavNuances evaluator requires all five skill splits."
        ),
    )
    parser.add_argument(
        "--standard",
        choices=("auto", "include", "skip"),
        default="skip",
        help=(
            "How to handle standard R2R_val_unseen. Default skips it for "
            "NavNuances-only Ability Atlas evaluation."
        ),
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip copying predictions and only run the evaluator on --submission-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths and evaluator command without copying or executing.",
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


def prediction_filename(prefix: str, split: str) -> str:
    return f"{prefix}_{normalize_split(split)}.json"


def submission_filename(split: str) -> str:
    return f"submit_{normalize_split(split)}.json"


def annotation_filename(split: str) -> str:
    return f"R2R_{normalize_split(split)}.json"


def load_json_list(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{repo_rel(path)} must contain a JSON list")
    return data


def validate_prediction_file(path: Path) -> int:
    data = load_json_list(path)
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{repo_rel(path)} item {index} must be a JSON object")
        if "instr_id" not in item:
            raise ValueError(f"{repo_rel(path)} item {index} is missing instr_id")
        if "trajectory" not in item:
            raise ValueError(f"{repo_rel(path)} item {index} is missing trajectory")
        if not isinstance(item["trajectory"], list):
            raise ValueError(f"{repo_rel(path)} item {index} trajectory must be a list")
    return len(data)


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


def resolve_eval_splits(
    *,
    pred_dir: Path,
    pred_prefix: str,
    annotation_root: Path,
    requested_splits: list[str] | tuple[str, ...] | None,
    standard: str,
) -> list[str]:
    if requested_splits is not None:
        raw_splits = [normalize_split(split) for split in requested_splits]
    else:
        raw_splits = list(DEFAULT_SPLITS)
        standard_pred = pred_dir / prediction_filename(pred_prefix, STANDARD_SPLIT)
        standard_annotation = annotation_root / annotation_filename(STANDARD_SPLIT)
        if standard == "include":
            raw_splits.append(STANDARD_SPLIT)
        elif standard == "auto" and standard_pred.exists() and standard_annotation.exists():
            raw_splits.append(STANDARD_SPLIT)

    unique: list[str] = []
    seen: set[str] = set()
    for split in raw_splits:
        if split not in seen:
            unique.append(split)
            seen.add(split)
    return unique


def export_submissions(
    *,
    pred_dir: Path,
    submission_dir: Path,
    splits: list[str] | tuple[str, ...],
    pred_prefix: str = "submit",
    dry_run: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not dry_run:
        submission_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        pred_path = pred_dir / prediction_filename(pred_prefix, split)
        output_path = submission_dir / submission_filename(split)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing NavGPT-2 prediction file: {repo_rel(pred_path)}")

        count = validate_prediction_file(pred_path)
        counts[split] = count
        if dry_run:
            print(
                f"Would copy {repo_rel(pred_path)} -> {repo_rel(output_path)} ({count} trajectories)",
                flush=True,
            )
            continue

        shutil.copy2(pred_path, output_path)
        print(
            f"Wrote {repo_rel(output_path)}: {count} trajectories from {repo_rel(pred_path)}",
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
        command.insert(2, "--skip-standard")
    return command


def validate_required_navnuances_splits(splits: list[str]) -> None:
    missing = [split for split in DEFAULT_SPLITS if split not in splits]
    if missing:
        raise ValueError(
            "The stock NavNuances evaluator requires all five skill splits. "
            f"Missing: {' '.join(missing)}"
        )


def main() -> int:
    args = parse_args()
    experiment_dir = resolve_experiment_dir(
        experiment_dir=args.experiment_dir,
        experiment_id=args.experiment_id,
        experiment_outputs_dir=args.experiment_outputs_dir,
    )
    pred_dir = (args.pred_dir or experiment_dir / "preds").resolve()
    submission_dir = (args.submission_dir or experiment_dir / "navnuances_submission").resolve()
    out_dir = (args.out_dir or experiment_dir / "navnuances_eval").resolve()
    annotation_root = args.annotation_root.resolve()
    connectivity_dir = args.connectivity_dir.resolve()
    eval_script = args.eval_script.resolve()

    splits = resolve_eval_splits(
        pred_dir=pred_dir,
        pred_prefix=args.pred_prefix,
        annotation_root=annotation_root,
        requested_splits=args.splits,
        standard=args.standard,
    )
    include_standard = STANDARD_SPLIT in splits
    validate_required_navnuances_splits(splits)

    if args.standard == "include" and STANDARD_SPLIT not in splits:
        raise ValueError("standard=include must include val_unseen")

    for split in splits:
        annotation_path = annotation_root / annotation_filename(split)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing NavNuances annotation: {repo_rel(annotation_path)}")

    command = build_evaluator_command(
        python_bin=args.python,
        eval_script=eval_script,
        annotation_root=annotation_root,
        submission_dir=submission_dir,
        out_dir=out_dir,
        connectivity_dir=connectivity_dir,
        include_standard=include_standard,
    )

    print(f"Experiment dir: {repo_rel(experiment_dir)}", flush=True)
    print(f"NavGPT-2 preds: {repo_rel(pred_dir)}", flush=True)
    print(f"Submission: {repo_rel(submission_dir)}", flush=True)
    print(f"NavNuances eval: {repo_rel(out_dir)}", flush=True)
    print(f"Splits: {' '.join(splits)}", flush=True)
    print(f"Evaluator command: {shlex.join(command)}", flush=True)

    if args.dry_run:
        return 0

    if not args.no_export:
        export_submissions(
            pred_dir=pred_dir,
            submission_dir=submission_dir,
            splits=splits,
            pred_prefix=args.pred_prefix,
        )
    elif not submission_dir.exists():
        raise FileNotFoundError(f"Missing submission directory: {repo_rel(submission_dir)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    result = subprocess.run(command, cwd=eval_script.parent, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
