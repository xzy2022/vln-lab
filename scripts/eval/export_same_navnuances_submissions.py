#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_OUTPUTS_DIR = REPO_ROOT / "experiment_outputs"
DEFAULT_SPLITS = ("DC", "LR", "RR", "VM", "NU")
STANDARD_SPLIT = "val_unseen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SAME R2R split results as NavNuances submit_*.json files."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--experiment-id",
        help="SAME experiment id under experiment_outputs/<experiment-id>/.",
    )
    source_group.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing R2R_DC_results.json, R2R_LR_results.json, ...",
    )
    parser.add_argument(
        "--experiment-outputs-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUTS_DIR,
        help="Root directory used with --experiment-id.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional root directory for submissions when --output-dir is not set. "
            "With --experiment-id, the default is "
            "experiment_outputs/<experiment-id>/navnuances_submission."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Exact output directory for submit_*.json files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="R2R",
        help="Dataset prefix in SAME result filenames.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=(
            "Split suffixes to export. Default exports DC/LR/RR/VM/NU and "
            "auto-adds val_unseen when the result file exists."
        ),
    )
    parser.add_argument(
        "--standard",
        choices=("auto", "include", "skip"),
        default="auto",
        help="How to handle standard R2R_val_unseen when --splits is not set.",
    )
    parser.add_argument(
        "--keep-extra-fields",
        action="store_true",
        help="Keep SAME per-item metric fields. Default writes only instr_id and trajectory.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print JSON with this indent. Default writes compact JSON.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def normalize_split(split: str) -> str:
    split = split.strip()
    if split.lower().startswith("r2r_"):
        split = split[4:]
    if split.lower().endswith(".json"):
        split = split[:-5]
    if split.lower() in {"val_unseen", "standard"}:
        return STANDARD_SPLIT
    return split.upper()


def result_filename(dataset_name: str, split: str) -> str:
    return f"{dataset_name}_{split}_results.json"


def submission_filename(split: str) -> str:
    return f"submit_{split}.json"


def resolve_export_splits(
    results_dir: Path,
    dataset_name: str,
    requested_splits: list[str] | tuple[str, ...] | None,
    standard: str = "auto",
) -> list[str]:
    if requested_splits is not None:
        return [normalize_split(split) for split in requested_splits]

    splits = [normalize_split(split) for split in DEFAULT_SPLITS]
    standard_result = results_dir / result_filename(dataset_name, STANDARD_SPLIT)
    if standard == "include":
        splits.append(STANDARD_SPLIT)
    elif standard == "auto" and standard_result.exists():
        splits.append(STANDARD_SPLIT)
    return splits


def normalize_instr_id(value: Any) -> str:
    instr_id = str(value)
    if instr_id.startswith("r2r_"):
        instr_id = "_".join(instr_id.split("_")[1:])
    return instr_id


def normalize_trajectory(trajectory: Any) -> list[Any]:
    if not isinstance(trajectory, list):
        raise ValueError("trajectory must be a list")

    normalized = []
    for step in trajectory:
        if isinstance(step, str):
            normalized.append([step, 0, 0])
        elif isinstance(step, list):
            normalized.append(step)
        else:
            raise ValueError(f"unsupported trajectory step: {step!r}")
    return normalized


def normalize_result_item(item: dict[str, Any], keep_extra_fields: bool) -> dict[str, Any]:
    if "instr_id" not in item:
        raise ValueError(f"result item is missing instr_id: {item!r}")
    if "trajectory" not in item:
        raise ValueError(f"result item is missing trajectory: {item!r}")

    if keep_extra_fields:
        new_item = dict(item)
        new_item["instr_id"] = normalize_instr_id(new_item["instr_id"])
        new_item["trajectory"] = normalize_trajectory(new_item["trajectory"])
        return new_item

    return {
        "instr_id": normalize_instr_id(item["instr_id"]),
        "trajectory": normalize_trajectory(item["trajectory"]),
    }


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.experiment_id:
        results_dir = args.experiment_outputs_dir / args.experiment_id / "results"
        if args.output_dir:
            output_dir = args.output_dir
        elif args.output_root:
            output_dir = args.output_root / args.experiment_id
        else:
            output_dir = args.experiment_outputs_dir / args.experiment_id / "navnuances_submission"
    else:
        results_dir = args.results_dir
        if args.output_dir:
            output_dir = args.output_dir
        elif args.output_root:
            experiment_id = results_dir.parent.name if results_dir.name == "results" else results_dir.name
            output_dir = args.output_root / experiment_id
        elif results_dir.name == "results":
            output_dir = results_dir.parent / "navnuances_submission"
        else:
            output_dir = results_dir / "navnuances_submission"

    return results_dir.resolve(), output_dir.resolve()


def export_submissions(
    *,
    results_dir: Path,
    output_dir: Path,
    dataset_name: str = "R2R",
    splits: list[str] | tuple[str, ...] | None = None,
    standard: str = "auto",
    keep_extra_fields: bool = False,
    indent: int | None = None,
) -> int:
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing SAME results directory: {repo_rel(results_dir)}")
    splits = resolve_export_splits(results_dir, dataset_name, splits, standard=standard)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_items = 0
    for split in splits:
        input_path = results_dir / result_filename(dataset_name, split)
        output_path = output_dir / submission_filename(split)
        if not input_path.exists():
            raise FileNotFoundError(f"Missing SAME result file: {repo_rel(input_path)}")

        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{repo_rel(input_path)} must contain a JSON list")

        submission = [
            normalize_result_item(item, keep_extra_fields=keep_extra_fields)
            for item in data
        ]
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(submission, handle, ensure_ascii=False, indent=indent)

        total_items += len(submission)
        print(f"Wrote {repo_rel(output_path)}: {len(submission)} trajectories", flush=True)

    print(f"Done: {total_items} trajectories -> {repo_rel(output_dir)}", flush=True)
    return total_items


def main() -> None:
    args = parse_args()
    results_dir, output_dir = infer_paths(args)
    export_submissions(
        results_dir=results_dir,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        splits=args.splits,
        standard=args.standard,
        keep_extra_fields=args.keep_extra_fields,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()
