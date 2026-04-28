#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from export_same_navnuances_submissions import (
    DEFAULT_SPLITS,
    STANDARD_SPLIT,
    export_submissions,
    repo_rel,
    resolve_export_splits,
    result_filename,
    submission_filename,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_OUTPUTS_DIR = REPO_ROOT / "experiment_outputs"
DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "data" / "navnuances" / "annotations" / "NavNuances"
DEFAULT_CONNECTIVITY_DIR = REPO_ROOT / "data" / "same" / "simulator" / "connectivity"
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "third_party" / "navnuances" / "evaluation" / "eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SAME NavNuances submissions and run the NavNuances evaluator "
            "for one experiment id."
        )
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="SAME experiment id under experiment_outputs/<experiment-id>/.",
    )
    parser.add_argument(
        "--experiment-outputs-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUTS_DIR,
        help="Root directory containing SAME experiment outputs.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help="Directory containing the original NavNuances R2R_*.json annotations.",
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
        "--submission-dir",
        type=Path,
        help="Output directory for submit_*.json. Default: experiment_outputs/<id>/navnuances_submission.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Evaluator output directory. Default: experiment_outputs/<id>/navnuances_eval.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="SAME results directory. Default: experiment_outputs/<id>/results.",
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
            "auto-adds val_unseen when both result and annotation files exist."
        ),
    )
    parser.add_argument(
        "--standard",
        choices=("auto", "include", "skip"),
        default="auto",
        help=(
            "How to handle standard R2R_val_unseen when --splits is not set. "
            "Use include to require it, skip to force NavNuances-only evaluation."
        ),
    )
    parser.add_argument(
        "--keep-extra-fields",
        action="store_true",
        help="Keep SAME per-item metric fields in submit_*.json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print submission JSON with this indent.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip submission export and only run the evaluator on --submission-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths and evaluator command without executing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = (args.experiment_outputs_dir / args.experiment_id).resolve()
    results_dir = (args.results_dir or (experiment_dir / "results")).resolve()
    submission_dir = (args.submission_dir or (experiment_dir / "navnuances_submission")).resolve()
    out_dir = (args.out_dir or (experiment_dir / "navnuances_eval")).resolve()
    annotation_root = args.annotation_root.resolve()
    connectivity_dir = args.connectivity_dir.resolve()
    eval_script = args.eval_script.resolve()
    standard_annotation = annotation_root / "R2R_val_unseen.json"
    standard_result = results_dir / result_filename(args.dataset_name, STANDARD_SPLIT)

    if args.splits is None:
        if args.standard == "include":
            if not standard_annotation.exists():
                raise FileNotFoundError(f"Missing standard annotation: {repo_rel(standard_annotation)}")
            if not standard_result.exists():
                raise FileNotFoundError(f"Missing standard SAME result: {repo_rel(standard_result)}")
            include_standard = True
        elif args.standard == "skip":
            include_standard = False
        else:
            include_standard = standard_annotation.exists() and standard_result.exists()
    else:
        resolved_requested = resolve_export_splits(results_dir, args.dataset_name, args.splits, standard="skip")
        include_standard = STANDARD_SPLIT in resolved_requested

    if include_standard and not standard_annotation.exists():
        raise FileNotFoundError(f"Missing standard annotation: {repo_rel(standard_annotation)}")

    command = [
        args.python,
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

    print(f"Experiment: {args.experiment_id}", flush=True)
    print(f"SAME results: {repo_rel(results_dir)}", flush=True)
    print(f"Submission: {repo_rel(submission_dir)}", flush=True)
    print(f"NavNuances eval: {repo_rel(out_dir)}", flush=True)
    print(f"Evaluator command: {shlex.join(command)}", flush=True)

    if args.dry_run:
        return 0

    if not args.no_export:
        export_submissions(
            results_dir=results_dir,
            output_dir=submission_dir,
            dataset_name=args.dataset_name,
            splits=args.splits,
            standard=("include" if include_standard else "skip"),
            keep_extra_fields=args.keep_extra_fields,
            indent=args.indent,
        )
        if include_standard and not (submission_dir / submission_filename(STANDARD_SPLIT)).exists():
            raise FileNotFoundError(
                f"Missing exported standard submission: {repo_rel(submission_dir / submission_filename(STANDARD_SPLIT))}"
            )
    elif not submission_dir.exists():
        raise FileNotFoundError(f"Missing submission directory: {repo_rel(submission_dir)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    result = subprocess.run(command, cwd=eval_script.parent, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
