#!/usr/bin/env python3
"""Run phase-4.6 endpoint preference ranker ablations."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import select_endpoint_frozen_config as frozen_select  # noqa: E402
import train_endpoint_preference_ranker as pref_ranker  # noqa: E402
import train_endpoint_ranker_baseline as baseline  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_preference_ablation.v1"
DEFAULT_OUTPUT_NAME = "preference_ablation"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_DEV_SPLIT = "dev"
DEFAULT_PRESET = "phase4_6"

ABLATION_COLUMNS = [
    "run_name",
    "objective",
    "feature_set",
    "pair_weights",
    "group_target",
    "group_loss_weight",
    "group_final_success_bonus",
    "selection_status",
    "selection_error",
    "gate_threshold",
    "tau",
    "allow_change_final",
    "delta_SR",
    "delta_SPL",
    "recovery_rate",
    "harm_rate",
    "net_recovery_rate",
    "changed_endpoint_rate",
    "gate_pass_rate",
    "gate_precision",
    "gate_recall",
    "max_dataset_harm_rate",
    "worst_harm_dataset",
    "CVDN_delta_SR",
    "CVDN_harm_rate",
    "success_gt_fail_acc",
    "better_spl_acc",
    "final_preserve_acc",
    "training_pairs",
    "group_targets",
    "passes_continue_line",
    "ranker_scores_csv",
    "ranker_report_md",
    "selection_report_md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small phase-4.6 endpoint preference objective ablation grid.",
    )
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--endpoint-learning-dir", default=None)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--episode-csv", default=None)
    parser.add_argument("--pair-csv", default=None)
    parser.add_argument("--gate-score-csv", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <experiment-dir>/endpoint_learning/preference_ablation.",
    )
    parser.add_argument("--target-scope", default=DEFAULT_TARGET_SCOPE)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--dev-split", default=DEFAULT_DEV_SPLIT)
    parser.add_argument(
        "--preset",
        choices=("phase4_6", "phase4_6_final_bias_sanity", "smoke"),
        default=DEFAULT_PRESET,
        help=(
            "Ablation preset. smoke keeps the grid tiny for wiring checks; "
            "phase4_6_final_bias_sanity runs the single final-bias decoupling sanity config."
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-pairs-per-type", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument(
        "--gate-thresholds",
        default="0,0.5,0.7,0.85,0.9,0.95",
        help="Gate thresholds passed to evaluator and frozen selection.",
    )
    parser.add_argument(
        "--taus",
        default="0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3",
        help="Tau grid passed to evaluator and frozen selection.",
    )
    parser.add_argument("--allow-change-final", default="true,false")
    parser.add_argument("--default-gate-score", type=float, default=1.0)
    parser.add_argument(
        "--selection-min-delta-sr",
        type=float,
        default=0.0,
        help="Frozen-selection eligibility floor. Keep 0 for ablation; continue-line is reported separately.",
    )
    parser.add_argument("--selection-min-delta-spl", type=float, default=0.0)
    parser.add_argument("--selection-max-harm-rate", type=float, default=0.01)
    parser.add_argument("--selection-max-dataset-harm-rate", type=float, default=0.01)
    parser.add_argument("--continue-min-delta-sr", type=float, default=0.002)
    parser.add_argument("--continue-min-delta-spl", type=float, default=0.0)
    parser.add_argument("--continue-max-harm-rate", type=float, default=0.01)
    parser.add_argument("--continue-max-dataset-harm-rate", type=float, default=0.01)
    parser.add_argument("--continue-min-cvdn-delta-sr", type=float, default=-0.001)
    parser.add_argument(
        "--skip-inner-eval",
        action="store_true",
        help="Skip train script's evaluator bridge; frozen selection still runs its own grid.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stderr progress logs. JSON manifest is still printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    verbose = not args.quiet
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    endpoint_learning_dir = baseline.resolve_endpoint_learning_dir(experiment_dir, args.endpoint_learning_dir)
    candidate_csv = resolve_path(args.candidate_csv, endpoint_learning_dir / "candidate_groups" / "endpoint_candidates.csv")
    episode_csv = resolve_path(args.episode_csv, endpoint_learning_dir / "candidate_groups" / "episode_groups.csv")
    pair_csv = resolve_path(args.pair_csv, endpoint_learning_dir / "preference_pairs" / "preference_pairs.csv")
    gate_score_csv = baseline.resolve_gate_score_csv(endpoint_learning_dir, args.gate_score_csv)
    output_dir = resolve_path(args.output_dir, endpoint_learning_dir / DEFAULT_OUTPUT_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_preset_configs(args.preset)
    log_progress(
        f"Starting endpoint preference ablation preset={args.preset}, configs={len(configs)}, output_dir={output_dir}",
        verbose=verbose,
        start_time=start_time,
    )
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        run_dir = output_dir / "runs" / config["run_name"]
        log_progress(
            f"[{index + 1}/{len(configs)}] Training {config['run_name']} "
            f"objective={config['objective']}, group={config.get('group_target')}, "
            f"features={config.get('feature_set', 'full')}",
            verbose=verbose,
            start_time=start_time,
        )
        ranker_manifest = pref_ranker.train_endpoint_preference_ranker(
            candidate_csv=candidate_csv,
            episode_csv=episode_csv,
            pair_csv=pair_csv,
            gate_score_csv=gate_score_csv,
            output_dir=run_dir,
            target_scope=args.target_scope,
            train_split=args.train_split,
            dev_split=args.dev_split,
            eval_split_filters=(args.dev_split,),
            objective=config["objective"],
            pair_weights=pref_ranker.parse_pair_weights(config["pair_weights"]),
            max_pairs_per_type=args.max_pairs_per_type,
            epochs=args.epochs if args.preset != "smoke" else min(args.epochs, 2),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2=args.l2,
            random_state=args.random_state + index,
            group_loss_weight=config["group_loss_weight"],
            group_scope=config.get("group_scope", "should_rerank"),
            group_target=config.get("group_target", "soft_reward"),
            group_temperature=config.get("group_temperature", 0.25),
            group_success_weight=config.get("group_success_weight", 1.0),
            group_spl_weight=config.get("group_spl_weight", 1.0),
            group_fail_reward=config.get("group_fail_reward", -1.0),
            group_late_penalty=config.get("group_late_penalty", 0.25),
            group_final_success_bonus=config.get("group_final_success_bonus", 1.0),
            feature_set=config.get("feature_set", "full"),
            score_transform=config.get("score_transform", "sigmoid"),
            gate_thresholds=tuple(baseline.parse_float_list(args.gate_thresholds)),
            taus=tuple(baseline.parse_float_list(args.taus)),
            allow_change_final_values=tuple(baseline.parse_bool_list(args.allow_change_final)),
            default_gate_score=args.default_gate_score,
            run_reranker_eval=not args.skip_inner_eval,
            include_test_summary=False,
            verbose=verbose,
        )
        rows.append(
            build_ablation_row(
                config=config,
                ranker_manifest=ranker_manifest,
                candidate_csv=candidate_csv,
                episode_csv=episode_csv,
                endpoint_learning_dir=endpoint_learning_dir,
                output_dir=run_dir,
                args=args,
                verbose=verbose,
                start_time=start_time,
            )
        )
        log_progress(
            f"[{index + 1}/{len(configs)}] Finished {config['run_name']}",
            verbose=verbose,
            start_time=start_time,
        )

    table = pd.DataFrame(rows, columns=ABLATION_COLUMNS)
    table_csv = output_dir / "preference_ablation_table.csv"
    report_md = output_dir / "preference_ablation_report.md"
    manifest_json = output_dir / "manifest.json"
    table.to_csv(table_csv, index=False)
    write_report(report_md, table, args)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": args.target_scope,
        "preset": args.preset,
        "configs": configs,
        "files": {
            "preference_ablation_table_csv": baseline.path_to_string(table_csv),
            "preference_ablation_report_md": baseline.path_to_string(report_md),
            "manifest_json": baseline.path_to_string(manifest_json),
        },
    }
    baseline.write_json(manifest_json, manifest)
    log_progress(
        f"Ablation done. table={table_csv}",
        verbose=verbose,
        start_time=start_time,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def build_ablation_row(
    config: dict[str, Any],
    ranker_manifest: dict[str, Any],
    candidate_csv: Path,
    episode_csv: Path,
    endpoint_learning_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    verbose: bool = True,
    start_time: float | None = None,
) -> dict[str, Any]:
    row = empty_row(config, ranker_manifest)
    score_csv = Path(ranker_manifest["files"]["preference_ranker_scores_csv"])
    ranker_model_json = Path(ranker_manifest["files"]["preference_ranker_model_json"])
    ranker_model_joblib = Path(ranker_manifest["files"]["preference_ranker_model_joblib"])
    selection_dir = output_dir / "frozen_selection"
    try:
        log_progress(
            f"Selecting frozen config for {config['run_name']} into {selection_dir}",
            verbose=verbose,
            start_time=start_time,
        )
        selection_manifest = frozen_select.select_endpoint_frozen_config(
            candidate_csv=candidate_csv,
            episode_csv=episode_csv,
            score_csv=score_csv,
            output_dir=selection_dir,
            gate_model_json=endpoint_learning_dir / "gate_baseline" / "gate_model.json",
            ranker_model_json=ranker_model_json,
            gate_model_joblib=endpoint_learning_dir / "gate_baseline" / "gate_model.joblib",
            ranker_model_joblib=ranker_model_joblib,
            target_scope=args.target_scope,
            selection_split=args.dev_split,
            selection_aggregation="weighted",
            diagnostic_splits=(args.train_split, args.dev_split),
            gate_thresholds=tuple(baseline.parse_float_list(args.gate_thresholds)),
            taus=tuple(baseline.parse_float_list(args.taus)),
            allow_change_final_values=tuple(baseline.parse_bool_list(args.allow_change_final)),
            min_delta_sr=args.selection_min_delta_sr,
            min_delta_spl=args.selection_min_delta_spl,
            max_harm_rate=args.selection_max_harm_rate,
            max_dataset_harm_rate=args.selection_max_dataset_harm_rate,
            allow_frozen_final_fallback=False,
        )
        row["selection_status"] = "selected"
        row["selection_report_md"] = selection_manifest["files"]["dev_selection_report_md"]
        selected_row = read_selected_row(selection_manifest["files"]["dev_selection_grid_csv"])
        for key in [
            "gate_threshold",
            "tau",
            "allow_change_final",
            "delta_SR",
            "delta_SPL",
            "recovery_rate",
            "harm_rate",
            "net_recovery_rate",
            "changed_endpoint_rate",
            "gate_pass_rate",
            "gate_precision",
            "gate_recall",
            "max_dataset_harm_rate",
            "worst_harm_dataset",
        ]:
            row[key] = selected_row.get(key)
        cvdn = read_dataset_row(
            selection_manifest["files"]["dev_selection_dataset_grid_csv"],
            selected_row=selected_row,
            dataset="CVDN",
        )
        row["CVDN_delta_SR"] = cvdn.get("delta_SR")
        row["CVDN_harm_rate"] = cvdn.get("harm_rate")
        row["passes_continue_line"] = passes_continue_line(row, args)
        log_progress(
            f"Selected {config['run_name']}: gate={row.get('gate_threshold')}, tau={row.get('tau')}, "
            f"dSR={format_pct(row.get('delta_SR'))}, harm={format_pct(row.get('harm_rate'))}, "
            f"continue={row.get('passes_continue_line')}",
            verbose=verbose,
            start_time=start_time,
        )
    except Exception as exc:  # noqa: BLE001 - ablation should continue after a failed config.
        row["selection_status"] = "failed"
        row["selection_error"] = f"{type(exc).__name__}: {exc}"
        log_progress(
            f"Selection failed for {config['run_name']}: {row['selection_error']}",
            verbose=verbose,
            start_time=start_time,
        )

    pair_metrics = read_pair_metrics(
        ranker_manifest["files"]["preference_ranker_pair_summary_csv"],
        split_filter=args.dev_split,
    )
    row["success_gt_fail_acc"] = pair_metrics.get("success_gt_fail")
    row["better_spl_acc"] = pair_metrics.get("better_spl_success_gt_lower_spl_success")
    row["final_preserve_acc"] = pair_metrics.get("final_success_final_gt_failed_nonfinal")
    return row


def empty_row(config: dict[str, Any], ranker_manifest: dict[str, Any]) -> dict[str, Any]:
    row = {column: None for column in ABLATION_COLUMNS}
    row.update(
        {
            "run_name": config["run_name"],
            "objective": config["objective"],
            "feature_set": config.get("feature_set", "full"),
            "pair_weights": config["pair_weights"],
            "group_target": config.get("group_target"),
            "group_loss_weight": config.get("group_loss_weight"),
            "group_final_success_bonus": config.get("group_final_success_bonus", 1.0),
            "selection_status": "not_run",
            "selection_error": "",
            "training_pairs": ranker_manifest["counts"]["training_pairs"],
            "group_targets": ranker_manifest["counts"]["group_targets"],
            "passes_continue_line": False,
            "ranker_scores_csv": ranker_manifest["files"]["preference_ranker_scores_csv"],
            "ranker_report_md": ranker_manifest["files"]["endpoint_preference_ranker_report_md"],
        }
    )
    return row


def build_preset_configs(preset: str) -> list[dict[str, Any]]:
    if preset == "smoke":
        return [
            {
                "run_name": "smoke_pairwise_spl2_final4",
                "objective": "pairwise",
                "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=2,final_success_final_gt_failed_nonfinal=4",
                "group_loss_weight": 0.0,
                "group_target": "soft_reward",
                "feature_set": "full",
            }
        ]
    if preset == "phase4_6_final_bias_sanity":
        return [
            {
                "run_name": "final_bias_decoupled_pairwise_listwise_best_spl",
                "objective": "pairwise_listwise",
                "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=2,final_success_final_gt_failed_nonfinal=0",
                "group_loss_weight": 1.0,
                "group_scope": "should_rerank",
                "group_target": "best_spl_onehot",
                "group_final_success_bonus": 0.0,
                "feature_set": "no_final_bias",
            }
        ]
    if preset != "phase4_6":
        raise ValueError(f"Unsupported preset={preset!r}")
    return [
        {
            "run_name": "pairwise_spl1_final2",
            "objective": "pairwise",
            "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=1,final_success_final_gt_failed_nonfinal=2",
            "group_loss_weight": 0.0,
            "group_target": "soft_reward",
            "feature_set": "full",
        },
        {
            "run_name": "pairwise_spl2_final4",
            "objective": "pairwise",
            "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=2,final_success_final_gt_failed_nonfinal=4",
            "group_loss_weight": 0.0,
            "group_target": "soft_reward",
            "feature_set": "full",
        },
        {
            "run_name": "pairwise_spl4_final8",
            "objective": "pairwise",
            "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=4,final_success_final_gt_failed_nonfinal=8",
            "group_loss_weight": 0.0,
            "group_target": "soft_reward",
            "feature_set": "full",
        },
        {
            "run_name": "pairwise_listwise_spl2_final4_soft",
            "objective": "pairwise_listwise",
            "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=2,final_success_final_gt_failed_nonfinal=4",
            "group_loss_weight": 1.0,
            "group_target": "soft_reward",
            "feature_set": "full",
        },
        {
            "run_name": "pairwise_listwise_spl2_final4_best_spl",
            "objective": "pairwise_listwise",
            "pair_weights": "success_gt_fail=1,better_spl_success_gt_lower_spl_success=2,final_success_final_gt_failed_nonfinal=4",
            "group_loss_weight": 1.0,
            "group_target": "best_spl_onehot",
            "feature_set": "full",
        },
    ]


def passes_continue_line(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        finite_ge(row.get("delta_SR"), args.continue_min_delta_sr)
        and finite_ge(row.get("delta_SPL"), args.continue_min_delta_spl)
        and finite_le(row.get("harm_rate"), args.continue_max_harm_rate)
        and finite_le(row.get("max_dataset_harm_rate"), args.continue_max_dataset_harm_rate)
        and finite_ge(row.get("CVDN_delta_SR"), args.continue_min_cvdn_delta_sr)
        and finite_gt(row.get("recovery_rate"), row.get("harm_rate"))
    )


def read_selected_row(path_value: str) -> dict[str, Any]:
    frame = pd.read_csv(resolve_manifest_path(path_value))
    if frame.empty or "selected" not in frame.columns:
        return {}
    selected = frame[frame["selected"].astype(str).str.lower().isin({"true", "1"})]
    if selected.empty:
        return {}
    return selected.iloc[0].to_dict()


def read_dataset_row(path_value: str, selected_row: dict[str, Any], dataset: str) -> dict[str, Any]:
    frame = pd.read_csv(resolve_manifest_path(path_value))
    if frame.empty:
        return {}
    frame["allow_change_final"] = frame["allow_change_final"].astype(str).str.lower()
    allow_value = str(selected_row.get("allow_change_final")).lower()
    mask = (
        (frame["dataset"] == dataset)
        & (abs(pd.to_numeric(frame["gate_threshold"], errors="coerce") - float(selected_row["gate_threshold"])) <= 1e-12)
        & (abs(pd.to_numeric(frame["tau"], errors="coerce") - float(selected_row["tau"])) <= 1e-12)
        & (frame["allow_change_final"] == allow_value)
    )
    rows = frame[mask]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def read_pair_metrics(path_value: str, split_filter: str) -> dict[str, float]:
    path = resolve_manifest_path(path_value)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    subset = frame[frame["split_filter"] == split_filter]
    metrics: dict[str, float] = {}
    for _, row in subset.iterrows():
        pair_type = str(row.get("pair_type"))
        if pair_type == "all":
            continue
        metrics[pair_type] = float(row.get("accuracy"))
    return metrics


def write_report(path: Path, table: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Endpoint Preference Ablation Report",
        "",
        "This report is generated by `scripts/analysis/run_endpoint_preference_ablation.py`.",
        "",
        "## Continue Line",
        "",
        f"- weighted delta_SR >= `{args.continue_min_delta_sr}`",
        f"- weighted delta_SPL >= `{args.continue_min_delta_spl}`",
        f"- harm_rate <= `{args.continue_max_harm_rate}`",
        f"- max_dataset_harm_rate <= `{args.continue_max_dataset_harm_rate}`",
        f"- CVDN delta_SR >= `{args.continue_min_cvdn_delta_sr}`",
        "- recovery_rate > harm_rate",
        "",
        "## Ablation Table",
        "",
        baseline.markdown_table(
            [
                "run",
                "status",
                "features",
                "gate",
                "tau",
                "dSR",
                "dSPL",
                "recovery",
                "harm",
                "max_harm",
                "CVDN_dSR",
                "better_SPL_acc",
                "continue",
            ],
            [
                [
                    row.get("run_name"),
                    row.get("selection_status"),
                    row.get("feature_set"),
                    baseline.fmt(row.get("gate_threshold")),
                    baseline.fmt(row.get("tau")),
                    baseline.pct(row.get("delta_SR")),
                    baseline.pct(row.get("delta_SPL")),
                    baseline.pct(row.get("recovery_rate")),
                    baseline.pct(row.get("harm_rate")),
                    baseline.pct(row.get("max_dataset_harm_rate")),
                    baseline.pct(row.get("CVDN_delta_SR")),
                    baseline.pct(row.get("better_spl_acc")),
                    str(row.get("passes_continue_line")).lower(),
                ]
                for _, row in table.iterrows()
            ],
        ),
        "",
        "## Files",
        "",
        "- preference_ablation_table.csv: `preference_ablation_table.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finite_ge(value: Any, threshold: float) -> bool:
    parsed = parse_float(value)
    return parsed is not None and parsed >= threshold - 1e-12


def finite_le(value: Any, threshold: float) -> bool:
    parsed = parse_float(value)
    return parsed is not None and parsed <= threshold + 1e-12


def finite_gt(left: Any, right: Any) -> bool:
    left_value = parse_float(left)
    right_value = parse_float(right)
    return left_value is not None and right_value is not None and left_value > right_value + 1e-12


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def log_progress(message: str, verbose: bool, start_time: float | None = None) -> None:
    if not verbose:
        return
    elapsed = f" +{format_duration(time.monotonic() - start_time)}" if start_time is not None else ""
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}{elapsed}] {message}", file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def format_pct(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "nan"
    return f"{parsed * 100:.2f}%"


def resolve_path(value: str | None, default: Path) -> Path:
    return Path(value).resolve() if value else default.resolve()


def resolve_manifest_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
