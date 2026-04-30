#!/usr/bin/env python3
"""Diagnose gate generalization for phase 4.8 without training new models."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_endpoint_ranker_top1 as ranker_diag  # noqa: E402
import eval_endpoint_reranker as reranker_eval  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_gate_generalization_diagnostics.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_SPLITS = ("dev", "val_unseen")
DEFAULT_GATE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
DEFAULT_TAUS = (0.01, 0.03)
DEFAULT_FEATURE_CONFIGS = "dev_best:0.90:0.01,safety:0.95:0.03"
DEFAULT_OUTPUT_SUBDIR = Path("endpoint_learning/gate_generalization_diagnostics/phase4_8_final_bias_sanity")
DEFAULT_RANKER_DIAG_SUBDIR = Path("endpoint_ranker_diagnostics/phase4_8_final_bias_sanity_dev_val_unseen")
EPS = 1e-12

BOOL_TRUE = {"true", "1", "yes", "y", "t"}
BOOL_FALSE = {"false", "0", "no", "n", "f"}

CALIBRATION_COLUMNS = [
    "split_filter",
    "dataset",
    "items",
    "positives",
    "base_rate",
    "mean_gate_score",
    "calibration_gap",
    "roc_auc",
    "average_precision",
    "brier",
    "ece_10bin",
    "mean_score_positive",
    "mean_score_negative",
    "val_minus_dev_base_rate",
    "val_minus_dev_mean_gate_score",
    "val_minus_dev_calibration_gap",
    "val_minus_dev_roc_auc",
    "val_minus_dev_average_precision",
    "val_minus_dev_brier",
    "val_minus_dev_ece_10bin",
]

CALIBRATION_BIN_COLUMNS = [
    "split_filter",
    "dataset",
    "bin_index",
    "bin_low",
    "bin_high",
    "items",
    "positives",
    "base_rate",
    "mean_gate_score",
    "calibration_gap",
]

THRESHOLD_COLUMNS = [
    "split_filter",
    "dataset",
    "tau",
    "gate_threshold",
    "items",
    "gate_pass_rate",
    "changed_rate",
    "recovery_rate",
    "harm_rate",
    "net_recovery_rate",
    "changed_safe_rate",
    "changed_unrecovered_rate",
    "recovered_items",
    "harmed_items",
    "changed_safe_items",
    "changed_unrecovered_items",
    "harm_per_recovered",
    "blocked_should_rerank_rate",
    "passes_safety_line",
    "val_minus_dev_recovery_rate",
    "val_minus_dev_harm_rate",
    "val_minus_dev_net_recovery_rate",
    "val_minus_dev_changed_rate",
]

FEATURE_SHIFT_COLUMNS = [
    "split_filter",
    "dataset",
    "config",
    "gate_threshold",
    "tau",
    "outcome_class",
    "feature",
    "items",
    "mean",
    "median",
    "p25",
    "p75",
    "std",
    "delta_vs_recovered",
    "val_minus_dev",
]

FEATURE_EFFECT_COLUMNS = [
    "split_filter",
    "dataset",
    "config",
    "gate_threshold",
    "tau",
    "feature",
    "recovered_items",
    "harmed_items",
    "recovered_mean",
    "harmed_mean",
    "harm_minus_recovered",
    "pooled_std",
    "standardized_effect",
    "val_minus_dev_standardized_effect",
]

RANKER_STABILITY_COLUMNS = [
    "split_filter",
    "dataset",
    "should_rerank_items",
    "ranker_top1_success_rate",
    "best_scored_success_top1_rate",
    "best_spl_success_top1_rate",
    "best_spl_success_top3_rate",
    "median_best_spl_success_rank",
    "top1_failed_rate",
    "high_conf_failed_rate_margin_gt_0_2",
    "failed_top1_is_final_rate",
    "failed_after_last_success_rate",
    "success_gt_fail_acc",
    "better_spl_success_acc",
    "val_minus_dev_ranker_top1_success_rate",
    "val_minus_dev_best_spl_success_top1_rate",
    "val_minus_dev_median_best_spl_success_rank",
    "val_minus_dev_better_spl_success_acc",
]

FEATURE_COLUMNS = [
    "gate_score",
    "score_margin_over_final",
    "trajectory_step_count",
    "candidate_count",
    "final_stop_prob",
    "max_stop_prob",
    "max_stop_minus_final_stop",
    "final_stop_rank_frac",
    "revisit_rate",
    "loop_region_rate",
    "last_k_rate",
    "selected_step_minus_final_step",
    "selected_path_delta_m",
    "selected_distance_delta_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phase-4.8 no-training gate generalization diagnostics.",
    )
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--endpoint-learning-dir", default=None)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--episode-csv", default=None)
    parser.add_argument("--gate-features-csv", default=None)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--pair-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ranker-diagnostics-dir", default=None)
    parser.add_argument("--target-scope", default=DEFAULT_TARGET_SCOPE)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument(
        "--gate-thresholds",
        default=",".join(format_float(value) for value in DEFAULT_GATE_THRESHOLDS),
    )
    parser.add_argument("--taus", default=",".join(format_float(value) for value in DEFAULT_TAUS))
    parser.add_argument(
        "--feature-configs",
        default=DEFAULT_FEATURE_CONFIGS,
        help="Comma-separated name:gate_threshold:tau triples for changed-item feature shift.",
    )
    parser.add_argument(
        "--skip-ranker-diagnostics",
        action="store_true",
        help="Reuse existing ranker diagnostics in --ranker-diagnostics-dir instead of generating them.",
    )
    parser.add_argument("--failure-sample-size", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    endpoint_learning_dir = resolve_endpoint_learning_dir(experiment_dir, args.endpoint_learning_dir)
    candidate_csv = resolve_path(
        args.candidate_csv,
        endpoint_learning_dir / "candidate_groups" / "endpoint_candidates.csv",
    )
    episode_csv = resolve_path(
        args.episode_csv,
        endpoint_learning_dir / "candidate_groups" / "episode_groups.csv",
    )
    gate_features_csv = resolve_path(
        args.gate_features_csv,
        endpoint_learning_dir / "gate_baseline" / "gate_features.csv",
    )
    pair_csv = resolve_optional_path(
        args.pair_csv,
        endpoint_learning_dir / "preference_pairs" / "preference_pairs.csv",
    )
    output_dir = resolve_output_dir(args.output_dir, experiment_dir, endpoint_learning_dir)
    ranker_diagnostics_dir = resolve_ranker_diagnostics_dir(
        args.ranker_diagnostics_dir,
        experiment_dir,
        output_dir,
    )

    manifest = diagnose_gate_generalization(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        gate_features_csv=gate_features_csv,
        score_csv=Path(args.score_csv).resolve(),
        pair_csv=pair_csv,
        output_dir=output_dir,
        ranker_diagnostics_dir=ranker_diagnostics_dir,
        target_scope=args.target_scope,
        split_filters=parse_string_list(args.splits),
        gate_thresholds=tuple(parse_float_list(args.gate_thresholds)),
        taus=tuple(parse_float_list(args.taus)),
        feature_configs=parse_feature_configs(args.feature_configs),
        run_ranker_diagnostics=not args.skip_ranker_diagnostics,
        failure_sample_size=args.failure_sample_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def diagnose_gate_generalization(
    candidate_csv: Path,
    episode_csv: Path,
    gate_features_csv: Path,
    score_csv: Path,
    pair_csv: Path | None,
    output_dir: Path,
    ranker_diagnostics_dir: Path,
    target_scope: str,
    split_filters: tuple[str, ...],
    gate_thresholds: tuple[float, ...],
    taus: tuple[float, ...],
    feature_configs: tuple[dict[str, Any], ...],
    run_ranker_diagnostics: bool,
    failure_sample_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_features = load_gate_features(gate_features_csv, target_scope, split_filters)
    calibration, calibration_bins = build_gate_calibration_tables(gate_features, split_filters)

    grid_eval_dir = output_dir / "grid_eval_protocol"
    grid_eval_manifest = reranker_eval.evaluate_endpoint_reranker(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=score_csv,
        output_dir=grid_eval_dir,
        target_scope=target_scope,
        split_filters=split_filters,
        gate_thresholds=gate_thresholds,
        taus=taus,
        allow_change_final_values=(True,),
        candidate_score_column="candidate_score",
        gate_score_column="gate_score",
        default_gate_score=1.0,
    )
    item_csv = resolve_manifest_path(grid_eval_manifest["files"]["items_csv"])
    items = pd.read_csv(item_csv, low_memory=False)
    items = normalize_eval_items(items, split_filters)
    threshold_confusion = build_threshold_confusion(items)
    threshold_confusion_all = threshold_confusion[threshold_confusion["dataset"] == "ALL"].copy()
    threshold_confusion_by_dataset = threshold_confusion[threshold_confusion["dataset"] != "ALL"].copy()

    feature_shift, feature_effects = build_changed_feature_shift(
        items=items,
        gate_features=gate_features,
        feature_configs=feature_configs,
    )

    ranker_manifest = None
    if run_ranker_diagnostics or not (ranker_diagnostics_dir / "manifest.json").exists():
        ranker_manifest = ranker_diag.build_ranker_top1_diagnostics(
            candidate_csv=candidate_csv,
            episode_csv=episode_csv,
            score_csv=score_csv,
            pair_csv=pair_csv,
            selected_items_csv=None,
            failure_slice_summary_csv=None,
            output_dir=ranker_diagnostics_dir,
            target_scope=target_scope,
            split_filters=split_filters,
            failure_sample_size=failure_sample_size,
        )
    else:
        ranker_manifest = read_json(ranker_diagnostics_dir / "manifest.json")

    ranker_stability = build_ranker_stability_table(
        ranker_diagnostics_dir=ranker_diagnostics_dir,
        pair_csv=pair_csv,
        score_csv=score_csv,
        target_scope=target_scope,
        split_filters=split_filters,
    )
    ranker_stability_all = ranker_stability[ranker_stability["dataset"] == "ALL"].copy()
    ranker_stability_by_dataset = ranker_stability[ranker_stability["dataset"] != "ALL"].copy()

    files = write_outputs(
        output_dir=output_dir,
        calibration=calibration,
        calibration_bins=calibration_bins,
        threshold_confusion_all=threshold_confusion_all,
        threshold_confusion_by_dataset=threshold_confusion_by_dataset,
        feature_shift=feature_shift,
        feature_effects=feature_effects,
        ranker_stability_all=ranker_stability_all,
        ranker_stability_by_dataset=ranker_stability_by_dataset,
    )

    diagnosis = build_diagnosis(
        calibration=calibration,
        threshold_confusion_all=threshold_confusion_all,
        feature_effects=feature_effects,
        ranker_stability_all=ranker_stability_all,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_training": False,
        "val_unseen_used_for_selection": False,
        "target_scope": target_scope,
        "split_filters": list(split_filters),
        "gate_thresholds": list(gate_thresholds),
        "taus": list(taus),
        "feature_configs": feature_configs,
        "inputs": {
            "candidate_csv": path_to_string(candidate_csv),
            "episode_csv": path_to_string(episode_csv),
            "gate_features_csv": path_to_string(gate_features_csv),
            "score_csv": path_to_string(score_csv),
            "pair_csv": path_to_string(pair_csv),
        },
        "source_eval_outputs": grid_eval_manifest,
        "source_ranker_diagnostics": ranker_manifest,
        "diagnosis": diagnosis,
        "files": files,
    }
    manifest_json = output_dir / "manifest.json"
    report_md = output_dir / "gate_generalization_diagnostics_report.md"
    manifest["files"]["manifest_json"] = path_to_string(manifest_json)
    manifest["files"]["gate_generalization_diagnostics_report_md"] = path_to_string(report_md)

    write_json(manifest_json, manifest)
    write_report(
        report_md,
        manifest=manifest,
        calibration=calibration,
        threshold_confusion_all=threshold_confusion_all,
        feature_effects=feature_effects,
        ranker_stability_all=ranker_stability_all,
    )
    return manifest


def load_gate_features(path: Path, target_scope: str, split_filters: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "target_scope" in frame.columns:
        frame = frame[frame["target_scope"].astype(str) == target_scope].copy()
    frame = select_splits(frame, split_filters)
    if frame.empty:
        raise ValueError(f"No gate feature rows matched target_scope={target_scope!r}, splits={split_filters!r}")
    frame = frame.drop_duplicates("episode_id", keep="last").copy()
    frame["_split_filter"] = assign_split_filter(frame, split_filters)
    frame["should_rerank"] = to_bool_series(frame["should_rerank"])
    frame["final_success"] = to_bool_series(frame["final_success"])
    frame["oracle_success"] = to_bool_series(frame["oracle_success"])
    frame["gate_score"] = pd.to_numeric(frame["gate_score"], errors="coerce")
    return frame[frame["_split_filter"].notna()].copy()


def build_gate_calibration_tables(
    gate_features: pd.DataFrame,
    split_filters: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for split_filter in split_filters:
        split_frame = gate_features[gate_features["_split_filter"] == split_filter].copy()
        for dataset, group in dataset_groups(split_frame):
            rows.append(calibration_row(split_filter, dataset, group))
            bin_rows.extend(calibration_bin_rows(split_filter, dataset, group))

    calibration = pd.DataFrame(rows)
    calibration = add_val_minus_dev_columns(
        calibration,
        key_columns=("dataset",),
        metric_columns=(
            "base_rate",
            "mean_gate_score",
            "calibration_gap",
            "roc_auc",
            "average_precision",
            "brier",
            "ece_10bin",
        ),
    )
    calibration = ensure_columns(calibration, CALIBRATION_COLUMNS)
    bins = ensure_columns(pd.DataFrame(bin_rows), CALIBRATION_BIN_COLUMNS)
    return calibration[CALIBRATION_COLUMNS], bins[CALIBRATION_BIN_COLUMNS]


def calibration_row(split_filter: str, dataset: str, group: pd.DataFrame) -> dict[str, Any]:
    y_true = group["should_rerank"].astype(int)
    y_score = pd.to_numeric(group["gate_score"], errors="coerce")
    valid = y_true.notna() & y_score.notna()
    y_true = y_true[valid]
    y_score = y_score[valid]
    positives = int(y_true.sum())
    items = int(len(y_true))
    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]
    return {
        "split_filter": split_filter,
        "dataset": dataset,
        "items": items,
        "positives": positives,
        "base_rate": safe_divide(positives, items),
        "mean_gate_score": safe_mean(y_score),
        "calibration_gap": safe_mean(y_score) - safe_divide(positives, items),
        "roc_auc": binary_metric(roc_auc_score, y_true, y_score),
        "average_precision": binary_metric(average_precision_score, y_true, y_score),
        "brier": binary_metric(brier_score_loss, y_true, y_score),
        "ece_10bin": expected_calibration_error(y_true, y_score, bins=10),
        "mean_score_positive": safe_mean(positive_scores),
        "mean_score_negative": safe_mean(negative_scores),
    }


def calibration_bin_rows(split_filter: str, dataset: str, group: pd.DataFrame) -> list[dict[str, Any]]:
    frame = group[["should_rerank", "gate_score"]].copy()
    frame["gate_score"] = pd.to_numeric(frame["gate_score"], errors="coerce")
    frame = frame.dropna(subset=["gate_score"])
    rows: list[dict[str, Any]] = []
    for index in range(10):
        low = index / 10.0
        high = (index + 1) / 10.0
        if index == 9:
            subset = frame[(frame["gate_score"] >= low) & (frame["gate_score"] <= high)]
        else:
            subset = frame[(frame["gate_score"] >= low) & (frame["gate_score"] < high)]
        items = int(len(subset))
        positives = int(subset["should_rerank"].astype(int).sum()) if items else 0
        mean_score = safe_mean(subset["gate_score"]) if items else math.nan
        base_rate = safe_divide(positives, items)
        rows.append(
            {
                "split_filter": split_filter,
                "dataset": dataset,
                "bin_index": index,
                "bin_low": low,
                "bin_high": high,
                "items": items,
                "positives": positives,
                "base_rate": base_rate,
                "mean_gate_score": mean_score,
                "calibration_gap": mean_score - base_rate if math.isfinite(mean_score) and math.isfinite(base_rate) else math.nan,
            }
        )
    return rows


def normalize_eval_items(items: pd.DataFrame, split_filters: tuple[str, ...]) -> pd.DataFrame:
    frame = items.copy()
    frame["_split_filter"] = assign_split_filter(frame, split_filters)
    frame = frame[frame["_split_filter"].notna()].copy()
    for column in [
        "allow_change_final",
        "gate_passed",
        "selected_changed",
        "final_success",
        "selected_success",
        "should_rerank",
        "recovered",
        "harmed",
    ]:
        if column in frame.columns:
            frame[column] = to_bool_series(frame[column])
    for column in [
        "gate_threshold",
        "tau",
        "gate_score",
        "final_score",
        "selected_score",
        "best_score",
        "score_margin_over_final",
        "final_step",
        "selected_step",
        "best_step",
        "final_spl",
        "selected_spl",
        "final_distance_m",
        "selected_distance_m",
        "final_path_length_m",
        "selected_path_length_m",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["outcome_class"] = frame.apply(classify_outcome, axis=1)
    frame["selected_step_minus_final_step"] = frame["selected_step"] - frame["final_step"]
    frame["selected_path_delta_m"] = frame["selected_path_length_m"] - frame["final_path_length_m"]
    frame["selected_distance_delta_m"] = frame["selected_distance_m"] - frame["final_distance_m"]
    return frame


def classify_outcome(row: pd.Series) -> str:
    if not bool(row.get("selected_changed")):
        return "kept"
    final_success = bool(row.get("final_success"))
    selected_success = bool(row.get("selected_success"))
    if not final_success and selected_success:
        return "recovered"
    if final_success and not selected_success:
        return "harmed"
    if final_success and selected_success:
        return "changed_safe"
    return "changed_unrecovered"


def build_threshold_confusion(items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["_split_filter", "tau", "gate_threshold"]
    for key, group in items.groupby(group_columns, dropna=False, sort=True):
        split_filter, tau, gate_threshold = key
        rows.append(threshold_row(str(split_filter), "ALL", tau, gate_threshold, group))
    for key, group in items.groupby(["_split_filter", "dataset", "tau", "gate_threshold"], dropna=False, sort=True):
        split_filter, dataset, tau, gate_threshold = key
        rows.append(threshold_row(str(split_filter), str(dataset), tau, gate_threshold, group))
    frame = pd.DataFrame(rows)
    frame = add_val_minus_dev_columns(
        frame,
        key_columns=("dataset", "tau", "gate_threshold"),
        metric_columns=("recovery_rate", "harm_rate", "net_recovery_rate", "changed_rate"),
    )
    frame = ensure_columns(frame, THRESHOLD_COLUMNS)
    return frame[THRESHOLD_COLUMNS]


def threshold_row(
    split_filter: str,
    dataset: str,
    tau: float,
    gate_threshold: float,
    group: pd.DataFrame,
) -> dict[str, Any]:
    items = int(len(group))
    recovered = int((group["outcome_class"] == "recovered").sum())
    harmed = int((group["outcome_class"] == "harmed").sum())
    changed_safe = int((group["outcome_class"] == "changed_safe").sum())
    changed_unrecovered = int((group["outcome_class"] == "changed_unrecovered").sum())
    blocked = int((group["should_rerank"] & ~group["gate_passed"]).sum())
    recovery_rate = safe_divide(recovered, items)
    harm_rate = safe_divide(harmed, items)
    net = recovery_rate - harm_rate if math.isfinite(recovery_rate) and math.isfinite(harm_rate) else math.nan
    return {
        "split_filter": split_filter,
        "dataset": dataset,
        "tau": float(tau),
        "gate_threshold": float(gate_threshold),
        "items": items,
        "gate_pass_rate": safe_divide(int(group["gate_passed"].sum()), items),
        "changed_rate": safe_divide(int(group["selected_changed"].sum()), items),
        "recovery_rate": recovery_rate,
        "harm_rate": harm_rate,
        "net_recovery_rate": net,
        "changed_safe_rate": safe_divide(changed_safe, items),
        "changed_unrecovered_rate": safe_divide(changed_unrecovered, items),
        "recovered_items": recovered,
        "harmed_items": harmed,
        "changed_safe_items": changed_safe,
        "changed_unrecovered_items": changed_unrecovered,
        "harm_per_recovered": safe_divide(harmed, recovered),
        "blocked_should_rerank_rate": safe_divide(blocked, items),
        "passes_safety_line": bool(math.isfinite(harm_rate) and harm_rate <= 0.01 + EPS and math.isfinite(net) and net > EPS),
    }


def build_changed_feature_shift(
    items: pd.DataFrame,
    gate_features: pd.DataFrame,
    feature_configs: tuple[dict[str, Any], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_keep = [
        "episode_id",
        "trajectory_step_count",
        "candidate_count",
        "final_stop_prob",
        "max_stop_prob",
        "max_stop_minus_final_stop",
        "final_stop_rank_frac",
        "revisit_rate",
        "loop_region_rate",
        "last_k_rate",
    ]
    gate_keep = [column for column in gate_keep if column in gate_features.columns]
    feature_frame = items.merge(
        gate_features[gate_keep].drop_duplicates("episode_id"),
        on="episode_id",
        how="left",
        suffixes=("", "_gate"),
    )
    for column in FEATURE_COLUMNS:
        if column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    config_frames: list[pd.DataFrame] = []
    for config in feature_configs:
        subset = feature_frame[
            (np.isclose(feature_frame["gate_threshold"], float(config["gate_threshold"])))
            & (np.isclose(feature_frame["tau"], float(config["tau"])))
        ].copy()
        if subset.empty:
            continue
        subset["config"] = config["name"]
        config_frames.append(subset)
    if not config_frames:
        return pd.DataFrame(columns=FEATURE_SHIFT_COLUMNS), pd.DataFrame(columns=FEATURE_EFFECT_COLUMNS)
    selected = pd.concat(config_frames, ignore_index=True)
    selected = selected[selected["outcome_class"].isin(["recovered", "harmed", "changed_safe", "changed_unrecovered"])].copy()

    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(["_split_filter", "dataset", "config", "gate_threshold", "tau", "outcome_class"], dropna=False, sort=True):
        split_filter, dataset, config, gate_threshold, tau, outcome_class = key
        for feature in FEATURE_COLUMNS:
            if feature not in group.columns:
                continue
            rows.append(feature_summary_row(split_filter, dataset, config, gate_threshold, tau, outcome_class, feature, group[feature]))
    for key, group in selected.groupby(["_split_filter", "config", "gate_threshold", "tau", "outcome_class"], dropna=False, sort=True):
        split_filter, config, gate_threshold, tau, outcome_class = key
        for feature in FEATURE_COLUMNS:
            if feature not in group.columns:
                continue
            rows.append(feature_summary_row(split_filter, "ALL", config, gate_threshold, tau, outcome_class, feature, group[feature]))

    shift = pd.DataFrame(rows)
    if not shift.empty:
        shift = add_delta_vs_recovered(shift)
        shift = add_val_minus_dev_columns(
            shift,
            key_columns=("dataset", "config", "gate_threshold", "tau", "outcome_class", "feature"),
            metric_columns=("mean",),
            output_prefix="val_minus_dev",
            replace_single_metric=True,
        )
    shift = ensure_columns(shift, FEATURE_SHIFT_COLUMNS)

    effects = build_feature_effects(shift)
    return shift[FEATURE_SHIFT_COLUMNS], effects[FEATURE_EFFECT_COLUMNS]


def feature_summary_row(
    split_filter: str,
    dataset: str,
    config: str,
    gate_threshold: float,
    tau: float,
    outcome_class: str,
    feature: str,
    values: pd.Series,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "split_filter": split_filter,
        "dataset": dataset,
        "config": config,
        "gate_threshold": float(gate_threshold),
        "tau": float(tau),
        "outcome_class": outcome_class,
        "feature": feature,
        "items": int(len(numeric)),
        "mean": safe_mean(numeric),
        "median": safe_median(numeric),
        "p25": safe_quantile(numeric, 0.25),
        "p75": safe_quantile(numeric, 0.75),
        "std": safe_std(numeric),
    }


def add_delta_vs_recovered(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["delta_vs_recovered"] = math.nan
    keys = ["split_filter", "dataset", "config", "gate_threshold", "tau", "feature"]
    recovered = output[output["outcome_class"] == "recovered"].set_index(keys)["mean"].to_dict()
    for index, row in output.iterrows():
        base = recovered.get(tuple(row[key] for key in keys))
        value = row.get("mean")
        if base is not None and is_finite(value) and is_finite(base):
            output.at[index, "delta_vs_recovered"] = float(value) - float(base)
    return output


def build_feature_effects(shift: pd.DataFrame) -> pd.DataFrame:
    if shift.empty:
        return pd.DataFrame(columns=FEATURE_EFFECT_COLUMNS)
    rows: list[dict[str, Any]] = []
    keys = ["split_filter", "dataset", "config", "gate_threshold", "tau", "feature"]
    for key, group in shift.groupby(keys, dropna=False, sort=True):
        lookup = {row["outcome_class"]: row for _, row in group.iterrows()}
        recovered = lookup.get("recovered")
        harmed = lookup.get("harmed")
        if recovered is None or harmed is None:
            continue
        recovered_std = recovered.get("std")
        harmed_std = harmed.get("std")
        recovered_items = recovered.get("items")
        harmed_items = harmed.get("items")
        pooled_std = pooled_standard_deviation(recovered_std, harmed_std, recovered_items, harmed_items)
        harm_minus_recovered = subtract_or_nan(harmed.get("mean"), recovered.get("mean"))
        effect = safe_divide(harm_minus_recovered, pooled_std)
        split_filter, dataset, config, gate_threshold, tau, feature = key
        rows.append(
            {
                "split_filter": split_filter,
                "dataset": dataset,
                "config": config,
                "gate_threshold": gate_threshold,
                "tau": tau,
                "feature": feature,
                "recovered_items": recovered_items,
                "harmed_items": harmed_items,
                "recovered_mean": recovered.get("mean"),
                "harmed_mean": harmed.get("mean"),
                "harm_minus_recovered": harm_minus_recovered,
                "pooled_std": pooled_std,
                "standardized_effect": effect,
            }
        )
    effects = pd.DataFrame(rows)
    if not effects.empty:
        effects = add_val_minus_dev_columns(
            effects,
            key_columns=("dataset", "config", "gate_threshold", "tau", "feature"),
            metric_columns=("standardized_effect",),
            output_prefix="val_minus_dev_standardized_effect",
            replace_single_metric=True,
        )
    return ensure_columns(effects, FEATURE_EFFECT_COLUMNS)


def build_ranker_stability_table(
    ranker_diagnostics_dir: Path,
    pair_csv: Path | None,
    score_csv: Path,
    target_scope: str,
    split_filters: tuple[str, ...],
) -> pd.DataFrame:
    episode = pd.read_csv(ranker_diagnostics_dir / "episode_rank_diagnostics.csv", low_memory=False)
    if episode.empty:
        return pd.DataFrame(columns=RANKER_STABILITY_COLUMNS)
    for column in [
        "should_rerank",
        "ranker_top1_success",
        "ranker_top1_is_final",
    ]:
        if column in episode.columns:
            episode[column] = to_bool_series(episode[column])
    for column in [
        "best_scored_success_rank",
        "best_spl_success_rank",
        "top1_margin_over_best_success",
    ]:
        if column in episode.columns:
            episode[column] = pd.to_numeric(episode[column], errors="coerce")

    pair_agreement = build_pair_agreement_by_split(
        pair_csv=pair_csv,
        score_csv=score_csv,
        target_scope=target_scope,
        split_filters=split_filters,
    )

    rows: list[dict[str, Any]] = []
    for split_filter, split_frame in episode.groupby("split_filter", sort=True):
        for dataset, group in dataset_groups(split_frame):
            base = group[group["should_rerank"]].copy()
            failed = base[~base["ranker_top1_success"]].copy()
            pair_lookup: dict[str, pd.Series] = {}
            if not pair_agreement.empty:
                pair_rows = pair_agreement[
                    (pair_agreement["split_filter"] == split_filter)
                    & (pair_agreement["dataset"] == dataset)
                ]
                pair_lookup = {row["pair_type"]: row for _, row in pair_rows.iterrows()}
            success_pair = pair_lookup.get("success_gt_fail")
            better_spl_pair = pair_lookup.get("better_spl_success_gt_lower_spl_success")
            rows.append(
                {
                    "split_filter": split_filter,
                    "dataset": dataset,
                    "should_rerank_items": int(len(base)),
                    "ranker_top1_success_rate": mean_bool(base["ranker_top1_success"]),
                    "best_scored_success_top1_rate": mean_bool(base["best_scored_success_rank"] <= 1),
                    "best_spl_success_top1_rate": mean_bool(base["best_spl_success_rank"] <= 1),
                    "best_spl_success_top3_rate": mean_bool(base["best_spl_success_rank"] <= 3),
                    "median_best_spl_success_rank": safe_median(base["best_spl_success_rank"]),
                    "top1_failed_rate": safe_divide(len(failed), len(base)),
                    "high_conf_failed_rate_margin_gt_0_2": mean_bool(
                        pd.to_numeric(failed["top1_margin_over_best_success"], errors="coerce") > 0.2
                    ),
                    "failed_top1_is_final_rate": mean_bool(failed["ranker_top1_is_final"]),
                    "failed_after_last_success_rate": mean_bool(
                        failed["top1_location_vs_success_window"].astype(str) == "after_last_success"
                    ),
                    "success_gt_fail_acc": success_pair.get("accuracy") if success_pair is not None else math.nan,
                    "better_spl_success_acc": better_spl_pair.get("accuracy") if better_spl_pair is not None else math.nan,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = add_val_minus_dev_columns(
            frame,
            key_columns=("dataset",),
            metric_columns=(
                "ranker_top1_success_rate",
                "best_spl_success_top1_rate",
                "median_best_spl_success_rank",
                "better_spl_success_acc",
            ),
        )
    frame = ensure_columns(frame, RANKER_STABILITY_COLUMNS)
    return frame[RANKER_STABILITY_COLUMNS]


def build_pair_agreement_by_split(
    pair_csv: Path | None,
    score_csv: Path,
    target_scope: str,
    split_filters: tuple[str, ...],
) -> pd.DataFrame:
    if pair_csv is None or not pair_csv.exists():
        return pd.DataFrame()
    pairs = pd.read_csv(pair_csv, low_memory=False)
    if "target_scope" in pairs.columns:
        pairs = pairs[pairs["target_scope"].astype(str) == target_scope].copy()
    pairs = select_splits(pairs, split_filters)
    pairs["_split_filter"] = assign_split_filter(pairs, split_filters)
    pairs = pairs[pairs["_split_filter"].notna()].copy()
    if pairs.empty:
        return pd.DataFrame()

    scores = pd.read_csv(score_csv, low_memory=False, usecols=lambda column: column in {"candidate_id", "candidate_score"})
    scores["candidate_score"] = pd.to_numeric(scores["candidate_score"], errors="coerce")
    score_map = scores.drop_duplicates("candidate_id", keep="last").set_index("candidate_id")["candidate_score"]
    pairs["chosen_score"] = pairs["chosen_candidate_id"].map(score_map)
    pairs["rejected_score"] = pairs["rejected_candidate_id"].map(score_map)
    pairs = pairs.dropna(subset=["chosen_score", "rejected_score"]).copy()
    pairs["score_margin"] = pairs["chosen_score"] - pairs["rejected_score"]
    pairs["accuracy"] = pairs["score_margin"] > EPS

    rows: list[dict[str, Any]] = []
    for key, group in pairs.groupby(["_split_filter", "pair_type"], dropna=False, sort=True):
        split_filter, pair_type = key
        rows.append(pair_agreement_row(split_filter, "ALL", pair_type, group))
    for key, group in pairs.groupby(["_split_filter", "dataset", "pair_type"], dropna=False, sort=True):
        split_filter, dataset, pair_type = key
        rows.append(pair_agreement_row(split_filter, str(dataset), pair_type, group))
    return pd.DataFrame(rows)


def pair_agreement_row(split_filter: str, dataset: str, pair_type: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "split_filter": split_filter,
        "dataset": dataset,
        "pair_type": pair_type,
        "pairs": int(len(group)),
        "accuracy": mean_bool(group["accuracy"]),
        "mean_score_margin": safe_mean(group["score_margin"]),
    }


def write_outputs(
    output_dir: Path,
    calibration: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    threshold_confusion_all: pd.DataFrame,
    threshold_confusion_by_dataset: pd.DataFrame,
    feature_shift: pd.DataFrame,
    feature_effects: pd.DataFrame,
    ranker_stability_all: pd.DataFrame,
    ranker_stability_by_dataset: pd.DataFrame,
) -> dict[str, str | None]:
    paths = {
        "gate_calibration_by_split_dataset_csv": output_dir / "gate_calibration_by_split_dataset.csv",
        "gate_calibration_bins_csv": output_dir / "gate_calibration_bins.csv",
        "gate_threshold_confusion_csv": output_dir / "gate_threshold_confusion.csv",
        "gate_threshold_confusion_by_dataset_csv": output_dir / "gate_threshold_confusion_by_dataset.csv",
        "changed_item_feature_shift_csv": output_dir / "changed_item_feature_shift.csv",
        "changed_item_feature_shift_effects_csv": output_dir / "changed_item_feature_shift_effects.csv",
        "ranker_top1_stability_csv": output_dir / "ranker_top1_stability.csv",
        "ranker_top1_stability_by_dataset_csv": output_dir / "ranker_top1_stability_by_dataset.csv",
    }
    calibration.to_csv(paths["gate_calibration_by_split_dataset_csv"], index=False)
    calibration_bins.to_csv(paths["gate_calibration_bins_csv"], index=False)
    threshold_confusion_all.to_csv(paths["gate_threshold_confusion_csv"], index=False)
    threshold_confusion_by_dataset.to_csv(paths["gate_threshold_confusion_by_dataset_csv"], index=False)
    feature_shift.to_csv(paths["changed_item_feature_shift_csv"], index=False)
    feature_effects.to_csv(paths["changed_item_feature_shift_effects_csv"], index=False)
    ranker_stability_all.to_csv(paths["ranker_top1_stability_csv"], index=False)
    ranker_stability_by_dataset.to_csv(paths["ranker_top1_stability_by_dataset_csv"], index=False)
    return {name: path_to_string(path) for name, path in paths.items()}


def build_diagnosis(
    calibration: pd.DataFrame,
    threshold_confusion_all: pd.DataFrame,
    feature_effects: pd.DataFrame,
    ranker_stability_all: pd.DataFrame,
) -> dict[str, Any]:
    all_cal = calibration[calibration["dataset"] == "ALL"].copy()
    val_cal = all_cal[all_cal["split_filter"] == "val_unseen"]
    gate_drift = "inconclusive"
    if not val_cal.empty:
        row = val_cal.iloc[0]
        brier_drift = abs(float_or_nan(row.get("val_minus_dev_brier")))
        ece_drift = abs(float_or_nan(row.get("val_minus_dev_ece_10bin")))
        auc_drift = float_or_nan(row.get("val_minus_dev_roc_auc"))
        if brier_drift >= 0.02 or ece_drift >= 0.02 or auc_drift <= -0.05:
            gate_drift = "yes"
        else:
            gate_drift = "no"

    common_thresholds = threshold_confusion_all[
        (threshold_confusion_all["split_filter"] == "val_unseen")
        & (threshold_confusion_all["passes_safety_line"] == True)  # noqa: E712
        & (pd.to_numeric(threshold_confusion_all["val_minus_dev_net_recovery_rate"], errors="coerce").fillna(-math.inf) >= -0.001)
    ]
    stable_threshold_band = "yes" if not common_thresholds.empty else "no"

    ranker_drift = "inconclusive"
    val_ranker = ranker_stability_all[ranker_stability_all["split_filter"] == "val_unseen"]
    if not val_ranker.empty:
        row = val_ranker.iloc[0]
        top1_drift = float_or_nan(row.get("val_minus_dev_ranker_top1_success_rate"))
        median_rank_drift = float_or_nan(row.get("val_minus_dev_median_best_spl_success_rank"))
        if top1_drift <= -0.05 or median_rank_drift >= 1.0:
            ranker_drift = "yes"
        else:
            ranker_drift = "no"

    feature_separation = "inconclusive"
    all_effects = feature_effects[
        (feature_effects["dataset"] == "ALL")
        & (feature_effects["split_filter"] == "val_unseen")
    ].copy()
    if not all_effects.empty:
        effects = pd.to_numeric(all_effects["standardized_effect"], errors="coerce").abs().dropna()
        if effects.empty:
            feature_separation = "inconclusive"
        elif float(effects.median()) >= 0.8:
            feature_separation = "clearly separated"
        elif float(effects.quantile(0.75)) <= 0.3:
            feature_separation = "overlapping"
        else:
            feature_separation = "dataset-specific"

    if gate_drift == "yes" and ranker_drift == "yes":
        primary_breakpoint = "both"
    elif gate_drift == "yes":
        primary_breakpoint = "gate"
    elif ranker_drift == "yes":
        primary_breakpoint = "ranker"
    elif stable_threshold_band == "no":
        primary_breakpoint = "gate"
    else:
        primary_breakpoint = "inconclusive"

    return {
        "gate_calibration_drift": gate_drift,
        "stable_threshold_band": stable_threshold_band,
        "changed_feature_separation": feature_separation,
        "ranker_top1_drift": ranker_drift,
        "primary_breakpoint": primary_breakpoint,
    }


def write_report(
    path: Path,
    manifest: dict[str, Any],
    calibration: pd.DataFrame,
    threshold_confusion_all: pd.DataFrame,
    feature_effects: pd.DataFrame,
    ranker_stability_all: pd.DataFrame,
) -> None:
    diagnosis = manifest["diagnosis"]
    lines = [
        "# Endpoint Gate Generalization Diagnostics",
        "",
        "This report is generated by `scripts/analysis/diagnose_endpoint_gate_generalization.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        "- model_training: `false`",
        "- val_unseen_used_for_selection: `false`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- split_filters: `{','.join(manifest['split_filters'])}`",
        f"- gate_thresholds: `{','.join(format_float(value) for value in manifest['gate_thresholds'])}`",
        f"- taus: `{','.join(format_float(value) for value in manifest['taus'])}`",
        "",
        "## Diagnostic Answers",
        "",
        markdown_table(
            ["question", "answer"],
            [
                ["gate calibration drift", diagnosis["gate_calibration_drift"]],
                ["stable threshold band", diagnosis["stable_threshold_band"]],
                ["changed feature separation", diagnosis["changed_feature_separation"]],
                ["ranker top1 drift", diagnosis["ranker_top1_drift"]],
                ["primary breakpoint", diagnosis["primary_breakpoint"]],
            ],
        ),
        "",
        "## Table A: Gate Calibration",
        "",
        markdown_table(
            ["split", "dataset", "items", "base", "mean_gate", "gap", "AUC", "AP", "brier", "ECE"],
            [
                [
                    row["split_filter"],
                    row["dataset"],
                    int(row["items"]),
                    pct(row["base_rate"]),
                    pct(row["mean_gate_score"]),
                    pct(row["calibration_gap"]),
                    fmt(row["roc_auc"]),
                    fmt(row["average_precision"]),
                    fmt(row["brier"]),
                    fmt(row["ece_10bin"]),
                ]
                for _, row in calibration[calibration["dataset"] == "ALL"].iterrows()
            ],
        ),
        "",
        "## Table B: Threshold Confusion",
        "",
        markdown_table(
            ["split", "tau", "gate", "changed", "recovery", "harm", "net", "recovered", "harmed", "safe_line"],
            [
                [
                    row["split_filter"],
                    fmt(row["tau"]),
                    fmt(row["gate_threshold"]),
                    pct(row["changed_rate"]),
                    pct(row["recovery_rate"]),
                    pct(row["harm_rate"]),
                    pct(row["net_recovery_rate"]),
                    int(row["recovered_items"]),
                    int(row["harmed_items"]),
                    str(bool(row["passes_safety_line"])).lower(),
                ]
                for _, row in threshold_confusion_all.sort_values(
                    ["split_filter", "tau", "gate_threshold"],
                    ascending=[True, True, True],
                ).iterrows()
                if row["gate_threshold"] in {0.90, 0.95}
            ],
        ),
        "",
        "## Table C: Feature Effects",
        "",
        markdown_table(
            ["split", "config", "feature", "rec_items", "harm_items", "harm-rec", "std_effect"],
            [
                [
                    row["split_filter"],
                    row["config"],
                    row["feature"],
                    int(row["recovered_items"]),
                    int(row["harmed_items"]),
                    fmt(row["harm_minus_recovered"]),
                    fmt(row["standardized_effect"]),
                ]
                for _, row in top_feature_effect_rows(feature_effects).iterrows()
            ],
        ),
        "",
        "## Table D: Ranker Top1 Stability",
        "",
        markdown_table(
            [
                "split",
                "dataset",
                "should",
                "top1_success",
                "best_spl_top1",
                "best_spl_top3",
                "median_rank",
                "better_spl_acc",
            ],
            [
                [
                    row["split_filter"],
                    row["dataset"],
                    int(row["should_rerank_items"]),
                    pct(row["ranker_top1_success_rate"]),
                    pct(row["best_spl_success_top1_rate"]),
                    pct(row["best_spl_success_top3_rate"]),
                    fmt(row["median_best_spl_success_rank"]),
                    pct(row["better_spl_success_acc"]),
                ]
                for _, row in ranker_stability_all.iterrows()
            ],
        ),
        "",
        "## Files",
        "",
    ]
    for key, value in manifest["files"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_feature_effect_rows(feature_effects: pd.DataFrame) -> pd.DataFrame:
    if feature_effects.empty:
        return feature_effects
    frame = feature_effects[
        (feature_effects["dataset"] == "ALL")
        & (feature_effects["split_filter"].isin(["dev", "val_unseen"]))
    ].copy()
    frame["_abs_effect"] = pd.to_numeric(frame["standardized_effect"], errors="coerce").abs()
    return frame.sort_values(["split_filter", "config", "_abs_effect"], ascending=[True, True, False]).groupby(
        ["split_filter", "config"],
        sort=False,
    ).head(5)


def add_val_minus_dev_columns(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    metric_columns: tuple[str, ...],
    output_prefix: str = "val_minus_dev",
    replace_single_metric: bool = False,
) -> pd.DataFrame:
    output = frame.copy()
    for metric in metric_columns:
        column = output_prefix if replace_single_metric and len(metric_columns) == 1 else f"{output_prefix}_{metric}"
        output[column] = math.nan
    if output.empty:
        return output
    dev = output[output["split_filter"] == "dev"].set_index(list(key_columns))
    for index, row in output.iterrows():
        if row.get("split_filter") != "val_unseen":
            continue
        key = row[key_columns[0]] if len(key_columns) == 1 else tuple(row[column] for column in key_columns)
        if key not in dev.index:
            continue
        dev_row = dev.loc[key]
        if isinstance(dev_row, pd.DataFrame):
            dev_row = dev_row.iloc[0]
        for metric in metric_columns:
            column = output_prefix if replace_single_metric and len(metric_columns) == 1 else f"{output_prefix}_{metric}"
            output.at[index, column] = subtract_or_nan(row.get(metric), dev_row.get(metric))
    return output


def select_splits(frame: pd.DataFrame, split_filters: tuple[str, ...]) -> pd.DataFrame:
    if not split_filters or "all" in split_filters:
        return frame.copy()
    mask = pd.Series(False, index=frame.index)
    if "split" in frame.columns:
        mask |= frame["split"].astype(str).isin(split_filters)
    if "protocol_split" in frame.columns:
        mask |= frame["protocol_split"].astype(str).isin(split_filters)
    return frame[mask].copy()


def assign_split_filter(frame: pd.DataFrame, split_filters: tuple[str, ...]) -> pd.Series:
    values = pd.Series(pd.NA, index=frame.index, dtype="object")
    for split_filter in split_filters:
        mask = pd.Series(False, index=frame.index)
        if "split" in frame.columns:
            mask |= frame["split"].astype(str) == split_filter
        if "protocol_split" in frame.columns:
            mask |= frame["protocol_split"].astype(str) == split_filter
        values = values.mask(values.isna() & mask, split_filter)
    return values


def dataset_groups(frame: pd.DataFrame):
    yield "ALL", frame
    if "dataset" in frame.columns:
        for dataset, group in frame.groupby("dataset", sort=True):
            yield str(dataset), group


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    result = normalized.isin(BOOL_TRUE)
    result = result.mask(normalized.isin(BOOL_FALSE), False)
    return result.fillna(False).astype(bool)


def binary_metric(metric_fn, y_true: pd.Series, y_score: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")
    valid = y_score.notna()
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(y_true) == 0:
        return math.nan
    if metric_fn in (roc_auc_score, average_precision_score) and y_true.nunique() < 2:
        return math.nan
    try:
        return float(metric_fn(y_true.to_numpy(), y_score.to_numpy()))
    except ValueError:
        return math.nan


def expected_calibration_error(y_true: pd.Series, y_score: pd.Series, bins: int = 10) -> float:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")
    valid = y_score.notna()
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(y_true) == 0:
        return math.nan
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index == bins - 1:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        if not mask.any():
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
    return ece


def pooled_standard_deviation(left_std: Any, right_std: Any, left_count: Any, right_count: Any) -> float:
    left_std = float_or_nan(left_std)
    right_std = float_or_nan(right_std)
    left_count = float_or_nan(left_count)
    right_count = float_or_nan(right_count)
    if not all(math.isfinite(value) for value in [left_std, right_std, left_count, right_count]):
        return math.nan
    if left_count + right_count <= 2:
        return math.nan
    numerator = max(left_count - 1, 0) * left_std**2 + max(right_count - 1, 0) * right_std**2
    return math.sqrt(numerator / max(left_count + right_count - 2, 1))


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = math.nan
    return output


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def mean_bool(values: pd.Series) -> float:
    series = pd.Series(values)
    if len(series) == 0:
        return math.nan
    if series.dtype != bool:
        series = to_bool_series(series)
    return float(series.mean()) if len(series) else math.nan


def safe_median(values: pd.Series) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.median()) if len(values) else math.nan


def safe_quantile(values: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.quantile(quantile)) if len(values) else math.nan


def safe_std(values: pd.Series) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan


def safe_divide(numerator: Any, denominator: Any) -> float:
    numerator = float_or_nan(numerator)
    denominator = float_or_nan(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= EPS:
        return math.nan
    return numerator / denominator


def subtract_or_nan(left: Any, right: Any) -> float:
    left = float_or_nan(left)
    right = float_or_nan(right)
    if not math.isfinite(left) or not math.isfinite(right):
        return math.nan
    return left - right


def float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def is_finite(value: Any) -> bool:
    return math.isfinite(float_or_nan(value))


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def resolve_path(raw: str | None, default: Path) -> Path:
    return Path(raw).resolve() if raw else default.resolve()


def resolve_optional_path(raw: str | None, default: Path) -> Path | None:
    if raw:
        return Path(raw).resolve()
    return default.resolve() if default.exists() else None


def resolve_output_dir(raw: str | None, experiment_dir: Path | None, endpoint_learning_dir: Path) -> Path:
    if raw:
        return Path(raw).resolve()
    if experiment_dir is not None:
        return (experiment_dir / DEFAULT_OUTPUT_SUBDIR).resolve()
    return (endpoint_learning_dir / "gate_generalization_diagnostics" / "phase4_8_final_bias_sanity").resolve()


def resolve_ranker_diagnostics_dir(raw: str | None, experiment_dir: Path | None, output_dir: Path) -> Path:
    if raw:
        return Path(raw).resolve()
    if experiment_dir is not None:
        return (experiment_dir / DEFAULT_RANKER_DIAG_SUBDIR).resolve()
    return (output_dir / "ranker_top1_diagnostics").resolve()


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def parse_string_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.replace("/", ",").split(",") if part.strip())


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        parsed = float(text)
        if not math.isfinite(parsed):
            raise ValueError(f"Non-finite float: {text}")
        values.append(parsed)
    if not values:
        raise ValueError("Expected at least one float")
    return values


def parse_feature_configs(value: str) -> tuple[dict[str, Any], ...]:
    configs: list[dict[str, Any]] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        pieces = text.split(":")
        if len(pieces) != 3:
            raise ValueError(f"Feature config must be name:gate:tau, got {text!r}")
        configs.append(
            {
                "name": pieces[0],
                "gate_threshold": float(pieces[1]),
                "tau": float(pieces[2]),
            }
        )
    if not configs:
        raise ValueError("Expected at least one feature config")
    return tuple(configs)


def path_to_string(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def format_float(value: float) -> str:
    return format(float(value), ".12g")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    value = float_or_nan(value)
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def pct(value: Any) -> str:
    value = float_or_nan(value)
    return "" if not math.isfinite(value) else f"{value * 100:.2f}"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
