#!/usr/bin/env python3
"""Diagnose group top-1 failures for the candidate-level endpoint ranker."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_ranker_top1_diagnostics.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_OUTPUT_NAME = "ranker_diagnostics"
DEFAULT_REPORTS_SUBDIR = Path("reports") / "endpoint_ranker_diagnostics"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_SPLITS = ("dev",)
EPS = 1e-12

BOOL_COLUMNS = [
    "trace_available",
    "has_decision_trace",
    "is_route_intermediate",
    "is_route_expanded_without_decision",
    "is_final",
    "is_revisit",
    "is_loop_region",
    "is_last_k",
    "final_success",
    "oracle_success",
    "should_rerank",
    "success_label",
    "is_best_success_candidate",
    "is_nearest_candidate",
]

NUMERIC_COLUMNS = [
    "trajectory_step_count",
    "candidate_step",
    "step_frac",
    "path_length_m",
    "stop_prob",
    "stop_margin_prob",
    "selected_prob",
    "top1_top2_margin",
    "moe_router_entropy",
    "fuse_weight",
    "final_step",
    "spl_at_candidate",
    "distance_to_goal_m",
    "reward",
    "candidate_score",
    "gate_score",
]

TARGET_PREFIXES = [
    "best_scored_success",
    "nearest",
    "first_success",
    "best_spl_success",
    "final",
]

DELTA_TARGETS = [
    "best_scored_success",
    "nearest",
    "first_success",
    "best_spl_success",
]

DELTA_FEATURES = [
    ("score", "candidate_score"),
    ("step", "candidate_step"),
    ("step_frac", "step_frac"),
    ("path_length_m", "path_length_m"),
    ("stop_prob", "stop_prob"),
    ("fuse_weight", "fuse_weight"),
    ("distance_m", "distance_to_goal_m"),
    ("spl", "spl_at_candidate"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build no-training group top-1 diagnostics for endpoint ranker scores.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Experiment output directory. Used to infer endpoint_learning paths.",
    )
    parser.add_argument(
        "--endpoint-learning-dir",
        default=None,
        help="Endpoint learning directory. Defaults to <experiment-dir>/endpoint_learning.",
    )
    parser.add_argument(
        "--candidate-csv",
        default=None,
        help="Endpoint candidates CSV. Defaults to <endpoint-learning-dir>/candidate_groups/endpoint_candidates.csv.",
    )
    parser.add_argument(
        "--episode-csv",
        default=None,
        help="Episode groups CSV. Defaults to <endpoint-learning-dir>/candidate_groups/episode_groups.csv.",
    )
    parser.add_argument(
        "--score-csv",
        default=None,
        help="Ranker score CSV. Defaults to <endpoint-learning-dir>/ranker_baseline/ranker_scores.csv.",
    )
    parser.add_argument(
        "--pair-csv",
        default=None,
        help="Preference pairs CSV. Defaults to <endpoint-learning-dir>/preference_pairs/preference_pairs.csv.",
    )
    parser.add_argument(
        "--selected-items-csv",
        default=None,
        help="Optional frozen selected items CSV. Defaults to <endpoint-learning-dir>/frozen_gate_ranker/dev_selected_items.csv.",
    )
    parser.add_argument(
        "--failure-slice-summary-csv",
        default=None,
        help="Optional frozen failure slice summary CSV. Defaults to <endpoint-learning-dir>/frozen_gate_ranker/failure_slice_summary.csv.",
    )
    parser.add_argument(
        "--target-scope",
        default=DEFAULT_TARGET_SCOPE,
        help="Target scope to diagnose. Defaults to official.",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated protocol/eval splits to diagnose. Defaults to dev.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to <experiment-dir>/reports/endpoint_ranker_diagnostics "
            "when --experiment-dir is provided."
        ),
    )
    parser.add_argument(
        "--failure-sample-size",
        type=int,
        default=200,
        help="Number of high-margin failed top1 episodes to export.",
    )
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
    score_csv = resolve_path(
        args.score_csv,
        endpoint_learning_dir / "ranker_baseline" / "ranker_scores.csv",
    )
    pair_csv = resolve_optional_path(
        args.pair_csv,
        endpoint_learning_dir / "preference_pairs" / "preference_pairs.csv",
    )
    selected_items_csv = resolve_optional_path(
        args.selected_items_csv,
        endpoint_learning_dir / "frozen_gate_ranker" / "dev_selected_items.csv",
    )
    failure_slice_summary_csv = resolve_optional_path(
        args.failure_slice_summary_csv,
        endpoint_learning_dir / "frozen_gate_ranker" / "failure_slice_summary.csv",
    )
    output_dir = resolve_path(
        args.output_dir,
        default_output_dir(experiment_dir, endpoint_learning_dir),
    )
    splits = parse_string_list(args.splits)

    manifest = build_ranker_top1_diagnostics(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=score_csv,
        pair_csv=pair_csv,
        selected_items_csv=selected_items_csv,
        failure_slice_summary_csv=failure_slice_summary_csv,
        output_dir=output_dir,
        target_scope=args.target_scope,
        split_filters=splits,
        failure_sample_size=args.failure_sample_size,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_ranker_top1_diagnostics(
    candidate_csv: Path,
    episode_csv: Path,
    score_csv: Path,
    pair_csv: Path | None,
    selected_items_csv: Path | None,
    failure_slice_summary_csv: Path | None,
    output_dir: Path,
    target_scope: str,
    split_filters: tuple[str, ...],
    failure_sample_size: int,
) -> dict[str, Any]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write diagnostics output to {output_dir}. "
            "Fix the experiment directory permissions or pass --output-dir explicitly."
        ) from exc
    candidates = load_scored_candidates(candidate_csv, score_csv, target_scope, split_filters)
    selected_items = load_selected_items(selected_items_csv)

    episode_rows = []
    for split_filter in split_filters:
        split_frame = select_split(candidates, split_filter)
        if split_frame.empty:
            continue
        episode_rows.append(build_episode_diagnostics(split_frame, split_filter))
    if episode_rows:
        episode_diagnostics = pd.concat(episode_rows, ignore_index=True)
    else:
        episode_diagnostics = pd.DataFrame()

    if selected_items is not None and not episode_diagnostics.empty:
        episode_diagnostics = episode_diagnostics.merge(
            selected_items,
            on="episode_id",
            how="left",
        )

    target_rank_summary = build_target_rank_summary(episode_diagnostics)
    top1_failure_summary = build_top1_failure_summary(episode_diagnostics)
    group_size_summary = build_group_size_summary(episode_diagnostics)
    feature_delta_summary = build_feature_delta_summary(episode_diagnostics)
    pair_agreement = build_pair_agreement(pair_csv, candidates, target_scope, split_filters)
    failure_samples = build_failure_samples(episode_diagnostics, failure_sample_size)
    frozen_failure_slice_summary = read_optional_csv(failure_slice_summary_csv)

    episode_diagnostics_csv = output_dir / "episode_rank_diagnostics.csv"
    target_rank_summary_csv = output_dir / "target_rank_summary.csv"
    top1_failure_summary_csv = output_dir / "top1_failure_summary.csv"
    group_size_summary_csv = output_dir / "top1_by_group_size.csv"
    feature_delta_summary_csv = output_dir / "top1_failure_feature_deltas.csv"
    pair_agreement_csv = output_dir / "pair_agreement_by_type.csv"
    failure_samples_csv = output_dir / "top1_failure_samples.csv"
    report_md = output_dir / "ranker_diagnostics_report.md"
    manifest_json = output_dir / "manifest.json"

    episode_diagnostics.to_csv(episode_diagnostics_csv, index=False)
    target_rank_summary.to_csv(target_rank_summary_csv, index=False)
    top1_failure_summary.to_csv(top1_failure_summary_csv, index=False)
    group_size_summary.to_csv(group_size_summary_csv, index=False)
    feature_delta_summary.to_csv(feature_delta_summary_csv, index=False)
    pair_agreement.to_csv(pair_agreement_csv, index=False)
    failure_samples.to_csv(failure_samples_csv, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": target_scope,
        "split_filters": list(split_filters),
        "inputs": {
            "candidate_csv": path_to_string(candidate_csv),
            "episode_csv": path_to_string(episode_csv),
            "score_csv": path_to_string(score_csv),
            "pair_csv": path_to_string(pair_csv) if pair_csv else None,
            "selected_items_csv": path_to_string(selected_items_csv) if selected_items_csv else None,
            "failure_slice_summary_csv": (
                path_to_string(failure_slice_summary_csv) if failure_slice_summary_csv else None
            ),
        },
        "outputs": {
            "episode_rank_diagnostics_csv": path_to_string(episode_diagnostics_csv),
            "target_rank_summary_csv": path_to_string(target_rank_summary_csv),
            "top1_failure_summary_csv": path_to_string(top1_failure_summary_csv),
            "top1_by_group_size_csv": path_to_string(group_size_summary_csv),
            "top1_failure_feature_deltas_csv": path_to_string(feature_delta_summary_csv),
            "pair_agreement_by_type_csv": path_to_string(pair_agreement_csv),
            "top1_failure_samples_csv": path_to_string(failure_samples_csv),
            "ranker_diagnostics_report_md": path_to_string(report_md),
            "manifest_json": path_to_string(manifest_json),
        },
        "counts": {
            "candidate_rows": int(len(candidates)),
            "episode_rows": int(len(episode_diagnostics)),
            "target_rank_rows": int(len(target_rank_summary)),
            "top1_failure_summary_rows": int(len(top1_failure_summary)),
            "pair_agreement_rows": int(len(pair_agreement)),
        },
    }
    write_report(
        report_md,
        manifest,
        episode_diagnostics,
        target_rank_summary,
        top1_failure_summary,
        group_size_summary,
        feature_delta_summary,
        pair_agreement,
        frozen_failure_slice_summary,
    )
    write_json(manifest_json, manifest)
    return manifest


def load_scored_candidates(
    candidate_csv: Path,
    score_csv: Path,
    target_scope: str,
    split_filters: tuple[str, ...],
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_csv, low_memory=False)
    scores = pd.read_csv(score_csv, low_memory=False)

    score_col = detect_column(scores.columns, ["candidate_score", "endpoint_score", "score", "model_score"])
    score_columns = ["candidate_id", score_col]
    if "gate_score" in scores.columns:
        score_columns.append("gate_score")
    scores = scores[score_columns].drop_duplicates("candidate_id", keep="last").copy()
    scores = scores.rename(columns={score_col: "candidate_score"})

    frame = candidates.merge(scores, on="candidate_id", how="left")
    if "target_scope" in frame.columns:
        frame = frame[frame["target_scope"].astype(str) == target_scope].copy()
    frame = select_splits(frame, split_filters)
    if frame.empty:
        raise ValueError(f"No candidate rows matched target_scope={target_scope!r}, splits={split_filters!r}")

    missing_scores = int(frame["candidate_score"].isna().sum())
    if missing_scores:
        first_missing = frame.loc[frame["candidate_score"].isna(), "candidate_id"].iloc[0]
        raise ValueError(f"Missing candidate_score for {missing_scores} candidates; first={first_missing}")

    normalize_columns(frame)
    return frame.reset_index(drop=True)


def build_episode_diagnostics(candidates: pd.DataFrame, split_filter: str) -> pd.DataFrame:
    frame = candidates.copy()
    frame["_score_sort"] = pd.to_numeric(frame["candidate_score"], errors="coerce").fillna(-np.inf)
    frame["_step_sort"] = pd.to_numeric(frame["candidate_step"], errors="coerce").fillna(-1)
    frame = frame.sort_values(
        ["episode_id", "_score_sort", "_step_sort"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    frame["ranker_rank"] = frame.groupby("episode_id").cumcount() + 1

    rows = [summarize_episode(group, split_filter) for _, group in frame.groupby("episode_id", sort=False)]
    return pd.DataFrame(rows)


def summarize_episode(group: pd.DataFrame, split_filter: str) -> dict[str, Any]:
    group = group.sort_values(["ranker_rank", "candidate_step"], ascending=[True, True])
    top1 = group.iloc[0]
    final_rows = group[group["is_final"]]
    final = final_rows.iloc[0] if not final_rows.empty else None
    success_rows = group[group["success_label"]]
    best_scored_success = best_ranked_row(success_rows)
    nearest = best_ranked_row(group[group["is_nearest_candidate"]])
    first_success = first_success_row(success_rows)
    best_spl_success = best_spl_success_row(success_rows)

    first_success_step = value_or_nan(first_success, "candidate_step")
    last_success_step = safe_max(success_rows["candidate_step"]) if not success_rows.empty else math.nan
    top1_location = location_vs_success_window(
        value_or_nan(top1, "candidate_step"),
        first_success_step,
        last_success_step,
        bool(top1.get("success_label")),
    )
    best_success_score = value_or_nan(best_scored_success, "candidate_score")
    top1_score = value_or_nan(top1, "candidate_score")
    margin_over_best_success = subtract_or_nan(top1_score, best_success_score)

    row: dict[str, Any] = {
        "split_filter": split_filter,
        "experiment_id": top1.get("experiment_id"),
        "dataset": top1.get("dataset"),
        "split": top1.get("split"),
        "protocol_split": top1.get("protocol_split"),
        "target_scope": top1.get("target_scope"),
        "episode_id": top1.get("episode_id"),
        "internal_item_id": top1.get("internal_item_id"),
        "saved_instr_id": top1.get("saved_instr_id"),
        "final_failure_bucket": top1.get("final_failure_bucket"),
        "candidate_count": int(len(group)),
        "success_candidate_count": int(len(success_rows)),
        "failed_candidate_count": int(len(group) - len(success_rows)),
        "trace_available": bool(top1.get("trace_available")),
        "trajectory_step_count": value_or_nan(top1, "trajectory_step_count"),
        "final_success": bool(top1.get("final_success")),
        "oracle_success": bool(top1.get("oracle_success")),
        "should_rerank": bool(top1.get("should_rerank")),
        "gate_score": value_or_nan(top1, "gate_score"),
        "ranker_top1_candidate_id": top1.get("candidate_id"),
        "ranker_top1_step": value_or_nan(top1, "candidate_step"),
        "ranker_top1_score": top1_score,
        "ranker_top1_success": bool(top1.get("success_label")),
        "ranker_top1_is_final": bool(top1.get("is_final")),
        "ranker_top1_is_nearest": bool(top1.get("is_nearest_candidate")),
        "ranker_top1_is_best_spl_success": bool(top1.get("is_best_success_candidate")),
        "ranker_top1_is_route_expanded_without_decision": bool(
            top1.get("is_route_expanded_without_decision")
        ),
        "ranker_top1_is_route_intermediate": bool(top1.get("is_route_intermediate")),
        "ranker_top1_is_revisit": bool(top1.get("is_revisit")),
        "ranker_top1_is_loop_region": bool(top1.get("is_loop_region")),
        "ranker_top1_is_last_k": bool(top1.get("is_last_k")),
        "ranker_top1_path_length_m": value_or_nan(top1, "path_length_m"),
        "ranker_top1_step_frac": value_or_nan(top1, "step_frac"),
        "ranker_top1_stop_prob": value_or_nan(top1, "stop_prob"),
        "ranker_top1_fuse_weight": value_or_nan(top1, "fuse_weight"),
        "ranker_top1_distance_m": value_or_nan(top1, "distance_to_goal_m"),
        "ranker_top1_spl": value_or_nan(top1, "spl_at_candidate"),
        "success_best_rank": value_or_nan(best_scored_success, "ranker_rank"),
        "success_best_score": best_success_score,
        "top1_margin_over_best_success": margin_over_best_success,
        "top1_margin_over_best_success_bin": margin_bin(margin_over_best_success),
        "top1_location_vs_success_window": top1_location,
        "top1_failure_hint": failure_hint(top1, top1_location, margin_over_best_success, success_rows.empty),
    }

    add_target_columns(row, "best_scored_success", best_scored_success, top1_score)
    add_target_columns(row, "nearest", nearest, top1_score)
    add_target_columns(row, "first_success", first_success, top1_score)
    add_target_columns(row, "best_spl_success", best_spl_success, top1_score)
    add_target_columns(row, "final", final, top1_score)
    add_delta_columns(row)
    return row


def add_target_columns(row: dict[str, Any], prefix: str, target: pd.Series | None, top1_score: float) -> None:
    row[f"{prefix}_candidate_id"] = value_or_none(target, "candidate_id")
    row[f"{prefix}_rank"] = value_or_nan(target, "ranker_rank")
    row[f"{prefix}_score"] = value_or_nan(target, "candidate_score")
    row[f"{prefix}_score_gap_to_top1"] = subtract_or_nan(top1_score, row[f"{prefix}_score"])
    row[f"{prefix}_step"] = value_or_nan(target, "candidate_step")
    row[f"{prefix}_path_length_m"] = value_or_nan(target, "path_length_m")
    row[f"{prefix}_step_frac"] = value_or_nan(target, "step_frac")
    row[f"{prefix}_stop_prob"] = value_or_nan(target, "stop_prob")
    row[f"{prefix}_fuse_weight"] = value_or_nan(target, "fuse_weight")
    row[f"{prefix}_distance_m"] = value_or_nan(target, "distance_to_goal_m")
    row[f"{prefix}_spl"] = value_or_nan(target, "spl_at_candidate")
    row[f"{prefix}_success"] = bool(target.get("success_label")) if target is not None else False
    row[f"{prefix}_matches_top1"] = bool(row[f"{prefix}_rank"] == 1) if not is_nan(row[f"{prefix}_rank"]) else False


def add_delta_columns(row: dict[str, Any]) -> None:
    for target in DELTA_TARGETS:
        row[f"top1_step_minus_{target}"] = subtract_or_nan(
            row["ranker_top1_step"],
            row[f"{target}_step"],
        )
        row[f"top1_score_minus_{target}"] = subtract_or_nan(
            row["ranker_top1_score"],
            row[f"{target}_score"],
        )
        row[f"top1_distance_minus_{target}"] = subtract_or_nan(
            row["ranker_top1_distance_m"],
            row[f"{target}_distance_m"],
        )
        row[f"top1_path_length_minus_{target}"] = subtract_or_nan(
            row["ranker_top1_path_length_m"],
            row[f"{target}_path_length_m"],
        )
        row[f"top1_spl_minus_{target}"] = subtract_or_nan(
            row["ranker_top1_spl"],
            row[f"{target}_spl"],
        )


def build_target_rank_summary(episode_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if episode_diagnostics.empty:
        return pd.DataFrame()
    base = episode_diagnostics[episode_diagnostics["should_rerank"]].copy()
    rows: list[dict[str, Any]] = []
    for dataset, group in dataset_groups(base):
        total = len(group)
        for prefix in TARGET_PREFIXES:
            rank_col = f"{prefix}_rank"
            target = group[group[rank_col].notna()].copy()
            rows.append(
                {
                    "split_filter": joined_unique(group["split_filter"]),
                    "dataset": dataset,
                    "target": prefix,
                    "should_rerank_items": int(total),
                    "target_available_items": int(len(target)),
                    "target_available_rate": safe_divide(len(target), total),
                    "target_top1_rate": mean_bool(target[rank_col] <= 1),
                    "target_top3_rate": mean_bool(target[rank_col] <= 3),
                    "target_top5_rate": mean_bool(target[rank_col] <= 5),
                    "mean_rank": safe_mean(target[rank_col]),
                    "median_rank": safe_median(target[rank_col]),
                    "mean_score_gap_to_top1": safe_mean(target[f"{prefix}_score_gap_to_top1"]),
                    "median_score_gap_to_top1": safe_median(target[f"{prefix}_score_gap_to_top1"]),
                    "mean_target_step": safe_mean(target[f"{prefix}_step"]),
                    "mean_target_spl": safe_mean(target[f"{prefix}_spl"]),
                    "mean_target_distance_m": safe_mean(target[f"{prefix}_distance_m"]),
                }
            )
    return pd.DataFrame(rows)


def build_top1_failure_summary(episode_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if episode_diagnostics.empty:
        return pd.DataFrame()
    base = episode_diagnostics[episode_diagnostics["should_rerank"]].copy()
    rows: list[dict[str, Any]] = []
    for dataset, group in dataset_groups(base):
        failed = group[~group["ranker_top1_success"]].copy()
        high_conf_failed = failed["top1_margin_over_best_success"] > 0.2
        rows.append(
            {
                "split_filter": joined_unique(group["split_filter"]),
                "dataset": dataset,
                "should_rerank_items": int(len(group)),
                "top1_success_items": int(group["ranker_top1_success"].sum()),
                "top1_success_rate": mean_bool(group["ranker_top1_success"]),
                "top1_failed_items": int(len(failed)),
                "top1_failed_rate": safe_divide(len(failed), len(group)),
                "failed_top1_is_final_rate": mean_bool(failed["ranker_top1_is_final"]),
                "failed_top1_is_route_expanded_rate": mean_bool(
                    failed["ranker_top1_is_route_expanded_without_decision"]
                ),
                "failed_top1_is_route_intermediate_rate": mean_bool(
                    failed["ranker_top1_is_route_intermediate"]
                ),
                "failed_top1_is_revisit_rate": mean_bool(failed["ranker_top1_is_revisit"]),
                "failed_top1_is_loop_region_rate": mean_bool(failed["ranker_top1_is_loop_region"]),
                "failed_top1_is_last_k_rate": mean_bool(failed["ranker_top1_is_last_k"]),
                "failed_before_first_success_rate": mean_bool(
                    failed["top1_location_vs_success_window"] == "before_first_success"
                ),
                "failed_inside_success_span_rate": mean_bool(
                    failed["top1_location_vs_success_window"] == "inside_success_span_failed"
                ),
                "failed_after_last_success_rate": mean_bool(
                    failed["top1_location_vs_success_window"] == "after_last_success"
                ),
                "high_conf_failed_rate_margin_gt_0_2": mean_bool(high_conf_failed),
                "mean_failed_margin_over_best_success": safe_mean(
                    failed["top1_margin_over_best_success"]
                ),
                "median_failed_margin_over_best_success": safe_median(
                    failed["top1_margin_over_best_success"]
                ),
                "best_spl_success_top1_rate": mean_bool(group["best_spl_success_rank"] <= 1),
                "best_spl_success_top3_rate": mean_bool(group["best_spl_success_rank"] <= 3),
                "median_best_spl_success_rank": safe_median(group["best_spl_success_rank"]),
                "mean_top1_step_minus_best_spl_success": safe_mean(
                    failed["top1_step_minus_best_spl_success"]
                ),
                "mean_top1_distance_minus_best_spl_success": safe_mean(
                    failed["top1_distance_minus_best_spl_success"]
                ),
                "mean_top1_path_length_minus_best_spl_success": safe_mean(
                    failed["top1_path_length_minus_best_spl_success"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_group_size_summary(episode_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if episode_diagnostics.empty:
        return pd.DataFrame()
    base = episode_diagnostics[episode_diagnostics["should_rerank"]].copy()
    if base.empty:
        return pd.DataFrame()
    base["candidate_count_bin"] = pd.cut(
        base["candidate_count"],
        bins=[0, 5, 10, 20, 40, math.inf],
        labels=["1-5", "6-10", "11-20", "21-40", ">40"],
        include_lowest=True,
    )
    base["success_candidate_count_bin"] = pd.cut(
        base["success_candidate_count"],
        bins=[0, 1, 3, 5, 10, math.inf],
        labels=["1", "2-3", "4-5", "6-10", ">10"],
        include_lowest=True,
    )

    rows: list[dict[str, Any]] = []
    for bin_column in ["candidate_count_bin", "success_candidate_count_bin"]:
        for dataset, dataset_group in dataset_groups(base):
            for bin_value, group in dataset_group.groupby(bin_column, observed=True):
                rows.append(
                    {
                        "split_filter": joined_unique(group["split_filter"]),
                        "dataset": dataset,
                        "bin_type": bin_column,
                        "bin": str(bin_value),
                        "items": int(len(group)),
                        "top1_success_rate": mean_bool(group["ranker_top1_success"]),
                        "mean_success_best_rank": safe_mean(group["success_best_rank"]),
                        "best_spl_success_top3_rate": mean_bool(group["best_spl_success_rank"] <= 3),
                        "mean_candidate_count": safe_mean(group["candidate_count"]),
                        "mean_success_candidate_count": safe_mean(group["success_candidate_count"]),
                    }
                )
    return pd.DataFrame(rows)


def build_feature_delta_summary(episode_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if episode_diagnostics.empty:
        return pd.DataFrame()
    base = episode_diagnostics[
        episode_diagnostics["should_rerank"] & ~episode_diagnostics["ranker_top1_success"]
    ].copy()
    rows: list[dict[str, Any]] = []
    for dataset, group in dataset_groups(base):
        for target in DELTA_TARGETS:
            for feature_name, source_column in DELTA_FEATURES:
                top_col = f"ranker_top1_{feature_name}" if feature_name != "score" else "ranker_top1_score"
                target_col = (
                    f"{target}_{feature_name}" if feature_name != "score" else f"{target}_score"
                )
                if top_col not in group.columns or target_col not in group.columns:
                    continue
                delta = pd.to_numeric(group[top_col], errors="coerce") - pd.to_numeric(
                    group[target_col], errors="coerce"
                )
                delta = delta.dropna()
                rows.append(
                    {
                        "split_filter": joined_unique(group["split_filter"]),
                        "dataset": dataset,
                        "target": target,
                        "feature": source_column,
                        "items": int(len(delta)),
                        "mean_delta_top1_minus_target": safe_mean(delta),
                        "median_delta_top1_minus_target": safe_median(delta),
                        "p25_delta": safe_quantile(delta, 0.25),
                        "p75_delta": safe_quantile(delta, 0.75),
                        "top1_greater_rate": mean_bool(delta > EPS),
                        "top1_lower_rate": mean_bool(delta < -EPS),
                    }
                )
    return pd.DataFrame(rows)


def build_pair_agreement(
    pair_csv: Path | None,
    scored_candidates: pd.DataFrame,
    target_scope: str,
    split_filters: tuple[str, ...],
) -> pd.DataFrame:
    if pair_csv is None or not pair_csv.exists():
        return pd.DataFrame()
    pairs = pd.read_csv(pair_csv, low_memory=False)
    if "target_scope" in pairs.columns:
        pairs = pairs[pairs["target_scope"].astype(str) == target_scope].copy()
    pairs = select_splits(pairs, split_filters)
    if pairs.empty:
        return pd.DataFrame()

    score_map = (
        scored_candidates[["candidate_id", "candidate_score"]]
        .drop_duplicates("candidate_id")
        .set_index("candidate_id")["candidate_score"]
    )
    pairs["chosen_score"] = pairs["chosen_candidate_id"].map(score_map)
    pairs["rejected_score"] = pairs["rejected_candidate_id"].map(score_map)
    pairs = pairs.dropna(subset=["chosen_score", "rejected_score"]).copy()
    pairs["score_margin"] = pairs["chosen_score"] - pairs["rejected_score"]
    pairs["accuracy"] = pairs["score_margin"] > EPS
    pairs["tie"] = pairs["score_margin"].abs() <= EPS

    rows: list[dict[str, Any]] = []
    for dataset, group in dataset_groups(pairs):
        for pair_type, pair_group in group.groupby("pair_type", sort=True):
            rows.append(pair_summary_row(dataset, pair_type, pair_group))
        rows.append(pair_summary_row(dataset, "all", group))
    return pd.DataFrame(rows)


def pair_summary_row(dataset: str, pair_type: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "split_filter": joined_unique(group["protocol_split"] if "protocol_split" in group else group["split"]),
        "dataset": dataset,
        "pair_type": pair_type,
        "pairs": int(len(group)),
        "accuracy": mean_bool(group["accuracy"]),
        "strict_accuracy": mean_bool(group["score_margin"] > EPS),
        "tie_rate": mean_bool(group["tie"]),
        "mean_score_margin": safe_mean(group["score_margin"]),
        "median_score_margin": safe_median(group["score_margin"]),
    }


def build_failure_samples(episode_diagnostics: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if episode_diagnostics.empty or sample_size <= 0:
        return pd.DataFrame()
    base = episode_diagnostics[
        episode_diagnostics["should_rerank"] & ~episode_diagnostics["ranker_top1_success"]
    ].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.sort_values(
        ["top1_margin_over_best_success", "dataset", "episode_id"],
        ascending=[False, True, True],
        na_position="last",
    )
    columns = [
        "dataset",
        "split",
        "protocol_split",
        "episode_id",
        "internal_item_id",
        "saved_instr_id",
        "final_failure_bucket",
        "frozen_failure_slice",
        "frozen_selection_reason",
        "candidate_count",
        "success_candidate_count",
        "ranker_top1_candidate_id",
        "ranker_top1_step",
        "ranker_top1_score",
        "ranker_top1_distance_m",
        "top1_margin_over_best_success",
        "top1_margin_over_best_success_bin",
        "top1_location_vs_success_window",
        "top1_failure_hint",
        "best_scored_success_candidate_id",
        "best_scored_success_rank",
        "best_scored_success_score",
        "best_spl_success_candidate_id",
        "best_spl_success_rank",
        "best_spl_success_score",
        "best_spl_success_spl",
        "first_success_candidate_id",
        "first_success_rank",
        "nearest_candidate_id",
        "nearest_rank",
        "final_candidate_id",
        "final_rank",
    ]
    available_columns = [column for column in columns if column in base.columns]
    return base[available_columns].head(sample_size).reset_index(drop=True)


def load_selected_items(selected_items_csv: Path | None) -> pd.DataFrame | None:
    if selected_items_csv is None or not selected_items_csv.exists():
        return None
    selected = pd.read_csv(selected_items_csv, low_memory=False)
    keep = [
        "episode_id",
        "failure_slice",
        "selection_reason",
        "gate_passed",
        "selected_changed",
        "selected_success",
        "recovered",
        "harmed",
    ]
    keep = [column for column in keep if column in selected.columns]
    selected = selected[keep].drop_duplicates("episode_id").copy()
    selected = selected.rename(
        columns={
            "failure_slice": "frozen_failure_slice",
            "selection_reason": "frozen_selection_reason",
            "gate_passed": "frozen_gate_passed",
            "selected_changed": "frozen_selected_changed",
            "selected_success": "frozen_selected_success",
            "recovered": "frozen_recovered",
            "harmed": "frozen_harmed",
        }
    )
    for column in [
        "frozen_gate_passed",
        "frozen_selected_changed",
        "frozen_selected_success",
        "frozen_recovered",
        "frozen_harmed",
    ]:
        if column in selected.columns:
            selected[column] = to_bool_series(selected[column])
    return selected


def write_report(
    path: Path,
    manifest: dict[str, Any],
    episode_diagnostics: pd.DataFrame,
    target_rank_summary: pd.DataFrame,
    top1_failure_summary: pd.DataFrame,
    group_size_summary: pd.DataFrame,
    feature_delta_summary: pd.DataFrame,
    pair_agreement: pd.DataFrame,
    frozen_failure_slice_summary: pd.DataFrame | None,
) -> None:
    recoverable = episode_diagnostics[episode_diagnostics["should_rerank"]].copy()
    all_failure = top1_failure_summary[top1_failure_summary["dataset"] == "ALL"]
    all_target = target_rank_summary[target_rank_summary["dataset"] == "ALL"]
    all_pairs = pair_agreement[pair_agreement["dataset"] == "ALL"] if not pair_agreement.empty else pd.DataFrame()

    lines = [
        "# Endpoint Ranker Top1 Diagnostics",
        "",
        "This report is generated by `scripts/analysis/diagnose_endpoint_ranker_top1.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- split_filters: `{','.join(manifest['split_filters'])}`",
        "- model training: no",
        "- diagnostic target: group-level top1 behavior of CE success ranker",
        "",
        "## Key Findings",
        "",
    ]

    if not recoverable.empty and not all_failure.empty:
        failure_row = all_failure.iloc[0]
        lines.extend(
            [
                f"- Recoverable `should_rerank` episodes: `{len(recoverable)}`.",
                (
                    "- Ranker top1 success on recoverable episodes: "
                    f"`{pct(failure_row['top1_success_rate'])}` "
                    f"({int(failure_row['top1_success_items'])}/{int(failure_row['should_rerank_items'])})."
                ),
                (
                    "- Failed top1 episodes: "
                    f"`{int(failure_row['top1_failed_items'])}`, "
                    f"mean failed margin over best scored success "
                    f"`{fmt(failure_row['mean_failed_margin_over_best_success'])}`."
                ),
            ]
        )
    if not all_target.empty:
        target_lookup = {row["target"]: row for _, row in all_target.iterrows()}
        for target in ["best_scored_success", "best_spl_success", "first_success", "nearest"]:
            row = target_lookup.get(target)
            if row is None:
                continue
            lines.append(
                f"- `{target}` top1/top3/median-rank: "
                f"`{pct(row['target_top1_rate'])}` / "
                f"`{pct(row['target_top3_rate'])}` / "
                f"`{fmt(row['median_rank'])}`."
            )
    if not all_pairs.empty:
        pair_lookup = {row["pair_type"]: row for _, row in all_pairs.iterrows()}
        for pair_type in ["success_gt_fail", "better_spl_success_gt_lower_spl_success"]:
            row = pair_lookup.get(pair_type)
            if row is None:
                continue
            lines.append(
                f"- Pair agreement `{pair_type}`: `{pct(row['accuracy'])}` "
                f"over `{int(row['pairs'])}` pairs, mean margin `{fmt(row['mean_score_margin'])}`."
            )

    lines.extend(
        [
            "",
            "## Top1 Failure Summary",
            "",
            markdown_table(
                [
                    "dataset",
                    "should",
                    "top1_success",
                    "top1_failed",
                    "failed_final",
                    "failed_after",
                    "high_conf_fail",
                    "median_best_spl_rank",
                ],
                [
                    [
                        row["dataset"],
                        int(row["should_rerank_items"]),
                        pct(row["top1_success_rate"]),
                        int(row["top1_failed_items"]),
                        pct(row["failed_top1_is_final_rate"]),
                        pct(row["failed_after_last_success_rate"]),
                        pct(row["high_conf_failed_rate_margin_gt_0_2"]),
                        fmt(row["median_best_spl_success_rank"]),
                    ]
                    for _, row in top1_failure_summary.iterrows()
                ],
            ),
            "",
            "## Target Rank Summary",
            "",
            markdown_table(
                [
                    "dataset",
                    "target",
                    "available",
                    "top1",
                    "top3",
                    "top5",
                    "mean_rank",
                    "median_rank",
                    "mean_gap",
                ],
                [
                    [
                        row["dataset"],
                        row["target"],
                        int(row["target_available_items"]),
                        pct(row["target_top1_rate"]),
                        pct(row["target_top3_rate"]),
                        pct(row["target_top5_rate"]),
                        fmt(row["mean_rank"]),
                        fmt(row["median_rank"]),
                        fmt(row["mean_score_gap_to_top1"]),
                    ]
                    for _, row in target_rank_summary.iterrows()
                    if row["dataset"] == "ALL"
                ],
            ),
            "",
            "## Pair Agreement",
            "",
            markdown_table(
                ["dataset", "pair_type", "pairs", "accuracy", "mean_margin"],
                [
                    [
                        row["dataset"],
                        row["pair_type"],
                        int(row["pairs"]),
                        pct(row["accuracy"]),
                        fmt(row["mean_score_margin"]),
                    ]
                    for _, row in pair_agreement.iterrows()
                    if row["dataset"] == "ALL"
                ],
            ),
            "",
            "## Group Size Sensitivity",
            "",
            markdown_table(
                ["bin_type", "bin", "items", "top1_success", "mean_success_rank", "best_spl_top3"],
                [
                    [
                        row["bin_type"],
                        row["bin"],
                        int(row["items"]),
                        pct(row["top1_success_rate"]),
                        fmt(row["mean_success_best_rank"]),
                        pct(row["best_spl_success_top3_rate"]),
                    ]
                    for _, row in group_size_summary.iterrows()
                    if row["dataset"] == "ALL"
                ],
            ),
            "",
            "## Top Failed Deltas",
            "",
            markdown_table(
                ["target", "feature", "items", "mean_delta", "median_delta", "top1_greater"],
                [
                    [
                        row["target"],
                        row["feature"],
                        int(row["items"]),
                        fmt(row["mean_delta_top1_minus_target"]),
                        fmt(row["median_delta_top1_minus_target"]),
                        pct(row["top1_greater_rate"]),
                    ]
                    for _, row in feature_delta_summary.iterrows()
                    if row["dataset"] == "ALL"
                    and row["target"] in {"best_spl_success", "first_success"}
                    and row["feature"] in {"candidate_step", "path_length_m", "distance_to_goal_m", "candidate_score"}
                ],
            ),
        ]
    )

    if frozen_failure_slice_summary is not None and not frozen_failure_slice_summary.empty:
        lines.extend(
            [
                "",
                "## Frozen Slice Reference",
                "",
                markdown_table(
                    ["slice", "items", "rate", "changed"],
                    [
                        [
                            row["failure_slice"],
                            int(row["items"]),
                            pct(row["rate"]),
                            int(row["changed_items"]) if "changed_items" in row else "",
                        ]
                        for _, row in frozen_failure_slice_summary.iterrows()
                    ],
                ),
            ]
        )

    lines.extend(["", "## Files", ""])
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_ranked_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values(["ranker_rank", "candidate_step"], ascending=[True, True]).iloc[0]


def first_success_row(success_rows: pd.DataFrame) -> pd.Series | None:
    if success_rows.empty:
        return None
    first_step = success_rows["candidate_step"].min()
    return best_ranked_row(success_rows[success_rows["candidate_step"] == first_step])


def best_spl_success_row(success_rows: pd.DataFrame) -> pd.Series | None:
    if success_rows.empty:
        return None
    best_spl = success_rows["spl_at_candidate"].max()
    return best_ranked_row(success_rows[success_rows["spl_at_candidate"] >= best_spl - EPS])


def failure_hint(
    top1: pd.Series,
    top1_location: str,
    margin_over_best_success: float,
    no_success: bool,
) -> str:
    if bool(top1.get("success_label")):
        return "top1_success"
    if no_success:
        return "no_success_candidate"
    if bool(top1.get("is_final")):
        return "top1_final_failed"
    if not is_nan(margin_over_best_success) and margin_over_best_success > 0.2:
        return "high_conf_failed_candidate"
    if top1_location == "after_last_success":
        return "overshoots_success_window"
    if top1_location == "before_first_success":
        return "stops_before_success_window"
    if bool(top1.get("is_route_expanded_without_decision")):
        return "route_expanded_candidate"
    if bool(top1.get("is_revisit")) or bool(top1.get("is_loop_region")):
        return "loop_or_revisit_candidate"
    return "low_margin_failed_candidate"


def location_vs_success_window(
    top1_step: float,
    first_success_step: float,
    last_success_step: float,
    top1_success: bool,
) -> str:
    if top1_success:
        return "on_success"
    if is_nan(first_success_step) or is_nan(last_success_step):
        return "no_success"
    if top1_step < first_success_step:
        return "before_first_success"
    if top1_step > last_success_step:
        return "after_last_success"
    return "inside_success_span_failed"


def margin_bin(value: float) -> str:
    if is_nan(value):
        return "nan"
    if value <= 0.02:
        return "<=0.02"
    if value <= 0.05:
        return "0.02-0.05"
    if value <= 0.10:
        return "0.05-0.10"
    if value <= 0.20:
        return "0.10-0.20"
    return ">0.20"


def normalize_columns(frame: pd.DataFrame) -> None:
    for column in BOOL_COLUMNS:
        if column in frame.columns:
            frame[column] = to_bool_series(frame[column])
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "gate_score" not in frame.columns:
        frame["gate_score"] = math.nan


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def select_splits(frame: pd.DataFrame, split_filters: tuple[str, ...]) -> pd.DataFrame:
    if not split_filters or "all" in split_filters:
        return frame.copy()
    mask = pd.Series(False, index=frame.index)
    if "split" in frame.columns:
        mask |= frame["split"].astype(str).isin(split_filters)
    if "protocol_split" in frame.columns:
        mask |= frame["protocol_split"].astype(str).isin(split_filters)
    return frame[mask].copy()


def select_split(frame: pd.DataFrame, split_filter: str) -> pd.DataFrame:
    return select_splits(frame, (split_filter,))


def dataset_groups(frame: pd.DataFrame):
    if frame.empty:
        return
    yield "ALL", frame
    for dataset, group in frame.groupby("dataset", sort=True):
        yield str(dataset), group


def read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def detect_column(columns: pd.Index, aliases: list[str]) -> str:
    for alias in aliases:
        if alias in columns:
            return alias
    raise ValueError(f"Could not detect score column. Available columns: {', '.join(columns)}")


def value_or_nan(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return math.nan
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return math.nan
    return value if not math.isnan(value) else math.nan


def value_or_none(row: pd.Series | None, column: str) -> Any:
    if row is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def subtract_or_nan(left: Any, right: Any) -> float:
    try:
        left = float(left)
        right = float(right)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return left - right


def safe_max(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else math.nan


def safe_median(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.median()) if not series.empty else math.nan


def safe_quantile(values: Any, q: float) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.quantile(q)) if not series.empty else math.nan


def safe_divide(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    if denominator == 0 or math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    return numerator / denominator


def mean_bool(values: Any) -> float:
    series = pd.Series(values)
    if series.empty:
        return math.nan
    return float(series.fillna(False).astype(bool).mean())


def is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def joined_unique(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique()]
    return ",".join(sorted(unique))


def parse_string_list(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return tuple()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def default_output_dir(experiment_dir: Path | None, endpoint_learning_dir: Path) -> Path:
    if experiment_dir is not None:
        return (experiment_dir / DEFAULT_REPORTS_SUBDIR).resolve()
    if endpoint_learning_dir.name == DEFAULT_ENDPOINT_LEARNING_DIR:
        return (endpoint_learning_dir.parent / DEFAULT_REPORTS_SUBDIR).resolve()
    return (endpoint_learning_dir / DEFAULT_OUTPUT_NAME).resolve()


def resolve_path(raw: str | None, default: Path) -> Path:
    return Path(raw).resolve() if raw else default.resolve()


def resolve_optional_path(raw: str | None, default: Path) -> Path | None:
    if raw:
        return Path(raw).resolve()
    return default.resolve() if default.exists() else None


def path_to_string(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = ["" if value is None else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{100.0 * value:.2f}"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
