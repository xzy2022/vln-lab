#!/usr/bin/env python3
"""Train a lightweight endpoint rerank gate baseline over fixed candidate groups."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_endpoint_reranker as reranker_eval  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_gate_baseline.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_OUTPUT_NAME = "gate_baseline"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_DEV_SPLIT = "dev"
DEFAULT_THRESHOLD_GRID = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
DEFAULT_TAUS = (0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT_RANDOM_STATE = 17
EPS = 1e-12

EPISODE_ID_COLUMNS = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "internal_item_id",
    "saved_instr_id",
]

FORBIDDEN_FEATURE_COLUMNS = {
    "success_label",
    "spl_at_candidate",
    "distance_to_goal_m",
    "reward",
    "is_best_success_candidate",
    "is_nearest_candidate",
    "final_success",
    "oracle_success",
    "nearest_endpoint_success",
    "should_rerank",
    "final_distance_m",
    "best_distance_m",
    "final_spl",
    "nearest_endpoint_spl",
}

NUMERIC_CANDIDATE_COLUMNS = [
    "step_frac",
    "path_length_m",
    "stop_prob",
    "stop_margin_prob",
    "selected_prob",
    "top1_top2_margin",
    "moe_router_entropy",
    "fuse_weight",
]

BOOLEAN_CANDIDATE_COLUMNS = [
    "has_decision_trace",
    "is_route_intermediate",
    "is_route_expanded_without_decision",
    "is_final",
    "is_revisit",
    "is_loop_region",
    "is_last_k",
]

FEATURE_COLUMNS = [
    "trajectory_step_count",
    "decision_trace_step_count",
    "candidate_count",
    "decision_trace_candidate_rate",
    "route_intermediate_candidate_rate",
    "route_expanded_without_decision_candidate_rate",
    "trace_missing_candidate_rate",
    "revisit_count",
    "revisit_rate",
    "loop_region_count",
    "loop_region_rate",
    "last_k_count",
    "last_k_rate",
    "final_step_frac",
    "final_path_length_m",
    "final_stop_prob",
    "max_stop_prob",
    "mean_stop_prob",
    "std_stop_prob",
    "last_k_max_stop_prob",
    "last_k_mean_stop_prob",
    "max_stop_minus_final_stop",
    "argmax_stop_step_frac",
    "argmax_stop_gap_to_final",
    "final_stop_rank_frac",
    "final_stop_margin_prob",
    "max_stop_margin_prob",
    "mean_stop_margin_prob",
    "std_stop_margin_prob",
    "last_k_max_stop_margin_prob",
    "max_stop_margin_minus_final",
    "final_selected_prob",
    "max_selected_prob",
    "mean_selected_prob",
    "std_selected_prob",
    "final_top1_top2_margin",
    "max_top1_top2_margin",
    "mean_top1_top2_margin",
    "final_router_entropy",
    "mean_router_entropy",
    "max_router_entropy",
    "final_fuse_weight",
    "mean_fuse_weight",
    "max_fuse_weight",
    "max_path_length_m",
    "mean_path_length_m",
]

GATE_FEATURE_FIELDNAMES = EPISODE_ID_COLUMNS + [
    "should_rerank",
    "final_success",
    "oracle_success",
    "final_failure_bucket",
] + FEATURE_COLUMNS + ["gate_score"]

EPISODE_SCORE_FIELDNAMES = EPISODE_ID_COLUMNS + [
    "should_rerank",
    "final_success",
    "oracle_success",
    "final_failure_bucket",
    "gate_score",
]

GATE_SCORE_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "candidate_id",
    "candidate_step",
    "candidate_score",
    "gate_score",
    "should_rerank",
]

THRESHOLD_FIELDNAMES = [
    "split_filter",
    "threshold",
    "items",
    "positives",
    "base_rate",
    "pass_rate",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "true_negative_rate",
    "final_success_pass_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "selection_tag",
]

SUMMARY_FIELDNAMES = [
    "split_filter",
    "items",
    "positives",
    "base_rate",
    "roc_auc",
    "average_precision",
    "brier",
    "log_loss",
    "mean_gate_score",
]

IMPORTANCE_FIELDNAMES = [
    "feature",
    "coefficient",
    "abs_coefficient",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sklearn gate-only baseline for endpoint reranking.",
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
        "--episode-csv",
        default=None,
        help="Episode groups CSV. Defaults to <endpoint-learning-dir>/candidate_groups/episode_groups.csv.",
    )
    parser.add_argument(
        "--candidate-csv",
        default=None,
        help="Endpoint candidates CSV. Defaults to <endpoint-learning-dir>/candidate_groups/endpoint_candidates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <endpoint-learning-dir>/gate_baseline.",
    )
    parser.add_argument(
        "--target-scope",
        default=DEFAULT_TARGET_SCOPE,
        help="Target scope to train. Defaults to official.",
    )
    parser.add_argument(
        "--train-split",
        default=DEFAULT_TRAIN_SPLIT,
        help="Protocol split used for training. Defaults to train.",
    )
    parser.add_argument(
        "--dev-split",
        default=DEFAULT_DEV_SPLIT,
        help="Protocol split used for threshold selection. Defaults to dev.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(format_float(value) for value in DEFAULT_THRESHOLD_GRID),
        help="Comma-separated gate thresholds to evaluate on dev.",
    )
    parser.add_argument(
        "--taus",
        default=",".join(format_float(value) for value in DEFAULT_TAUS),
        help="Comma-separated taus for optional reranker evaluator.",
    )
    parser.add_argument(
        "--candidate-score-column",
        default="stop_prob",
        help="Candidate column copied to candidate_score in gate_scores.csv.",
    )
    parser.add_argument(
        "--missing-candidate-score",
        type=float,
        default=0.0,
        help="Fallback candidate_score when the selected candidate score column is missing.",
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum LogisticRegression iterations.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random state for deterministic sklearn behavior.",
    )
    parser.add_argument(
        "--min-recall-for-conservative",
        type=float,
        default=0.25,
        help="Minimum dev recall when selecting conservative threshold candidate.",
    )
    parser.add_argument(
        "--skip-reranker-eval",
        action="store_true",
        help="Only train/export gate scores; do not call eval_endpoint_reranker.py.",
    )
    parser.add_argument(
        "--include-test-summary",
        action="store_true",
        help="Also summarize test/val_unseen gate metrics. Keep disabled while tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    endpoint_learning_dir = resolve_endpoint_learning_dir(experiment_dir, args.endpoint_learning_dir)
    episode_csv = (
        Path(args.episode_csv).resolve()
        if args.episode_csv
        else endpoint_learning_dir / "candidate_groups" / "episode_groups.csv"
    )
    candidate_csv = (
        Path(args.candidate_csv).resolve()
        if args.candidate_csv
        else endpoint_learning_dir / "candidate_groups" / "endpoint_candidates.csv"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else endpoint_learning_dir / DEFAULT_OUTPUT_NAME
    )

    manifest = train_endpoint_gate_baseline(
        episode_csv=episode_csv,
        candidate_csv=candidate_csv,
        output_dir=output_dir,
        target_scope=args.target_scope,
        train_split=args.train_split,
        dev_split=args.dev_split,
        thresholds=tuple(parse_float_list(args.thresholds)),
        taus=tuple(parse_float_list(args.taus)),
        candidate_score_column=args.candidate_score_column,
        missing_candidate_score=args.missing_candidate_score,
        logistic_c=args.logistic_c,
        max_iter=args.max_iter,
        random_state=args.random_state,
        min_recall_for_conservative=args.min_recall_for_conservative,
        run_reranker_eval=not args.skip_reranker_eval,
        include_test_summary=args.include_test_summary,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def train_endpoint_gate_baseline(
    episode_csv: Path,
    candidate_csv: Path,
    output_dir: Path,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    dev_split: str = DEFAULT_DEV_SPLIT,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    candidate_score_column: str = "stop_prob",
    missing_candidate_score: float = 0.0,
    logistic_c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_recall_for_conservative: float = 0.25,
    run_reranker_eval: bool = True,
    include_test_summary: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episode_csv)
    candidates = pd.read_csv(candidate_csv)
    episodes = episodes[episodes["target_scope"] == target_scope].copy()
    candidates = candidates[candidates["target_scope"] == target_scope].copy()
    if episodes.empty:
        raise ValueError(f"No episodes found for target_scope={target_scope!r}")
    if candidates.empty:
        raise ValueError(f"No candidates found for target_scope={target_scope!r}")

    feature_frame = build_gate_feature_frame(episodes, candidates)
    validate_features(feature_frame)

    train_mask = feature_frame["protocol_split"] == train_split
    dev_mask = feature_frame["protocol_split"] == dev_split
    if not train_mask.any():
        raise ValueError(f"No training rows with protocol_split={train_split!r}")
    if not dev_mask.any():
        raise ValueError(f"No dev rows with protocol_split={dev_split!r}")

    y_train = feature_frame.loc[train_mask, "should_rerank"].astype(int).to_numpy()
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training split must contain both positive and negative should_rerank labels")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=logistic_c,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(feature_frame.loc[train_mask, FEATURE_COLUMNS], y_train)

    feature_frame["gate_score"] = model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]

    threshold_rows = build_threshold_rows(
        feature_frame=feature_frame,
        split_filter=dev_split,
        thresholds=thresholds,
        min_recall_for_conservative=min_recall_for_conservative,
    )
    summary_splits = (train_split, dev_split)
    if include_test_summary:
        summary_splits = (train_split, dev_split, "test", "val_unseen")
    summary_rows = build_summary_rows(feature_frame, split_filters=summary_splits)
    importance_rows = build_importance_rows(model)
    recommended_threshold = recommended_threshold_from_rows(threshold_rows)

    gate_features_csv = output_dir / "gate_features.csv"
    episode_scores_csv = output_dir / "gate_episode_scores.csv"
    gate_scores_csv = output_dir / "gate_scores.csv"
    threshold_csv = output_dir / "gate_threshold_candidates.csv"
    summary_csv = output_dir / "gate_baseline_summary.csv"
    importance_csv = output_dir / "gate_feature_importance.csv"
    model_joblib = output_dir / "gate_model.joblib"
    model_json = output_dir / "gate_model.json"
    report_md = output_dir / "gate_baseline_report.md"
    manifest_path = output_dir / "manifest.json"

    write_csv(gate_features_csv, feature_frame, GATE_FEATURE_FIELDNAMES)
    write_csv(episode_scores_csv, feature_frame, EPISODE_SCORE_FIELDNAMES)
    write_gate_scores_csv(
        path=gate_scores_csv,
        candidates=candidates,
        episode_scores=feature_frame,
        candidate_score_column=candidate_score_column,
        missing_candidate_score=missing_candidate_score,
    )
    pd.DataFrame(threshold_rows, columns=THRESHOLD_FIELDNAMES).to_csv(threshold_csv, index=False)
    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDNAMES).to_csv(summary_csv, index=False)
    pd.DataFrame(importance_rows, columns=IMPORTANCE_FIELDNAMES).to_csv(importance_csv, index=False)
    joblib.dump(model, model_joblib)

    eval_manifest: dict[str, Any] | None = None
    if run_reranker_eval:
        eval_manifest = reranker_eval.evaluate_endpoint_reranker(
            candidate_csv=candidate_csv,
            episode_csv=episode_csv,
            score_csv=gate_scores_csv,
            output_dir=output_dir / "eval_protocol",
            target_scope=target_scope,
            split_filters=(dev_split,),
            gate_thresholds=thresholds,
            taus=taus,
            allow_change_final_values=(True,),
            candidate_score_column="candidate_score",
            gate_score_column="gate_score",
            default_gate_score=0.0,
        )

    model_payload = build_model_payload(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        target_scope=target_scope,
        train_split=train_split,
        dev_split=dev_split,
        thresholds=thresholds,
        recommended_threshold=recommended_threshold,
        logistic_c=logistic_c,
        max_iter=max_iter,
        random_state=random_state,
    )
    write_json(model_json, model_payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": target_scope,
        "train_split": train_split,
        "dev_split": dev_split,
        "thresholds": list(thresholds),
        "taus": list(taus),
        "recommended_threshold": recommended_threshold,
        "include_test_summary": include_test_summary,
        "candidate_score_column": candidate_score_column,
        "missing_candidate_score": missing_candidate_score,
        "model": {
            "type": "sklearn.pipeline.Pipeline",
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "logistic_c": logistic_c,
            "max_iter": max_iter,
            "random_state": random_state,
        },
        "counts": {
            "episodes": int(len(feature_frame)),
            "train_episodes": int(train_mask.sum()),
            "dev_episodes": int(dev_mask.sum()),
            "candidates": int(len(candidates)),
            "features": len(FEATURE_COLUMNS),
        },
        "files": {
            "episode_csv": path_to_string(episode_csv),
            "candidate_csv": path_to_string(candidate_csv),
            "gate_features_csv": path_to_string(gate_features_csv),
            "gate_episode_scores_csv": path_to_string(episode_scores_csv),
            "gate_scores_csv": path_to_string(gate_scores_csv),
            "gate_threshold_candidates_csv": path_to_string(threshold_csv),
            "gate_baseline_summary_csv": path_to_string(summary_csv),
            "gate_feature_importance_csv": path_to_string(importance_csv),
            "gate_model_joblib": path_to_string(model_joblib),
            "gate_model_json": path_to_string(model_json),
            "gate_baseline_report_md": path_to_string(report_md),
        },
        "reranker_eval": eval_manifest,
    }
    write_json(manifest_path, manifest)
    write_report(
        path=report_md,
        manifest=manifest,
        summary_rows=summary_rows,
        threshold_rows=threshold_rows,
        importance_rows=importance_rows,
        eval_manifest=eval_manifest,
    )
    return manifest


def build_gate_feature_frame(episodes: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    for column in NUMERIC_CANDIDATE_COLUMNS:
        candidates[column] = pd.to_numeric(candidates.get(column), errors="coerce")
    for column in BOOLEAN_CANDIDATE_COLUMNS:
        candidates[column] = to_bool_series(candidates.get(column, pd.Series(False, index=candidates.index)))

    episode_by_id = {str(row["episode_id"]): row for _, row in episodes.iterrows()}
    rows: list[dict[str, Any]] = []
    for episode_id, candidate_rows in candidates.groupby("episode_id", sort=False):
        episode = episode_by_id.get(str(episode_id))
        if episode is None:
            continue
        row = build_one_episode_features(episode, candidate_rows)
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        raise ValueError("No feature rows were built")
    feature_frame = feature_frame.sort_values(["split", "protocol_split", "episode_id"]).reset_index(drop=True)
    for column in FEATURE_COLUMNS:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    feature_frame["should_rerank"] = to_bool_series(feature_frame["should_rerank"]).astype(int)
    feature_frame["final_success"] = to_bool_series(feature_frame["final_success"]).astype(int)
    feature_frame["oracle_success"] = to_bool_series(feature_frame["oracle_success"]).astype(int)
    return feature_frame


def build_one_episode_features(episode: pd.Series, candidates: pd.DataFrame) -> dict[str, Any]:
    final_rows = candidates[candidates["is_final"]]
    final = final_rows.iloc[0] if not final_rows.empty else candidates.iloc[-1]
    last_k = candidates[candidates["is_last_k"]]
    stop_prob = candidates["stop_prob"]
    stop_margin = candidates["stop_margin_prob"]
    selected_prob = candidates["selected_prob"]
    top_margin = candidates["top1_top2_margin"]
    router_entropy = candidates["moe_router_entropy"]
    fuse_weight = candidates["fuse_weight"]
    path_length = candidates["path_length_m"]

    candidate_count = int(len(candidates))
    decision_trace_count = int(candidates["has_decision_trace"].sum())
    route_intermediate_count = int(candidates["is_route_intermediate"].sum())
    route_expanded_count = int(candidates["is_route_expanded_without_decision"].sum())
    revisit_count = int(candidates["is_revisit"].sum())
    loop_region_count = int(candidates["is_loop_region"].sum())
    last_k_count = int(candidates["is_last_k"].sum())

    final_stop = float_or_nan(final.get("stop_prob"))
    max_stop = series_max(stop_prob)
    argmax_stop_index = series_idxmax(stop_prob)
    argmax_stop_step_frac = (
        float_or_nan(candidates.loc[argmax_stop_index, "step_frac"])
        if argmax_stop_index is not None
        else math.nan
    )
    final_step_frac = float_or_nan(final.get("step_frac"))
    final_stop_rank_frac = rank_fraction(stop_prob, final_stop, descending=True)

    row: dict[str, Any] = {
        **{column: episode.get(column) for column in EPISODE_ID_COLUMNS},
        "should_rerank": episode.get("should_rerank"),
        "final_success": episode.get("final_success"),
        "oracle_success": episode.get("oracle_success"),
        "final_failure_bucket": episode.get("final_failure_bucket"),
        "trajectory_step_count": float_or_nan(episode.get("trajectory_step_count")),
        "decision_trace_step_count": float_or_nan(episode.get("decision_trace_step_count")),
        "candidate_count": candidate_count,
        "decision_trace_candidate_rate": safe_divide(decision_trace_count, candidate_count),
        "route_intermediate_candidate_rate": safe_divide(route_intermediate_count, candidate_count),
        "route_expanded_without_decision_candidate_rate": safe_divide(route_expanded_count, candidate_count),
        "trace_missing_candidate_rate": safe_divide(candidate_count - decision_trace_count, candidate_count),
        "revisit_count": revisit_count,
        "revisit_rate": safe_divide(revisit_count, candidate_count),
        "loop_region_count": loop_region_count,
        "loop_region_rate": safe_divide(loop_region_count, candidate_count),
        "last_k_count": last_k_count,
        "last_k_rate": safe_divide(last_k_count, candidate_count),
        "final_step_frac": final_step_frac,
        "final_path_length_m": float_or_nan(final.get("path_length_m")),
        "final_stop_prob": final_stop,
        "max_stop_prob": max_stop,
        "mean_stop_prob": series_mean(stop_prob),
        "std_stop_prob": series_std(stop_prob),
        "last_k_max_stop_prob": series_max(last_k["stop_prob"]) if not last_k.empty else math.nan,
        "last_k_mean_stop_prob": series_mean(last_k["stop_prob"]) if not last_k.empty else math.nan,
        "max_stop_minus_final_stop": subtract_or_nan(max_stop, final_stop),
        "argmax_stop_step_frac": argmax_stop_step_frac,
        "argmax_stop_gap_to_final": subtract_or_nan(argmax_stop_step_frac, final_step_frac),
        "final_stop_rank_frac": final_stop_rank_frac,
        "final_stop_margin_prob": float_or_nan(final.get("stop_margin_prob")),
        "max_stop_margin_prob": series_max(stop_margin),
        "mean_stop_margin_prob": series_mean(stop_margin),
        "std_stop_margin_prob": series_std(stop_margin),
        "last_k_max_stop_margin_prob": series_max(last_k["stop_margin_prob"]) if not last_k.empty else math.nan,
        "max_stop_margin_minus_final": subtract_or_nan(
            series_max(stop_margin),
            float_or_nan(final.get("stop_margin_prob")),
        ),
        "final_selected_prob": float_or_nan(final.get("selected_prob")),
        "max_selected_prob": series_max(selected_prob),
        "mean_selected_prob": series_mean(selected_prob),
        "std_selected_prob": series_std(selected_prob),
        "final_top1_top2_margin": float_or_nan(final.get("top1_top2_margin")),
        "max_top1_top2_margin": series_max(top_margin),
        "mean_top1_top2_margin": series_mean(top_margin),
        "final_router_entropy": float_or_nan(final.get("moe_router_entropy")),
        "mean_router_entropy": series_mean(router_entropy),
        "max_router_entropy": series_max(router_entropy),
        "final_fuse_weight": float_or_nan(final.get("fuse_weight")),
        "mean_fuse_weight": series_mean(fuse_weight),
        "max_fuse_weight": series_max(fuse_weight),
        "max_path_length_m": series_max(path_length),
        "mean_path_length_m": series_mean(path_length),
    }
    return row


def validate_features(feature_frame: pd.DataFrame) -> None:
    overlap = set(FEATURE_COLUMNS).intersection(FORBIDDEN_FEATURE_COLUMNS)
    if overlap:
        raise ValueError(f"Forbidden columns included as features: {sorted(overlap)}")
    missing = [column for column in FEATURE_COLUMNS if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def build_summary_rows(feature_frame: pd.DataFrame, split_filters: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_filter in split_filters:
        subset = select_split(feature_frame, split_filter)
        if subset.empty:
            continue
        y_true = subset["should_rerank"].astype(int).to_numpy()
        y_score = subset["gate_score"].astype(float).to_numpy()
        rows.append(
            {
                "split_filter": split_filter,
                "items": int(len(subset)),
                "positives": int(y_true.sum()),
                "base_rate": safe_divide(int(y_true.sum()), len(y_true)),
                "roc_auc": binary_metric(roc_auc_score, y_true, y_score),
                "average_precision": binary_metric(average_precision_score, y_true, y_score),
                "brier": float(brier_score_loss(y_true, y_score)),
                "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
                "mean_gate_score": float(np.mean(y_score)),
            }
        )
    return rows


def build_threshold_rows(
    feature_frame: pd.DataFrame,
    split_filter: str,
    thresholds: tuple[float, ...],
    min_recall_for_conservative: float,
) -> list[dict[str, Any]]:
    subset = select_split(feature_frame, split_filter)
    if subset.empty:
        raise ValueError(f"No rows for threshold split {split_filter!r}")
    y_true = subset["should_rerank"].astype(int).to_numpy()
    y_score = subset["gate_score"].astype(float).to_numpy()
    final_success = subset["final_success"].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        final_success_count = int((final_success == 1).sum())
        final_success_pass = int(((y_pred == 1) & (final_success == 1)).sum())
        rows.append(
            {
                "split_filter": split_filter,
                "threshold": float(threshold),
                "items": int(len(y_true)),
                "positives": int(y_true.sum()),
                "base_rate": safe_divide(int(y_true.sum()), len(y_true)),
                "pass_rate": safe_divide(int(y_pred.sum()), len(y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "false_positive_rate": safe_divide(fp, fp + tn),
                "true_negative_rate": safe_divide(tn, fp + tn),
                "final_success_pass_rate": safe_divide(final_success_pass, final_success_count),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "selection_tag": "",
            }
        )

    tag_threshold_candidates(rows, min_recall_for_conservative=min_recall_for_conservative)
    return rows


def tag_threshold_candidates(rows: list[dict[str, Any]], min_recall_for_conservative: float) -> None:
    if not rows:
        return
    best_f1 = max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"]))
    best_f1["selection_tag"] = append_tag(best_f1["selection_tag"], "best_f1")

    eligible = [row for row in rows if row["recall"] >= min_recall_for_conservative and row["tp"] > 0]
    if eligible:
        conservative = max(
            eligible,
            key=lambda row: (
                row["precision"],
                -row["pass_rate"],
                row["recall"],
                row["threshold"],
            ),
        )
    else:
        conservative = max(rows, key=lambda row: (row["precision"], row["recall"], -row["pass_rate"]))
    conservative["selection_tag"] = append_tag(conservative["selection_tag"], "conservative")

    pass_rate_limited = [row for row in rows if row["pass_rate"] <= 0.2 and row["tp"] > 0]
    if pass_rate_limited:
        low_pass = max(pass_rate_limited, key=lambda row: (row["recall"], row["precision"], row["threshold"]))
        low_pass["selection_tag"] = append_tag(low_pass["selection_tag"], "low_pass")


def recommended_threshold_from_rows(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        tags = str(row.get("selection_tag") or "").split(";")
        if "conservative" in tags:
            return float(row["threshold"])
    return None


def build_importance_rows(model: Pipeline) -> list[dict[str, Any]]:
    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    rows = [
        {
            "feature": feature,
            "coefficient": float(coef),
            "abs_coefficient": float(abs(coef)),
        }
        for feature, coef in zip(FEATURE_COLUMNS, coefficients)
    ]
    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)


def write_gate_scores_csv(
    path: Path,
    candidates: pd.DataFrame,
    episode_scores: pd.DataFrame,
    candidate_score_column: str,
    missing_candidate_score: float,
) -> None:
    score_map = episode_scores.set_index("episode_id")["gate_score"].to_dict()
    label_map = episode_scores.set_index("episode_id")["should_rerank"].to_dict()
    output = candidates[
        [
            "experiment_id",
            "dataset",
            "split",
            "protocol_split",
            "target_scope",
            "episode_id",
            "candidate_id",
            "candidate_step",
        ]
    ].copy()
    if candidate_score_column not in candidates.columns:
        raise ValueError(f"Missing candidate score column {candidate_score_column!r}")
    output["candidate_score"] = pd.to_numeric(candidates[candidate_score_column], errors="coerce").fillna(
        missing_candidate_score
    )
    output["gate_score"] = output["episode_id"].map(score_map)
    output["should_rerank"] = output["episode_id"].map(label_map)
    if output["gate_score"].isna().any():
        missing = output.loc[output["gate_score"].isna(), "episode_id"].iloc[0]
        raise ValueError(f"Missing gate score for episode_id={missing}")
    output.to_csv(path, index=False, columns=GATE_SCORE_FIELDNAMES)


def build_model_payload(
    model: Pipeline,
    feature_columns: list[str],
    target_scope: str,
    train_split: str,
    dev_split: str,
    thresholds: tuple[float, ...],
    recommended_threshold: float | None,
    logistic_c: float,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    classifier = model.named_steps["classifier"]
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    return {
        "schema_version": SCHEMA_VERSION,
        "target_scope": target_scope,
        "train_split": train_split,
        "dev_split": dev_split,
        "feature_columns": feature_columns,
        "thresholds": list(thresholds),
        "recommended_threshold": recommended_threshold,
        "classifier": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "c": logistic_c,
            "max_iter": max_iter,
            "random_state": random_state,
            "intercept": [float(value) for value in classifier.intercept_],
            "coef": [float(value) for value in classifier.coef_[0]],
        },
        "imputer": {
            "type": "SimpleImputer",
            "strategy": "median",
            "statistics": [none_if_nan(value) for value in imputer.statistics_],
        },
        "scaler": {
            "type": "StandardScaler",
            "mean": [none_if_nan(value) for value in scaler.mean_],
            "scale": [none_if_nan(value) for value in scaler.scale_],
        },
    }


def write_report(
    path: Path,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    eval_manifest: dict[str, Any] | None,
) -> None:
    selected_thresholds = [
        row for row in threshold_rows if row.get("selection_tag")
    ]
    lines = [
        "# Endpoint Gate Baseline Report",
        "",
        "This report is generated by `scripts/analysis/train_endpoint_gate_baseline.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- train_split: `{manifest['train_split']}`",
        f"- dev_split: `{manifest['dev_split']}`",
        f"- model: class-balanced sklearn LogisticRegression",
        f"- candidate_score in exported score CSV: `{manifest['candidate_score_column']}`",
        f"- recommended_threshold: `{manifest['recommended_threshold']}`",
        f"- include_test_summary: `{manifest['include_test_summary']}`",
        "",
        "## Summary",
        "",
        markdown_table(
            ["split", "items", "positives", "base_rate", "roc_auc", "average_precision", "brier", "log_loss"],
            [
                [
                    row["split_filter"],
                    row["items"],
                    row["positives"],
                    pct(row["base_rate"]),
                    fmt(row["roc_auc"]),
                    fmt(row["average_precision"]),
                    fmt(row["brier"]),
                    fmt(row["log_loss"]),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Threshold Candidates",
        "",
        markdown_table(
            ["tag", "threshold", "pass_rate", "precision", "recall", "f1", "final_success_pass_rate", "tp", "fp", "fn"],
            [
                [
                    row["selection_tag"],
                    fmt(row["threshold"]),
                    pct(row["pass_rate"]),
                    pct(row["precision"]),
                    pct(row["recall"]),
                    fmt(row["f1"]),
                    pct(row["final_success_pass_rate"]),
                    row["tp"],
                    row["fp"],
                    row["fn"],
                ]
                for row in selected_thresholds[:10]
            ],
        ),
        "",
        "## Top Coefficients",
        "",
        markdown_table(
            ["feature", "coef", "abs_coef"],
            [
                [row["feature"], fmt(row["coefficient"]), fmt(row["abs_coefficient"])]
                for row in importance_rows[:15]
            ],
        ),
        "",
        "## Files",
        "",
    ]
    for key, value in manifest["files"].items():
        lines.append(f"- {key}: `{value}`")
    if eval_manifest is not None:
        lines.extend(
            [
                "",
                "## Evaluator Bridge",
                "",
                "The generated `gate_scores.csv` was also passed to `eval_endpoint_reranker.py` on the dev split.",
                f"- evaluator output_dir: `{eval_manifest['output_dir']}`",
                f"- configs: `{eval_manifest['counts']['configs']}`",
                f"- summary_rows: `{eval_manifest['counts']['summary_rows']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def select_split(feature_frame: pd.DataFrame, split_filter: str) -> pd.DataFrame:
    return feature_frame[
        (feature_frame["protocol_split"] == split_filter)
        | (feature_frame["split"] == split_filter)
    ].copy()


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


def series_valid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def series_mean(series: pd.Series) -> float:
    values = series_valid(series)
    return float(values.mean()) if not values.empty else math.nan


def series_std(series: pd.Series) -> float:
    values = series_valid(series)
    return float(values.std(ddof=0)) if not values.empty else math.nan


def series_max(series: pd.Series) -> float:
    values = series_valid(series)
    return float(values.max()) if not values.empty else math.nan


def series_idxmax(series: pd.Series) -> Any | None:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return None
    return values.idxmax()


def rank_fraction(series: pd.Series, value: float, descending: bool = True) -> float:
    values = series_valid(series)
    if values.empty or math.isnan(value):
        return math.nan
    if descending:
        rank = int((values > value).sum()) + 1
    else:
        rank = int((values < value).sum()) + 1
    return safe_divide(rank, len(values))


def float_or_nan(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def subtract_or_nan(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return float(left - right)


def binary_metric(metric_fn: Any, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(metric_fn(y_true, y_score))


def append_tag(existing: str, new_tag: str) -> str:
    if not existing:
        return new_tag
    tags = existing.split(";")
    if new_tag not in tags:
        tags.append(new_tag)
    return ";".join(tags)


def none_if_nan(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = float(part)
        if not math.isfinite(parsed):
            raise ValueError(f"Non-finite float: {part}")
        values.append(parsed)
    if not values:
        raise ValueError("Expected at least one float")
    return values


def format_float(value: float) -> str:
    return f"{value:g}"


def path_to_string(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, frame: pd.DataFrame, fieldnames: list[str]) -> None:
    frame.to_csv(path, index=False, columns=fieldnames)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value * 100:.2f}"


if __name__ == "__main__":
    main()
