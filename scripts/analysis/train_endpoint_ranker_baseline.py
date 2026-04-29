#!/usr/bin/env python3
"""Train a lightweight candidate-level endpoint ranker baseline."""

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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_endpoint_reranker as reranker_eval  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_ranker_baseline.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_OUTPUT_NAME = "ranker_baseline"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_DEV_SPLIT = "dev"
DEFAULT_EVAL_SPLIT = "dev"
DEFAULT_GATE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7)
DEFAULT_TAUS = (0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT_ALLOW_CHANGE_FINAL = (True,)
DEFAULT_RANDOM_STATE = 17
EPS = 1e-12

ID_COLUMNS = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "candidate_id",
    "internal_item_id",
    "saved_instr_id",
    "candidate_step",
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

BOOLEAN_FEATURE_COLUMNS = [
    "trace_available",
    "is_final",
    "is_revisit",
    "is_loop_region",
    "is_last_k",
]

NUMERIC_BASE_FEATURE_COLUMNS = [
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
]

DERIVED_FEATURE_COLUMNS = [
    "candidate_count",
    "candidate_step_frac",
    "steps_to_final",
    "steps_to_final_frac",
    "path_length_ratio_to_final",
    "path_length_remaining_m",
    "stop_rank_frac",
    "stop_margin_rank_frac",
    "selected_rank_frac",
    "top_margin_rank_frac",
    "router_entropy_rank_frac",
    "path_length_rank_frac",
    "stop_minus_final_stop",
    "stop_margin_minus_final_stop",
    "selected_minus_final_selected",
    "router_entropy_minus_final_router_entropy",
    "fuse_weight_minus_final_fuse_weight",
]

FEATURE_COLUMNS = BOOLEAN_FEATURE_COLUMNS + NUMERIC_BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

FEATURE_FIELDNAMES = list(dict.fromkeys(ID_COLUMNS + [
    "success_label",
    "spl_at_candidate",
    "reward",
    "is_final",
    "final_success",
    "oracle_success",
    "should_rerank",
] + FEATURE_COLUMNS + ["candidate_score", "gate_score"]))

SCORE_FIELDNAMES = [
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
    "success_label",
    "should_rerank",
    "is_final",
]

SUMMARY_FIELDNAMES = [
    "split_filter",
    "items",
    "episodes",
    "positives",
    "positive_rate",
    "roc_auc",
    "average_precision",
    "brier",
    "log_loss",
    "mean_candidate_score",
    "top1_success_rate",
    "top1_is_final_rate",
    "top1_recovery_rate",
    "top1_harm_rate",
    "top1_mean_margin_over_final",
]

PAIR_SUMMARY_FIELDNAMES = [
    "split_filter",
    "pair_type",
    "pairs",
    "accuracy",
    "strict_accuracy",
    "tie_rate",
    "mean_score_margin",
]

IMPORTANCE_FIELDNAMES = [
    "feature",
    "coefficient",
    "abs_coefficient",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sklearn candidate-level endpoint ranker baseline.",
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
        "--pair-csv",
        default=None,
        help="Preference pairs CSV. Defaults to <endpoint-learning-dir>/preference_pairs/preference_pairs.csv.",
    )
    parser.add_argument(
        "--gate-score-csv",
        default=None,
        help=(
            "Optional gate score CSV. Defaults to <endpoint-learning-dir>/gate_baseline/gate_scores.csv "
            "when it exists; otherwise every episode uses --default-gate-score."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <endpoint-learning-dir>/ranker_baseline.",
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
        help="Protocol split used for model diagnostics. Defaults to dev.",
    )
    parser.add_argument(
        "--eval-split",
        default=DEFAULT_EVAL_SPLIT,
        help="Comma-separated split filter for evaluator bridge. Defaults to dev.",
    )
    parser.add_argument(
        "--gate-thresholds",
        default=",".join(format_float(value) for value in DEFAULT_GATE_THRESHOLDS),
        help="Comma-separated gate thresholds for evaluator bridge.",
    )
    parser.add_argument(
        "--taus",
        default=",".join(format_float(value) for value in DEFAULT_TAUS),
        help="Comma-separated final-stay taus for evaluator bridge.",
    )
    parser.add_argument(
        "--allow-change-final",
        default=",".join("true" if value else "false" for value in DEFAULT_ALLOW_CHANGE_FINAL),
        help="true, false, or comma-separated true,false grid for evaluator bridge.",
    )
    parser.add_argument(
        "--default-gate-score",
        type=float,
        default=1.0,
        help="Gate score used when no gate CSV is available or an episode is missing.",
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
        "--no-episode-balanced-sample-weight",
        action="store_true",
        help="Disable per-episode balanced candidate sample weights during training.",
    )
    parser.add_argument(
        "--skip-reranker-eval",
        action="store_true",
        help="Only train/export ranker scores; do not call eval_endpoint_reranker.py.",
    )
    parser.add_argument(
        "--include-test-summary",
        action="store_true",
        help="Also summarize test/val_unseen ranker metrics. Keep disabled while tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    endpoint_learning_dir = resolve_endpoint_learning_dir(experiment_dir, args.endpoint_learning_dir)
    candidate_csv = (
        Path(args.candidate_csv).resolve()
        if args.candidate_csv
        else endpoint_learning_dir / "candidate_groups" / "endpoint_candidates.csv"
    )
    episode_csv = (
        Path(args.episode_csv).resolve()
        if args.episode_csv
        else endpoint_learning_dir / "candidate_groups" / "episode_groups.csv"
    )
    pair_csv = (
        Path(args.pair_csv).resolve()
        if args.pair_csv
        else endpoint_learning_dir / "preference_pairs" / "preference_pairs.csv"
    )
    gate_score_csv = resolve_gate_score_csv(endpoint_learning_dir, args.gate_score_csv)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else endpoint_learning_dir / DEFAULT_OUTPUT_NAME
    )

    manifest = train_endpoint_ranker_baseline(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        pair_csv=pair_csv,
        gate_score_csv=gate_score_csv,
        output_dir=output_dir,
        target_scope=args.target_scope,
        train_split=args.train_split,
        dev_split=args.dev_split,
        eval_split_filters=parse_string_list(args.eval_split),
        gate_thresholds=tuple(parse_float_list(args.gate_thresholds)),
        taus=tuple(parse_float_list(args.taus)),
        allow_change_final_values=tuple(parse_bool_list(args.allow_change_final)),
        default_gate_score=args.default_gate_score,
        logistic_c=args.logistic_c,
        max_iter=args.max_iter,
        random_state=args.random_state,
        episode_balanced_sample_weight=not args.no_episode_balanced_sample_weight,
        run_reranker_eval=not args.skip_reranker_eval,
        include_test_summary=args.include_test_summary,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def train_endpoint_ranker_baseline(
    candidate_csv: Path,
    episode_csv: Path,
    pair_csv: Path,
    gate_score_csv: Path | None,
    output_dir: Path,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    dev_split: str = DEFAULT_DEV_SPLIT,
    eval_split_filters: tuple[str, ...] = (DEFAULT_EVAL_SPLIT,),
    gate_thresholds: tuple[float, ...] = DEFAULT_GATE_THRESHOLDS,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    allow_change_final_values: tuple[bool, ...] = DEFAULT_ALLOW_CHANGE_FINAL,
    default_gate_score: float = 1.0,
    logistic_c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = DEFAULT_RANDOM_STATE,
    episode_balanced_sample_weight: bool = True,
    run_reranker_eval: bool = True,
    include_test_summary: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_csv, low_memory=False)
    episodes = pd.read_csv(episode_csv, low_memory=False)
    candidates = candidates[candidates["target_scope"] == target_scope].copy()
    episodes = episodes[episodes["target_scope"] == target_scope].copy()
    if candidates.empty:
        raise ValueError(f"No candidates found for target_scope={target_scope!r}")
    if episodes.empty:
        raise ValueError(f"No episodes found for target_scope={target_scope!r}")

    feature_frame = build_ranker_feature_frame(candidates)
    validate_features(feature_frame)

    train_mask = feature_frame["protocol_split"] == train_split
    dev_mask = feature_frame["protocol_split"] == dev_split
    if not train_mask.any():
        raise ValueError(f"No training rows with protocol_split={train_split!r}")
    if not dev_mask.any():
        raise ValueError(f"No dev rows with protocol_split={dev_split!r}")

    y_train = feature_frame.loc[train_mask, "success_label"].astype(int).to_numpy()
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training split must contain both positive and negative success_label candidates")

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
    fit_kwargs: dict[str, Any] = {}
    if episode_balanced_sample_weight:
        fit_kwargs["classifier__sample_weight"] = build_episode_sample_weight(feature_frame.loc[train_mask])
    model.fit(feature_frame.loc[train_mask, FEATURE_COLUMNS], y_train, **fit_kwargs)

    feature_frame["candidate_score"] = model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]
    gate_score_map = load_gate_score_map(gate_score_csv) if gate_score_csv is not None else {}
    feature_frame["gate_score"] = feature_frame["episode_id"].map(gate_score_map).fillna(default_gate_score)

    summary_splits = [train_split, dev_split]
    if include_test_summary:
        summary_splits.extend(["test", "val_unseen"])
    summary_rows = build_summary_rows(feature_frame, split_filters=tuple(dict.fromkeys(summary_splits)))
    pair_summary_rows = build_pair_summary_rows(
        pair_csv=pair_csv,
        score_frame=feature_frame,
        split_filters=tuple(dict.fromkeys(summary_splits)),
        target_scope=target_scope,
    )
    importance_rows = build_importance_rows(model)

    features_csv = output_dir / "ranker_features.csv"
    scores_csv = output_dir / "ranker_scores.csv"
    summary_csv = output_dir / "ranker_candidate_summary.csv"
    pair_summary_csv = output_dir / "ranker_pair_summary.csv"
    importance_csv = output_dir / "ranker_feature_importance.csv"
    model_joblib = output_dir / "ranker_model.joblib"
    model_json = output_dir / "ranker_model.json"
    report_md = output_dir / "endpoint_ranker_report.md"
    manifest_path = output_dir / "manifest.json"

    write_csv(features_csv, feature_frame, FEATURE_FIELDNAMES)
    write_csv(scores_csv, feature_frame, SCORE_FIELDNAMES)
    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDNAMES).to_csv(summary_csv, index=False)
    pd.DataFrame(pair_summary_rows, columns=PAIR_SUMMARY_FIELDNAMES).to_csv(pair_summary_csv, index=False)
    pd.DataFrame(importance_rows, columns=IMPORTANCE_FIELDNAMES).to_csv(importance_csv, index=False)
    joblib.dump(model, model_joblib)

    eval_manifest: dict[str, Any] | None = None
    if run_reranker_eval:
        eval_manifest = reranker_eval.evaluate_endpoint_reranker(
            candidate_csv=candidate_csv,
            episode_csv=episode_csv,
            score_csv=scores_csv,
            output_dir=output_dir / "eval_protocol",
            target_scope=target_scope,
            split_filters=eval_split_filters,
            gate_thresholds=gate_thresholds,
            taus=taus,
            allow_change_final_values=allow_change_final_values,
            candidate_score_column="candidate_score",
            gate_score_column="gate_score",
            default_gate_score=default_gate_score,
        )

    model_payload = build_model_payload(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        target_scope=target_scope,
        train_split=train_split,
        dev_split=dev_split,
        logistic_c=logistic_c,
        max_iter=max_iter,
        random_state=random_state,
        episode_balanced_sample_weight=episode_balanced_sample_weight,
    )
    write_json(model_json, model_payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": target_scope,
        "train_split": train_split,
        "dev_split": dev_split,
        "eval_split_filters": list(eval_split_filters),
        "gate_thresholds": list(gate_thresholds),
        "taus": list(taus),
        "allow_change_final_values": list(allow_change_final_values),
        "default_gate_score": default_gate_score,
        "gate_score_csv": path_to_string(gate_score_csv) if gate_score_csv is not None else None,
        "include_test_summary": include_test_summary,
        "model": {
            "type": "sklearn.pipeline.Pipeline",
            "classifier": "LogisticRegression",
            "loss": "candidate_success_cross_entropy",
            "class_weight": "balanced",
            "episode_balanced_sample_weight": episode_balanced_sample_weight,
            "logistic_c": logistic_c,
            "max_iter": max_iter,
            "random_state": random_state,
        },
        "counts": {
            "episodes": int(feature_frame["episode_id"].nunique()),
            "train_candidates": int(train_mask.sum()),
            "dev_candidates": int(dev_mask.sum()),
            "candidates": int(len(feature_frame)),
            "features": len(FEATURE_COLUMNS),
        },
        "files": {
            "candidate_csv": path_to_string(candidate_csv),
            "episode_csv": path_to_string(episode_csv),
            "pair_csv": path_to_string(pair_csv),
            "ranker_features_csv": path_to_string(features_csv),
            "ranker_scores_csv": path_to_string(scores_csv),
            "ranker_candidate_summary_csv": path_to_string(summary_csv),
            "ranker_pair_summary_csv": path_to_string(pair_summary_csv),
            "ranker_feature_importance_csv": path_to_string(importance_csv),
            "ranker_model_joblib": path_to_string(model_joblib),
            "ranker_model_json": path_to_string(model_json),
            "endpoint_ranker_report_md": path_to_string(report_md),
        },
        "reranker_eval": eval_manifest,
    }
    write_json(manifest_path, manifest)
    write_report(
        path=report_md,
        manifest=manifest,
        summary_rows=summary_rows,
        pair_summary_rows=pair_summary_rows,
        importance_rows=importance_rows,
        eval_manifest=eval_manifest,
    )
    return manifest


def build_ranker_feature_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    for column in NUMERIC_BASE_FEATURE_COLUMNS:
        candidates[column] = pd.to_numeric(candidates.get(column), errors="coerce")
    for column in BOOLEAN_FEATURE_COLUMNS:
        candidates[column] = to_bool_series(candidates.get(column, pd.Series(False, index=candidates.index)))

    for column in ["success_label", "is_final", "final_success", "oracle_success", "should_rerank"]:
        candidates[column] = to_bool_series(candidates.get(column, pd.Series(False, index=candidates.index))).astype(int)
    for column in ["spl_at_candidate", "reward"]:
        candidates[column] = pd.to_numeric(candidates.get(column), errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, group in candidates.groupby("episode_id", sort=False):
        rows.extend(build_one_episode_candidate_features(group))

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        raise ValueError("No candidate feature rows were built")
    feature_frame = feature_frame.sort_values(["split", "protocol_split", "episode_id", "candidate_step"]).reset_index(drop=True)
    for column in FEATURE_COLUMNS:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    return feature_frame


def build_one_episode_candidate_features(group: pd.DataFrame) -> list[dict[str, Any]]:
    group = group.sort_values("candidate_step").copy()
    final_rows = group[group["is_final"].astype(bool)]
    final = final_rows.iloc[0] if not final_rows.empty else group.iloc[-1]
    candidate_count = int(len(group))
    final_step = float_or_nan(final.get("candidate_step"))
    final_path = float_or_nan(final.get("path_length_m"))
    final_stop = float_or_nan(final.get("stop_prob"))
    final_stop_margin = float_or_nan(final.get("stop_margin_prob"))
    final_selected = float_or_nan(final.get("selected_prob"))
    final_router_entropy = float_or_nan(final.get("moe_router_entropy"))
    final_fuse_weight = float_or_nan(final.get("fuse_weight"))

    rows: list[dict[str, Any]] = []
    for _, candidate in group.iterrows():
        candidate_step = float_or_nan(candidate.get("candidate_step"))
        path_length = float_or_nan(candidate.get("path_length_m"))
        row = {
            **{column: candidate.get(column) for column in ID_COLUMNS},
            "success_label": int(candidate.get("success_label")),
            "spl_at_candidate": float_or_nan(candidate.get("spl_at_candidate")),
            "reward": float_or_nan(candidate.get("reward")),
            "is_final": int(candidate.get("is_final")),
            "final_success": int(candidate.get("final_success")),
            "oracle_success": int(candidate.get("oracle_success")),
            "should_rerank": int(candidate.get("should_rerank")),
            "trace_available": int(candidate.get("trace_available")),
            "is_revisit": int(candidate.get("is_revisit")),
            "is_loop_region": int(candidate.get("is_loop_region")),
            "is_last_k": int(candidate.get("is_last_k")),
            "trajectory_step_count": float_or_nan(candidate.get("trajectory_step_count")),
            "candidate_step": candidate_step,
            "step_frac": float_or_nan(candidate.get("step_frac")),
            "path_length_m": path_length,
            "stop_prob": float_or_nan(candidate.get("stop_prob")),
            "stop_margin_prob": float_or_nan(candidate.get("stop_margin_prob")),
            "selected_prob": float_or_nan(candidate.get("selected_prob")),
            "top1_top2_margin": float_or_nan(candidate.get("top1_top2_margin")),
            "moe_router_entropy": float_or_nan(candidate.get("moe_router_entropy")),
            "fuse_weight": float_or_nan(candidate.get("fuse_weight")),
            "candidate_count": candidate_count,
            "candidate_step_frac": safe_divide(candidate_step, final_step),
            "steps_to_final": subtract_or_nan(final_step, candidate_step),
            "steps_to_final_frac": safe_divide(subtract_or_nan(final_step, candidate_step), final_step),
            "path_length_ratio_to_final": safe_divide(path_length, final_path),
            "path_length_remaining_m": subtract_or_nan(final_path, path_length),
            "stop_rank_frac": rank_fraction(group["stop_prob"], candidate.get("stop_prob"), descending=True),
            "stop_margin_rank_frac": rank_fraction(group["stop_margin_prob"], candidate.get("stop_margin_prob"), descending=True),
            "selected_rank_frac": rank_fraction(group["selected_prob"], candidate.get("selected_prob"), descending=True),
            "top_margin_rank_frac": rank_fraction(group["top1_top2_margin"], candidate.get("top1_top2_margin"), descending=True),
            "router_entropy_rank_frac": rank_fraction(group["moe_router_entropy"], candidate.get("moe_router_entropy"), descending=True),
            "path_length_rank_frac": rank_fraction(group["path_length_m"], candidate.get("path_length_m"), descending=False),
            "stop_minus_final_stop": subtract_or_nan(float_or_nan(candidate.get("stop_prob")), final_stop),
            "stop_margin_minus_final_stop": subtract_or_nan(float_or_nan(candidate.get("stop_margin_prob")), final_stop_margin),
            "selected_minus_final_selected": subtract_or_nan(float_or_nan(candidate.get("selected_prob")), final_selected),
            "router_entropy_minus_final_router_entropy": subtract_or_nan(
                float_or_nan(candidate.get("moe_router_entropy")),
                final_router_entropy,
            ),
            "fuse_weight_minus_final_fuse_weight": subtract_or_nan(
                float_or_nan(candidate.get("fuse_weight")),
                final_fuse_weight,
            ),
        }
        rows.append(row)
    return rows


def validate_features(feature_frame: pd.DataFrame) -> None:
    overlap = set(FEATURE_COLUMNS).intersection(FORBIDDEN_FEATURE_COLUMNS)
    if overlap:
        raise ValueError(f"Forbidden columns included as features: {sorted(overlap)}")
    missing = [column for column in FEATURE_COLUMNS if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def build_episode_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    group_sizes = frame.groupby("episode_id")["candidate_id"].transform("count").astype(float)
    return (1.0 / group_sizes).to_numpy()


def load_gate_score_map(gate_score_csv: Path) -> dict[str, float]:
    if not gate_score_csv.exists():
        raise FileNotFoundError(gate_score_csv)
    scores = pd.read_csv(gate_score_csv)
    if "gate_score" not in scores.columns:
        raise ValueError(f"Gate score CSV must contain gate_score: {gate_score_csv}")
    if "episode_id" not in scores.columns:
        raise ValueError(f"Gate score CSV must contain episode_id: {gate_score_csv}")

    score_map: dict[str, float] = {}
    for episode_id, rows in scores.groupby("episode_id", sort=False):
        values = pd.to_numeric(rows["gate_score"], errors="coerce").dropna().unique()
        if len(values) == 0:
            continue
        if len(values) > 1 and float(np.max(values) - np.min(values)) > EPS:
            raise ValueError(f"Conflicting gate_score values for episode_id={episode_id}")
        score_map[str(episode_id)] = float(values[0])
    return score_map


def build_summary_rows(feature_frame: pd.DataFrame, split_filters: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_filter in split_filters:
        subset = select_split(feature_frame, split_filter)
        if subset.empty:
            continue
        y_true = subset["success_label"].astype(int).to_numpy()
        y_score = subset["candidate_score"].astype(float).to_numpy()
        top1_rows = top1_candidate_rows(subset)
        rows.append(
            {
                "split_filter": split_filter,
                "items": int(len(subset)),
                "episodes": int(subset["episode_id"].nunique()),
                "positives": int(y_true.sum()),
                "positive_rate": safe_divide(int(y_true.sum()), len(y_true)),
                "roc_auc": binary_metric(roc_auc_score, y_true, y_score),
                "average_precision": binary_metric(average_precision_score, y_true, y_score),
                "brier": float(brier_score_loss(y_true, y_score)),
                "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
                "mean_candidate_score": float(np.mean(y_score)),
                "top1_success_rate": mean_series(top1_rows["success_label"]) if not top1_rows.empty else math.nan,
                "top1_is_final_rate": mean_series(top1_rows["is_final"]) if not top1_rows.empty else math.nan,
                "top1_recovery_rate": top1_recovery_rate(top1_rows),
                "top1_harm_rate": top1_harm_rate(top1_rows),
                "top1_mean_margin_over_final": top1_mean_margin_over_final(subset),
            }
        )
    return rows


def top1_candidate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.sort_values(["episode_id", "candidate_score", "candidate_step"], ascending=[True, False, False])
    return sorted_frame.groupby("episode_id", sort=False).head(1).copy()


def top1_recovery_rate(top1_rows: pd.DataFrame) -> float:
    if top1_rows.empty:
        return math.nan
    recovered = (top1_rows["final_success"].astype(int) == 0) & (top1_rows["success_label"].astype(int) == 1)
    return float(recovered.mean())


def top1_harm_rate(top1_rows: pd.DataFrame) -> float:
    if top1_rows.empty:
        return math.nan
    harmed = (top1_rows["final_success"].astype(int) == 1) & (top1_rows["success_label"].astype(int) == 0)
    return float(harmed.mean())


def top1_mean_margin_over_final(frame: pd.DataFrame) -> float:
    margins: list[float] = []
    for _, group in frame.groupby("episode_id", sort=False):
        final_rows = group[group["is_final"].astype(int) == 1]
        if final_rows.empty:
            continue
        final_score = float(final_rows.iloc[0]["candidate_score"])
        best_score = float(group["candidate_score"].max())
        margins.append(best_score - final_score)
    return float(np.mean(margins)) if margins else math.nan


def build_pair_summary_rows(
    pair_csv: Path,
    score_frame: pd.DataFrame,
    split_filters: tuple[str, ...],
    target_scope: str,
) -> list[dict[str, Any]]:
    if not pair_csv.exists():
        return []
    pairs = pd.read_csv(pair_csv)
    pairs = pairs[pairs["target_scope"] == target_scope].copy()
    if pairs.empty:
        return []
    score_map = score_frame.set_index("candidate_id")["candidate_score"].to_dict()
    rows: list[dict[str, Any]] = []
    for split_filter in split_filters:
        split_pairs = select_split(pairs, split_filter)
        if split_pairs.empty:
            continue
        for pair_type in ["all", *sorted(split_pairs["pair_type"].dropna().unique().tolist())]:
            subset = split_pairs if pair_type == "all" else split_pairs[split_pairs["pair_type"] == pair_type]
            margins = []
            for _, row in subset.iterrows():
                chosen_score = score_map.get(row.get("chosen_candidate_id"))
                rejected_score = score_map.get(row.get("rejected_candidate_id"))
                if chosen_score is None or rejected_score is None:
                    continue
                margins.append(float(chosen_score) - float(rejected_score))
            if not margins:
                continue
            margin_array = np.array(margins, dtype=float)
            strict_correct = int((margin_array > EPS).sum())
            ties = int((np.abs(margin_array) <= EPS).sum())
            rows.append(
                {
                    "split_filter": split_filter,
                    "pair_type": pair_type,
                    "pairs": int(len(margin_array)),
                    "accuracy": float((strict_correct + 0.5 * ties) / len(margin_array)),
                    "strict_accuracy": float(strict_correct / len(margin_array)),
                    "tie_rate": float(ties / len(margin_array)),
                    "mean_score_margin": float(np.mean(margin_array)),
                }
            )
    return rows


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


def build_model_payload(
    model: Pipeline,
    feature_columns: list[str],
    target_scope: str,
    train_split: str,
    dev_split: str,
    logistic_c: float,
    max_iter: int,
    random_state: int,
    episode_balanced_sample_weight: bool,
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
        "classifier": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "loss": "candidate_success_cross_entropy",
            "c": logistic_c,
            "max_iter": max_iter,
            "random_state": random_state,
            "episode_balanced_sample_weight": episode_balanced_sample_weight,
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
    pair_summary_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    eval_manifest: dict[str, Any] | None,
) -> None:
    eval_rows = read_eval_summary(eval_manifest)
    lines = [
        "# Endpoint Ranker Baseline Report",
        "",
        "This report is generated by `scripts/analysis/train_endpoint_ranker_baseline.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- train_split: `{manifest['train_split']}`",
        f"- dev_split: `{manifest['dev_split']}`",
        "- model: class-balanced sklearn LogisticRegression",
        "- loss: candidate success cross entropy",
        f"- episode_balanced_sample_weight: `{manifest['model']['episode_balanced_sample_weight']}`",
        f"- gate_score_csv: `{manifest['gate_score_csv']}`",
        "",
        "## Candidate Metrics",
        "",
        markdown_table(
            [
                "split",
                "candidates",
                "episodes",
                "positive_rate",
                "roc_auc",
                "average_precision",
                "brier",
                "log_loss",
                "top1_success",
                "top1_harm",
            ],
            [
                [
                    row["split_filter"],
                    row["items"],
                    row["episodes"],
                    pct(row["positive_rate"]),
                    fmt(row["roc_auc"]),
                    fmt(row["average_precision"]),
                    fmt(row["brier"]),
                    fmt(row["log_loss"]),
                    pct(row["top1_success_rate"]),
                    pct(row["top1_harm_rate"]),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Pair Agreement",
        "",
        markdown_table(
            ["split", "pair_type", "pairs", "accuracy", "strict_accuracy", "mean_margin"],
            [
                [
                    row["split_filter"],
                    row["pair_type"],
                    row["pairs"],
                    pct(row["accuracy"]),
                    pct(row["strict_accuracy"]),
                    fmt(row["mean_score_margin"]),
                ]
                for row in pair_summary_rows
                if row["pair_type"] == "all"
            ],
        ),
        "",
        "## Evaluator Bridge",
        "",
        markdown_table(
            [
                "gate",
                "tau",
                "SR",
                "delta_SR",
                "SPL",
                "delta_SPL",
                "recovery",
                "harm",
                "changed",
            ],
            [
                [
                    fmt(row["gate_threshold"]),
                    fmt(row["tau"]),
                    pct(row["SR"]),
                    pct(row["delta_SR"]),
                    pct(row["SPL"]),
                    pct(row["delta_SPL"]),
                    pct(row["recovery_rate"]),
                    pct(row["harm_rate"]),
                    pct(row["changed_endpoint_rate"]),
                ]
                for row in eval_rows[:10]
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
                "## Evaluator Files",
                "",
                f"- evaluator output_dir: `{eval_manifest['output_dir']}`",
                f"- configs: `{eval_manifest['counts']['configs']}`",
                f"- summary_rows: `{eval_manifest['counts']['summary_rows']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_eval_summary(eval_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if eval_manifest is None:
        return []
    summary_path = Path(eval_manifest["files"]["summary_csv"])
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return []
    summary = summary[summary["allow_change_final"].astype(str).str.lower().isin(["true", "1"])]
    summary = summary.sort_values(
        ["delta_SR", "net_recovery_rate", "harm_rate", "changed_endpoint_rate"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    return summary.to_dict("records")


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def resolve_gate_score_csv(endpoint_learning_dir: Path, gate_score_csv: str | None) -> Path | None:
    if gate_score_csv:
        return Path(gate_score_csv).resolve()
    default_path = endpoint_learning_dir / "gate_baseline" / "gate_scores.csv"
    return default_path if default_path.exists() else None


def select_split(frame: pd.DataFrame, split_filter: str) -> pd.DataFrame:
    return frame[
        (frame["protocol_split"] == split_filter)
        | (frame["split"] == split_filter)
    ].copy()


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def safe_divide(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    if denominator == 0 or math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    return numerator / denominator


def series_valid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def rank_fraction(series: pd.Series, value: Any, descending: bool = True) -> float:
    values = series_valid(series)
    value = float_or_nan(value)
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


def mean_series(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def binary_metric(metric_fn: Any, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(metric_fn(y_true, y_score))


def none_if_nan(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def parse_string_list(value: str) -> tuple[str, ...]:
    value = value.replace("/", ",")
    return tuple(item.strip() for item in value.split(",") if item.strip())


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


def parse_bool_list(value: str) -> list[bool]:
    values: list[bool] = []
    for part in value.split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text in {"true", "1", "yes", "y"}:
            values.append(True)
        elif text in {"false", "0", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Could not parse boolean value: {part!r}")
    if not values:
        raise ValueError("Expected at least one boolean")
    return values


def format_float(value: float) -> str:
    return f"{value:g}"


def path_to_string(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
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
