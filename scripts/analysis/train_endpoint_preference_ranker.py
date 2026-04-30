#!/usr/bin/env python3
"""Train SPL-aware endpoint preference rankers for phase 4.6.

The script keeps the phase-4 evaluator contract unchanged: it exports a
ranker_scores.csv containing candidate_score and gate_score, then optionally
bridges into eval_endpoint_reranker.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_endpoint_reranker as reranker_eval  # noqa: E402
import train_endpoint_ranker_baseline as baseline  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_preference_ranker.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_OUTPUT_NAME = "preference_ranker"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_DEV_SPLIT = "dev"
DEFAULT_EVAL_SPLIT = "dev"
DEFAULT_GATE_THRESHOLDS = (0.0, 0.5, 0.7, 0.85, 0.9, 0.95)
DEFAULT_TAUS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
DEFAULT_ALLOW_CHANGE_FINAL = (True, False)
DEFAULT_RANDOM_STATE = 17
EPS = 1e-12

DEFAULT_PAIR_WEIGHTS = {
    "success_gt_fail": 1.0,
    "better_spl_success_gt_lower_spl_success": 2.0,
    "final_success_final_gt_failed_nonfinal": 4.0,
}

TRAINING_CURVE_COLUMNS = [
    "epoch",
    "pairwise_loss",
    "group_loss",
    "objective_loss",
    "pair_batches",
    "groups",
    "learning_rate",
]

IMPORTANCE_FIELDNAMES = [
    "feature",
    "coefficient",
    "abs_coefficient",
]


@dataclass(frozen=True)
class PairArrays:
    chosen_idx: np.ndarray
    rejected_idx: np.ndarray
    weights: np.ndarray
    summary: list[dict[str, Any]]


@dataclass(frozen=True)
class GroupTarget:
    episode_id: str
    indices: np.ndarray
    target: np.ndarray
    weight: float
    target_kind: str


@dataclass
class AdamState:
    m_w: np.ndarray
    v_w: np.ndarray
    m_b: float = 0.0
    v_b: float = 0.0
    t: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a phase-4.6 endpoint preference ranker.",
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
            "Gate score CSV. Defaults to <endpoint-learning-dir>/gate_baseline/gate_scores.csv "
            "when it exists; otherwise every episode uses --default-gate-score."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <endpoint-learning-dir>/preference_ranker.",
    )
    parser.add_argument("--target-scope", default=DEFAULT_TARGET_SCOPE)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--dev-split", default=DEFAULT_DEV_SPLIT)
    parser.add_argument(
        "--eval-split",
        default=DEFAULT_EVAL_SPLIT,
        help="Comma-separated split filters for evaluator bridge. Defaults to dev.",
    )
    parser.add_argument(
        "--objective",
        choices=("pairwise", "pairwise_listwise", "listwise"),
        default="pairwise",
        help="Training objective family. pairwise_listwise enables group loss if weight is unset.",
    )
    parser.add_argument(
        "--pair-weights",
        default=format_pair_weights(DEFAULT_PAIR_WEIGHTS),
        help="Comma-separated pair_type=weight values.",
    )
    parser.add_argument(
        "--max-pairs-per-type",
        type=int,
        default=200_000,
        help="Deterministic cap for each pair type. Use <=0 for all pairs.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--group-loss-weight",
        type=float,
        default=None,
        help="Listwise softmax loss weight. Defaults to 0 for pairwise and 1 for listwise objectives.",
    )
    parser.add_argument(
        "--group-scope",
        choices=("should_rerank", "should_rerank_or_final_success", "all"),
        default="should_rerank",
        help="Episodes used by the group/listwise objective.",
    )
    parser.add_argument(
        "--group-target",
        choices=("soft_reward", "best_spl_onehot", "first_success_onehot"),
        default="soft_reward",
    )
    parser.add_argument("--group-temperature", type=float, default=0.25)
    parser.add_argument("--group-success-weight", type=float, default=1.0)
    parser.add_argument("--group-spl-weight", type=float, default=1.0)
    parser.add_argument("--group-fail-reward", type=float, default=-1.0)
    parser.add_argument("--group-late-penalty", type=float, default=0.25)
    parser.add_argument(
        "--group-final-success-bonus",
        type=float,
        default=1.0,
        help="Extra target reward for the final candidate when the original final is successful.",
    )
    parser.add_argument(
        "--score-transform",
        choices=("sigmoid", "minmax", "raw"),
        default="sigmoid",
        help="Transform raw linear scores before exporting candidate_score.",
    )
    parser.add_argument(
        "--gate-thresholds",
        default=",".join(baseline.format_float(value) for value in DEFAULT_GATE_THRESHOLDS),
        help="Comma-separated gate thresholds for evaluator bridge.",
    )
    parser.add_argument(
        "--taus",
        default=",".join(baseline.format_float(value) for value in DEFAULT_TAUS),
        help="Comma-separated final-stay taus for evaluator bridge.",
    )
    parser.add_argument(
        "--allow-change-final",
        default=",".join("true" if value else "false" for value in DEFAULT_ALLOW_CHANGE_FINAL),
        help="true, false, or comma-separated true,false grid for evaluator bridge.",
    )
    parser.add_argument("--default-gate-score", type=float, default=1.0)
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stderr progress logs. JSON manifest is still printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    endpoint_learning_dir = baseline.resolve_endpoint_learning_dir(experiment_dir, args.endpoint_learning_dir)
    candidate_csv = resolve_path(
        args.candidate_csv,
        endpoint_learning_dir / "candidate_groups" / "endpoint_candidates.csv",
    )
    episode_csv = resolve_path(
        args.episode_csv,
        endpoint_learning_dir / "candidate_groups" / "episode_groups.csv",
    )
    pair_csv = resolve_path(
        args.pair_csv,
        endpoint_learning_dir / "preference_pairs" / "preference_pairs.csv",
    )
    gate_score_csv = baseline.resolve_gate_score_csv(endpoint_learning_dir, args.gate_score_csv)
    output_dir = resolve_path(args.output_dir, endpoint_learning_dir / DEFAULT_OUTPUT_NAME)

    manifest = train_endpoint_preference_ranker(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        pair_csv=pair_csv,
        gate_score_csv=gate_score_csv,
        output_dir=output_dir,
        target_scope=args.target_scope,
        train_split=args.train_split,
        dev_split=args.dev_split,
        eval_split_filters=baseline.parse_string_list(args.eval_split),
        objective=args.objective,
        pair_weights=parse_pair_weights(args.pair_weights),
        max_pairs_per_type=args.max_pairs_per_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        random_state=args.random_state,
        group_loss_weight=args.group_loss_weight,
        group_scope=args.group_scope,
        group_target=args.group_target,
        group_temperature=args.group_temperature,
        group_success_weight=args.group_success_weight,
        group_spl_weight=args.group_spl_weight,
        group_fail_reward=args.group_fail_reward,
        group_late_penalty=args.group_late_penalty,
        group_final_success_bonus=args.group_final_success_bonus,
        score_transform=args.score_transform,
        gate_thresholds=tuple(baseline.parse_float_list(args.gate_thresholds)),
        taus=tuple(baseline.parse_float_list(args.taus)),
        allow_change_final_values=tuple(baseline.parse_bool_list(args.allow_change_final)),
        default_gate_score=args.default_gate_score,
        run_reranker_eval=not args.skip_reranker_eval,
        include_test_summary=args.include_test_summary,
        verbose=not args.quiet,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def train_endpoint_preference_ranker(
    candidate_csv: Path,
    episode_csv: Path,
    pair_csv: Path,
    gate_score_csv: Path | None,
    output_dir: Path,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    dev_split: str = DEFAULT_DEV_SPLIT,
    eval_split_filters: tuple[str, ...] = (DEFAULT_EVAL_SPLIT,),
    objective: str = "pairwise",
    pair_weights: dict[str, float] | None = None,
    max_pairs_per_type: int = 200_000,
    epochs: int = 40,
    batch_size: int = 8192,
    learning_rate: float = 0.03,
    l2: float = 1e-4,
    random_state: int = DEFAULT_RANDOM_STATE,
    group_loss_weight: float | None = None,
    group_scope: str = "should_rerank",
    group_target: str = "soft_reward",
    group_temperature: float = 0.25,
    group_success_weight: float = 1.0,
    group_spl_weight: float = 1.0,
    group_fail_reward: float = -1.0,
    group_late_penalty: float = 0.25,
    group_final_success_bonus: float = 1.0,
    score_transform: str = "sigmoid",
    gate_thresholds: tuple[float, ...] = DEFAULT_GATE_THRESHOLDS,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    allow_change_final_values: tuple[bool, ...] = DEFAULT_ALLOW_CHANGE_FINAL,
    default_gate_score: float = 1.0,
    run_reranker_eval: bool = True,
    include_test_summary: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if group_temperature <= 0:
        raise ValueError("--group-temperature must be positive")

    resolved_pair_weights = pair_weights or dict(DEFAULT_PAIR_WEIGHTS)
    resolved_group_weight = resolve_group_loss_weight(objective, group_loss_weight)

    log_progress(
        f"Starting preference ranker objective={objective}, output_dir={output_dir}",
        verbose=verbose,
        start_time=start_time,
    )
    log_progress(f"Loading candidates: {candidate_csv}", verbose=verbose, start_time=start_time)
    candidates = pd.read_csv(candidate_csv, low_memory=False)
    log_progress(f"Loading episodes: {episode_csv}", verbose=verbose, start_time=start_time)
    episodes = pd.read_csv(episode_csv, low_memory=False)
    candidates = candidates[candidates["target_scope"] == target_scope].copy()
    episodes = episodes[episodes["target_scope"] == target_scope].copy()
    if candidates.empty:
        raise ValueError(f"No candidates found for target_scope={target_scope!r}")
    if episodes.empty:
        raise ValueError(f"No episodes found for target_scope={target_scope!r}")
    log_progress(
        f"Loaded {len(candidates):,} candidates and {len(episodes):,} episodes for target_scope={target_scope}",
        verbose=verbose,
        start_time=start_time,
    )

    log_progress("Building vectorized candidate features", verbose=verbose, start_time=start_time)
    feature_frame = build_ranker_feature_frame_fast(candidates)
    baseline.validate_features(feature_frame)
    feature_frame["row_index"] = np.arange(len(feature_frame), dtype=int)

    train_mask = feature_frame["protocol_split"] == train_split
    dev_mask = feature_frame["protocol_split"] == dev_split
    if not train_mask.any():
        raise ValueError(f"No training rows with protocol_split={train_split!r}")
    if not dev_mask.any():
        raise ValueError(f"No dev rows with protocol_split={dev_split!r}")
    log_progress(
        f"Feature frame ready: {len(feature_frame):,} rows, train={int(train_mask.sum()):,}, "
        f"dev={int(dev_mask.sum()):,}",
        verbose=verbose,
        start_time=start_time,
    )

    log_progress("Imputing and scaling features", verbose=verbose, start_time=start_time)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(feature_frame.loc[train_mask, baseline.FEATURE_COLUMNS])
    scaler.fit(x_train)
    x_all = scaler.transform(imputer.transform(feature_frame[baseline.FEATURE_COLUMNS]))

    rng = np.random.default_rng(random_state)
    log_progress(
        f"Building pair arrays from {pair_csv} with max_pairs_per_type={max_pairs_per_type}",
        verbose=verbose,
        start_time=start_time,
    )
    pair_arrays = build_pair_arrays(
        pair_csv=pair_csv,
        feature_frame=feature_frame,
        train_split=train_split,
        target_scope=target_scope,
        pair_weights=resolved_pair_weights,
        max_pairs_per_type=max_pairs_per_type,
        rng=rng,
    )
    log_progress(
        f"Pair arrays ready: {len(pair_arrays.chosen_idx):,} usable pairs "
        f"({format_pair_summary(pair_arrays.summary)})",
        verbose=verbose,
        start_time=start_time,
    )
    log_progress(
        f"Building group targets scope={group_scope}, target={group_target}",
        verbose=verbose,
        start_time=start_time,
    )
    group_targets = build_group_targets(
        feature_frame=feature_frame,
        train_split=train_split,
        scope=group_scope,
        target_mode=group_target,
        temperature=group_temperature,
        success_weight=group_success_weight,
        spl_weight=group_spl_weight,
        fail_reward=group_fail_reward,
        late_penalty=group_late_penalty,
        final_success_bonus=group_final_success_bonus,
    )
    log_progress(f"Group targets ready: {len(group_targets):,}", verbose=verbose, start_time=start_time)
    if objective in {"pairwise", "pairwise_listwise"} and len(pair_arrays.chosen_idx) == 0:
        raise ValueError("Pairwise objective requested but no training pairs were built")
    if objective in {"listwise", "pairwise_listwise"} and not group_targets:
        raise ValueError("Listwise objective requested but no group targets were built")

    log_progress(
        f"Training linear preference model for {epochs} epochs (batch_size={batch_size}, lr={learning_rate})",
        verbose=verbose,
        start_time=start_time,
    )
    coef, intercept, training_curve = train_linear_preference_model(
        x_all=x_all,
        pair_arrays=pair_arrays,
        group_targets=group_targets,
        objective=objective,
        group_loss_weight=resolved_group_weight,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l2=l2,
        rng=rng,
        progress_fn=(
            (lambda message: log_progress(message, verbose=verbose, start_time=start_time))
            if verbose
            else None
        ),
    )

    log_progress("Scoring candidates and attaching gate scores", verbose=verbose, start_time=start_time)
    raw_scores = x_all @ coef + intercept
    feature_frame["candidate_score"] = transform_scores(raw_scores, score_transform)
    gate_score_map = baseline.load_gate_score_map(gate_score_csv) if gate_score_csv is not None else {}
    feature_frame["gate_score"] = feature_frame["episode_id"].map(gate_score_map).fillna(default_gate_score)

    summary_splits = [train_split, dev_split]
    if include_test_summary:
        summary_splits.extend(["test", "val_unseen"])
    summary_splits_tuple = tuple(dict.fromkeys(summary_splits))
    log_progress("Building candidate and pair summaries", verbose=verbose, start_time=start_time)
    summary_rows = baseline.build_summary_rows(feature_frame, split_filters=summary_splits_tuple)
    pair_summary_rows = build_pair_summary_rows_fast(
        pair_csv=pair_csv,
        score_frame=feature_frame,
        split_filters=summary_splits_tuple,
        target_scope=target_scope,
    )
    importance_rows = build_importance_rows(coef)

    features_csv = output_dir / "preference_ranker_features.csv"
    scores_csv = output_dir / "preference_ranker_scores.csv"
    summary_csv = output_dir / "preference_ranker_candidate_summary.csv"
    pair_summary_csv = output_dir / "preference_ranker_pair_summary.csv"
    pair_training_summary_csv = output_dir / "preference_pair_training_summary.csv"
    importance_csv = output_dir / "preference_ranker_feature_importance.csv"
    training_curve_csv = output_dir / "training_curve.csv"
    model_joblib = output_dir / "preference_ranker_model.joblib"
    model_json = output_dir / "preference_ranker_model.json"
    report_md = output_dir / "endpoint_preference_ranker_report.md"
    manifest_json = output_dir / "manifest.json"

    log_progress(f"Writing ranker outputs to {output_dir}", verbose=verbose, start_time=start_time)
    baseline.write_csv(features_csv, feature_frame, baseline.FEATURE_FIELDNAMES)
    baseline.write_csv(scores_csv, feature_frame, baseline.SCORE_FIELDNAMES)
    pd.DataFrame(summary_rows, columns=baseline.SUMMARY_FIELDNAMES).to_csv(summary_csv, index=False)
    pd.DataFrame(pair_summary_rows, columns=baseline.PAIR_SUMMARY_FIELDNAMES).to_csv(pair_summary_csv, index=False)
    pd.DataFrame(pair_arrays.summary).to_csv(pair_training_summary_csv, index=False)
    pd.DataFrame(importance_rows, columns=IMPORTANCE_FIELDNAMES).to_csv(importance_csv, index=False)
    pd.DataFrame(training_curve, columns=TRAINING_CURVE_COLUMNS).to_csv(training_curve_csv, index=False)

    model_payload = build_model_payload(
        coef=coef,
        intercept=intercept,
        imputer=imputer,
        scaler=scaler,
        target_scope=target_scope,
        train_split=train_split,
        dev_split=dev_split,
        objective=objective,
        pair_weights=resolved_pair_weights,
        max_pairs_per_type=max_pairs_per_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l2=l2,
        random_state=random_state,
        group_loss_weight=resolved_group_weight,
        group_scope=group_scope,
        group_target=group_target,
        group_temperature=group_temperature,
        group_success_weight=group_success_weight,
        group_spl_weight=group_spl_weight,
        group_fail_reward=group_fail_reward,
        group_late_penalty=group_late_penalty,
        group_final_success_bonus=group_final_success_bonus,
        score_transform=score_transform,
    )
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "coef": coef,
            "intercept": intercept,
            "imputer": imputer,
            "scaler": scaler,
            "feature_columns": baseline.FEATURE_COLUMNS,
            "model_payload": model_payload,
        },
        model_joblib,
    )
    baseline.write_json(model_json, model_payload)

    eval_manifest: dict[str, Any] | None = None
    if run_reranker_eval:
        log_progress(
            "Running evaluator bridge "
            f"(splits={','.join(eval_split_filters)}, gates={len(gate_thresholds)}, "
            f"taus={len(taus)}, allow_change_final={len(allow_change_final_values)})",
            verbose=verbose,
            start_time=start_time,
        )
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

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": target_scope,
        "train_split": train_split,
        "dev_split": dev_split,
        "eval_split_filters": list(eval_split_filters),
        "objective": objective,
        "pair_weights": resolved_pair_weights,
        "max_pairs_per_type": max_pairs_per_type,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2": l2,
            "random_state": random_state,
            "group_loss_weight": resolved_group_weight,
            "group_scope": group_scope,
            "group_target": group_target,
            "group_temperature": group_temperature,
            "group_success_weight": group_success_weight,
            "group_spl_weight": group_spl_weight,
            "group_fail_reward": group_fail_reward,
            "group_late_penalty": group_late_penalty,
            "group_final_success_bonus": group_final_success_bonus,
            "score_transform": score_transform,
        },
        "gate_thresholds": list(gate_thresholds),
        "taus": list(taus),
        "allow_change_final_values": list(allow_change_final_values),
        "default_gate_score": default_gate_score,
        "gate_score_csv": baseline.path_to_string(gate_score_csv) if gate_score_csv is not None else None,
        "include_test_summary": include_test_summary,
        "counts": {
            "episodes": int(feature_frame["episode_id"].nunique()),
            "train_candidates": int(train_mask.sum()),
            "dev_candidates": int(dev_mask.sum()),
            "candidates": int(len(feature_frame)),
            "training_pairs": int(len(pair_arrays.chosen_idx)),
            "group_targets": int(len(group_targets)),
            "features": len(baseline.FEATURE_COLUMNS),
        },
        "files": {
            "candidate_csv": baseline.path_to_string(candidate_csv),
            "episode_csv": baseline.path_to_string(episode_csv),
            "pair_csv": baseline.path_to_string(pair_csv),
            "preference_ranker_features_csv": baseline.path_to_string(features_csv),
            "preference_ranker_scores_csv": baseline.path_to_string(scores_csv),
            "preference_ranker_candidate_summary_csv": baseline.path_to_string(summary_csv),
            "preference_ranker_pair_summary_csv": baseline.path_to_string(pair_summary_csv),
            "preference_pair_training_summary_csv": baseline.path_to_string(pair_training_summary_csv),
            "preference_ranker_feature_importance_csv": baseline.path_to_string(importance_csv),
            "training_curve_csv": baseline.path_to_string(training_curve_csv),
            "preference_ranker_model_joblib": baseline.path_to_string(model_joblib),
            "preference_ranker_model_json": baseline.path_to_string(model_json),
            "endpoint_preference_ranker_report_md": baseline.path_to_string(report_md),
            "manifest_json": baseline.path_to_string(manifest_json),
        },
        "reranker_eval": eval_manifest,
    }
    baseline.write_json(manifest_json, manifest)
    write_report(
        path=report_md,
        manifest=manifest,
        summary_rows=summary_rows,
        pair_summary_rows=pair_summary_rows,
        pair_training_summary=pair_arrays.summary,
        importance_rows=importance_rows,
        training_curve=training_curve,
        eval_manifest=eval_manifest,
    )
    log_progress(f"Preference ranker done. scores={scores_csv}", verbose=verbose, start_time=start_time)
    return manifest


def build_ranker_feature_frame_fast(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    for column in baseline.NUMERIC_BASE_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    for column in baseline.BOOLEAN_FEATURE_COLUMNS:
        frame[column] = baseline.to_bool_series(frame.get(column, pd.Series(False, index=frame.index))).astype(int)
    for column in ["success_label", "final_success", "oracle_success", "should_rerank"]:
        frame[column] = baseline.to_bool_series(frame.get(column, pd.Series(False, index=frame.index))).astype(int)
    for column in ["spl_at_candidate", "reward", "candidate_step", "path_length_m"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    frame = frame.sort_values(["split", "protocol_split", "episode_id", "candidate_step"]).reset_index(drop=True)
    final_ref = build_final_reference(frame)
    frame = frame.merge(final_ref, on="episode_id", how="left", validate="many_to_one")
    candidate_count = frame.groupby("episode_id")["candidate_id"].transform("count").astype(float)

    frame["candidate_count"] = candidate_count
    frame["candidate_step_frac"] = safe_series_divide(frame["candidate_step"], frame["final_ref_step"])
    frame["steps_to_final"] = frame["final_ref_step"] - frame["candidate_step"]
    frame["steps_to_final_frac"] = safe_series_divide(frame["steps_to_final"], frame["final_ref_step"])
    frame["path_length_ratio_to_final"] = safe_series_divide(frame["path_length_m"], frame["final_ref_path_length_m"])
    frame["path_length_remaining_m"] = frame["final_ref_path_length_m"] - frame["path_length_m"]
    frame["stop_rank_frac"] = rank_fraction_fast(frame, "stop_prob", descending=True)
    frame["stop_margin_rank_frac"] = rank_fraction_fast(frame, "stop_margin_prob", descending=True)
    frame["selected_rank_frac"] = rank_fraction_fast(frame, "selected_prob", descending=True)
    frame["top_margin_rank_frac"] = rank_fraction_fast(frame, "top1_top2_margin", descending=True)
    frame["router_entropy_rank_frac"] = rank_fraction_fast(frame, "moe_router_entropy", descending=True)
    frame["path_length_rank_frac"] = rank_fraction_fast(frame, "path_length_m", descending=False)
    frame["stop_minus_final_stop"] = frame["stop_prob"] - frame["final_ref_stop_prob"]
    frame["stop_margin_minus_final_stop"] = frame["stop_margin_prob"] - frame["final_ref_stop_margin_prob"]
    frame["selected_minus_final_selected"] = frame["selected_prob"] - frame["final_ref_selected_prob"]
    frame["router_entropy_minus_final_router_entropy"] = (
        frame["moe_router_entropy"] - frame["final_ref_moe_router_entropy"]
    )
    frame["fuse_weight_minus_final_fuse_weight"] = frame["fuse_weight"] - frame["final_ref_fuse_weight"]

    output_columns = list(
        dict.fromkeys(
            baseline.ID_COLUMNS
            + [
                "success_label",
                "spl_at_candidate",
                "reward",
                "is_final",
                "final_success",
                "oracle_success",
                "should_rerank",
            ]
            + baseline.FEATURE_COLUMNS
        )
    )
    feature_frame = frame[output_columns].copy()
    for column in baseline.FEATURE_COLUMNS:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    return feature_frame


def build_final_reference(frame: pd.DataFrame) -> pd.DataFrame:
    sorted_frame = frame.sort_values(["episode_id", "candidate_step"])
    final_rows = sorted_frame[sorted_frame["is_final"].astype(bool)].drop_duplicates("episode_id", keep="last")
    missing_episode_ids = set(sorted_frame["episode_id"].astype(str)).difference(final_rows["episode_id"].astype(str))
    if missing_episode_ids:
        fallback = (
            sorted_frame[sorted_frame["episode_id"].astype(str).isin(missing_episode_ids)]
            .groupby("episode_id", sort=False)
            .tail(1)
        )
        final_rows = pd.concat([final_rows, fallback], ignore_index=True)
    columns = [
        "episode_id",
        "candidate_step",
        "path_length_m",
        "stop_prob",
        "stop_margin_prob",
        "selected_prob",
        "moe_router_entropy",
        "fuse_weight",
    ]
    final_ref = final_rows[columns].copy()
    return final_ref.rename(
        columns={
            "candidate_step": "final_ref_step",
            "path_length_m": "final_ref_path_length_m",
            "stop_prob": "final_ref_stop_prob",
            "stop_margin_prob": "final_ref_stop_margin_prob",
            "selected_prob": "final_ref_selected_prob",
            "moe_router_entropy": "final_ref_moe_router_entropy",
            "fuse_weight": "final_ref_fuse_weight",
        }
    )


def rank_fraction_fast(frame: pd.DataFrame, column: str, descending: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    ranks = values.groupby(frame["episode_id"]).rank(method="min", ascending=not descending)
    counts = frame.groupby("episode_id")["candidate_id"].transform("count").astype(float)
    return safe_series_divide(ranks, counts)


def safe_series_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def build_pair_arrays(
    pair_csv: Path,
    feature_frame: pd.DataFrame,
    train_split: str,
    target_scope: str,
    pair_weights: dict[str, float],
    max_pairs_per_type: int,
    rng: np.random.Generator,
) -> PairArrays:
    if not pair_csv.exists():
        raise FileNotFoundError(pair_csv)
    pairs = pd.read_csv(pair_csv, low_memory=False)
    pairs = pairs[pairs["target_scope"] == target_scope].copy()
    pairs = baseline.select_split(pairs, train_split)
    if pairs.empty:
        return PairArrays(
            chosen_idx=np.array([], dtype=np.int64),
            rejected_idx=np.array([], dtype=np.int64),
            weights=np.array([], dtype=np.float64),
            summary=[],
        )

    train_candidates = feature_frame[feature_frame["protocol_split"] == train_split]
    candidate_to_idx = {
        str(candidate_id): int(row_index)
        for candidate_id, row_index in zip(train_candidates["candidate_id"], train_candidates["row_index"])
    }

    chosen_parts: list[np.ndarray] = []
    rejected_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    summary: list[dict[str, Any]] = []
    for pair_type in sorted(pairs["pair_type"].dropna().unique()):
        weight = float(pair_weights.get(str(pair_type), 0.0))
        subset = pairs[pairs["pair_type"] == pair_type]
        original_pairs = int(len(subset))
        if weight <= 0 or subset.empty:
            summary.append(
                {
                    "pair_type": pair_type,
                    "available_pairs": original_pairs,
                    "sampled_pairs": 0,
                    "usable_pairs": 0,
                    "weight": weight,
                }
            )
            continue
        if max_pairs_per_type > 0 and len(subset) > max_pairs_per_type:
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            subset = subset.sample(n=max_pairs_per_type, random_state=random_state)

        chosen = subset["chosen_candidate_id"].astype(str).map(candidate_to_idx)
        rejected = subset["rejected_candidate_id"].astype(str).map(candidate_to_idx)
        valid = chosen.notna() & rejected.notna()
        chosen_idx = chosen[valid].astype(np.int64).to_numpy()
        rejected_idx = rejected[valid].astype(np.int64).to_numpy()
        if len(chosen_idx) > 0:
            chosen_parts.append(chosen_idx)
            rejected_parts.append(rejected_idx)
            weight_parts.append(np.full(len(chosen_idx), weight, dtype=np.float64))
        summary.append(
            {
                "pair_type": pair_type,
                "available_pairs": original_pairs,
                "sampled_pairs": int(len(subset)),
                "usable_pairs": int(len(chosen_idx)),
                "weight": weight,
            }
        )

    if not chosen_parts:
        return PairArrays(
            chosen_idx=np.array([], dtype=np.int64),
            rejected_idx=np.array([], dtype=np.int64),
            weights=np.array([], dtype=np.float64),
            summary=summary,
        )
    return PairArrays(
        chosen_idx=np.concatenate(chosen_parts),
        rejected_idx=np.concatenate(rejected_parts),
        weights=np.concatenate(weight_parts),
        summary=summary,
    )


def build_pair_summary_rows_fast(
    pair_csv: Path,
    score_frame: pd.DataFrame,
    split_filters: tuple[str, ...],
    target_scope: str,
) -> list[dict[str, Any]]:
    if not pair_csv.exists():
        return []
    pairs = pd.read_csv(pair_csv, low_memory=False)
    pairs = pairs[pairs["target_scope"] == target_scope].copy()
    if pairs.empty:
        return []
    score_map = score_frame.set_index("candidate_id")["candidate_score"]
    pairs["chosen_score"] = pairs["chosen_candidate_id"].map(score_map)
    pairs["rejected_score"] = pairs["rejected_candidate_id"].map(score_map)
    pairs = pairs[pairs["chosen_score"].notna() & pairs["rejected_score"].notna()].copy()
    if pairs.empty:
        return []
    pairs["score_margin"] = pd.to_numeric(pairs["chosen_score"], errors="coerce") - pd.to_numeric(
        pairs["rejected_score"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    for split_filter in split_filters:
        split_pairs = baseline.select_split(pairs, split_filter)
        if split_pairs.empty:
            continue
        pair_types = ["all", *sorted(split_pairs["pair_type"].dropna().unique().tolist())]
        for pair_type in pair_types:
            subset = split_pairs if pair_type == "all" else split_pairs[split_pairs["pair_type"] == pair_type]
            margins = pd.to_numeric(subset["score_margin"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(margins) == 0:
                continue
            strict_correct = int((margins > EPS).sum())
            ties = int((np.abs(margins) <= EPS).sum())
            rows.append(
                {
                    "split_filter": split_filter,
                    "pair_type": pair_type,
                    "pairs": int(len(margins)),
                    "accuracy": float((strict_correct + 0.5 * ties) / len(margins)),
                    "strict_accuracy": float(strict_correct / len(margins)),
                    "tie_rate": float(ties / len(margins)),
                    "mean_score_margin": float(np.mean(margins)),
                }
            )
    return rows


def build_group_targets(
    feature_frame: pd.DataFrame,
    train_split: str,
    scope: str,
    target_mode: str,
    temperature: float,
    success_weight: float,
    spl_weight: float,
    fail_reward: float,
    late_penalty: float,
    final_success_bonus: float,
) -> list[GroupTarget]:
    train_frame = feature_frame[feature_frame["protocol_split"] == train_split]
    targets: list[GroupTarget] = []
    for episode_id, group in train_frame.groupby("episode_id", sort=False):
        if not group_in_scope(group, scope):
            continue
        reward = build_group_reward(
            group=group,
            success_weight=success_weight,
            spl_weight=spl_weight,
            fail_reward=fail_reward,
            late_penalty=late_penalty,
            final_success_bonus=final_success_bonus,
        )
        if reward.size == 0 or np.all(~np.isfinite(reward)):
            continue
        if target_mode == "soft_reward":
            target = softmax(reward / temperature)
            target_kind = "soft_reward"
        elif target_mode == "best_spl_onehot":
            target = one_hot_best(group, reward, prefer_first_success=False)
            target_kind = "best_spl_onehot"
        elif target_mode == "first_success_onehot":
            target = one_hot_best(group, reward, prefer_first_success=True)
            target_kind = "first_success_onehot"
        else:
            raise ValueError(f"Unsupported group_target={target_mode!r}")
        if target.sum() <= 0:
            continue
        indices = group["row_index"].astype(np.int64).to_numpy()
        targets.append(
            GroupTarget(
                episode_id=str(episode_id),
                indices=indices,
                target=target / target.sum(),
                weight=1.0,
                target_kind=target_kind,
            )
        )
    return targets


def group_in_scope(group: pd.DataFrame, scope: str) -> bool:
    should_rerank = bool(int(group["should_rerank"].iloc[0]))
    final_success = bool(int(group["final_success"].iloc[0]))
    if scope == "should_rerank":
        return should_rerank
    if scope == "should_rerank_or_final_success":
        return should_rerank or final_success
    if scope == "all":
        return True
    raise ValueError(f"Unsupported group_scope={scope!r}")


def build_group_reward(
    group: pd.DataFrame,
    success_weight: float,
    spl_weight: float,
    fail_reward: float,
    late_penalty: float,
    final_success_bonus: float,
) -> np.ndarray:
    success = group["success_label"].astype(int).to_numpy()
    reward = np.full(len(group), fail_reward, dtype=np.float64)
    spl = pd.to_numeric(group["spl_at_candidate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    max_success_spl = float(np.max(spl[success == 1])) if np.any(success == 1) else 0.0
    spl_norm = spl / max(max_success_spl, EPS)
    reward[success == 1] = success_weight + spl_weight * spl_norm[success == 1]

    if np.any(success == 1) and late_penalty > 0:
        steps = pd.to_numeric(group["candidate_step"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        first_success_step = float(np.min(steps[success == 1]))
        final_step = max(float(np.max(steps)), EPS)
        late_frac = np.maximum(steps - first_success_step, 0.0) / final_step
        reward -= late_penalty * late_frac

    final_success = bool(int(group["final_success"].iloc[0]))
    if final_success and final_success_bonus != 0:
        is_final = group["is_final"].astype(int).to_numpy()
        reward[is_final == 1] += final_success_bonus
    return reward


def one_hot_best(group: pd.DataFrame, reward: np.ndarray, prefer_first_success: bool) -> np.ndarray:
    target = np.zeros(len(group), dtype=np.float64)
    success = group["success_label"].astype(int).to_numpy()
    final_success = bool(int(group["final_success"].iloc[0]))
    if final_success:
        final_positions = np.where(group["is_final"].astype(int).to_numpy() == 1)[0]
        if len(final_positions) > 0:
            target[int(final_positions[0])] = 1.0
            return target
    success_positions = np.where(success == 1)[0]
    if len(success_positions) == 0:
        best_position = int(np.argmax(reward))
    elif prefer_first_success:
        steps = pd.to_numeric(group["candidate_step"], errors="coerce").fillna(math.inf).to_numpy(dtype=np.float64)
        best_position = int(success_positions[np.argmin(steps[success_positions])])
    else:
        best_position = int(success_positions[np.argmax(reward[success_positions])])
    target[best_position] = 1.0
    return target


def train_linear_preference_model(
    x_all: np.ndarray,
    pair_arrays: PairArrays,
    group_targets: list[GroupTarget],
    objective: str,
    group_loss_weight: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    rng: np.random.Generator,
    progress_fn: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    feature_count = x_all.shape[1]
    coef = np.zeros(feature_count, dtype=np.float64)
    intercept = 0.0
    state = AdamState(m_w=np.zeros(feature_count, dtype=np.float64), v_w=np.zeros(feature_count, dtype=np.float64))
    training_curve: list[dict[str, Any]] = []

    use_pairs = objective in {"pairwise", "pairwise_listwise"} and len(pair_arrays.chosen_idx) > 0
    use_groups = objective in {"listwise", "pairwise_listwise"} and group_loss_weight > 0 and len(group_targets) > 0
    if not use_pairs and not use_groups:
        raise ValueError("No active loss terms. Check --objective and --group-loss-weight.")

    pair_count = len(pair_arrays.chosen_idx)
    progress_interval = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        pair_batches = 0
        if use_pairs:
            order = rng.permutation(pair_count)
            for start in range(0, pair_count, batch_size):
                batch = order[start : start + batch_size]
                grad_w, grad_b = pairwise_gradient(x_all, coef, pair_arrays, batch)
                if l2 > 0:
                    grad_w = grad_w + l2 * coef
                coef, intercept = adam_step(coef, intercept, grad_w, grad_b, state, learning_rate)
                pair_batches += 1

        if use_groups:
            grad_w, grad_b, _ = group_gradient(x_all, coef, intercept, group_targets, group_loss_weight)
            if l2 > 0:
                grad_w = grad_w + l2 * coef
            coef, intercept = adam_step(coef, intercept, grad_w, grad_b, state, learning_rate)

        pair_loss = pairwise_loss(x_all, coef, pair_arrays) if use_pairs else math.nan
        group_loss = group_objective_loss(x_all, coef, intercept, group_targets) if use_groups else math.nan
        objective_loss = nan_to_zero(pair_loss) + group_loss_weight * nan_to_zero(group_loss)
        training_curve.append(
            {
                "epoch": epoch,
                "pairwise_loss": pair_loss,
                "group_loss": group_loss,
                "objective_loss": objective_loss,
                "pair_batches": pair_batches,
                "groups": len(group_targets) if use_groups else 0,
                "learning_rate": learning_rate,
            }
        )
        if progress_fn is not None and (
            epoch == 1 or epoch == epochs or epoch % progress_interval == 0
        ):
            progress_fn(
                f"epoch {epoch}/{epochs}: pairwise_loss={format_loss(pair_loss)}, "
                f"group_loss={format_loss(group_loss)}, objective={format_loss(objective_loss)}, "
                f"pair_batches={pair_batches}"
            )
    return coef, intercept, training_curve


def pairwise_gradient(
    x_all: np.ndarray,
    coef: np.ndarray,
    pair_arrays: PairArrays,
    batch: np.ndarray,
) -> tuple[np.ndarray, float]:
    chosen = pair_arrays.chosen_idx[batch]
    rejected = pair_arrays.rejected_idx[batch]
    weights = pair_arrays.weights[batch]
    diff = x_all[chosen] - x_all[rejected]
    margins = diff @ coef
    deriv = -sigmoid(-margins)
    weighted = deriv * weights
    normalizer = max(float(np.sum(weights)), EPS)
    grad_w = (weighted[:, None] * diff).sum(axis=0) / normalizer
    return grad_w, 0.0


def pairwise_loss(x_all: np.ndarray, coef: np.ndarray, pair_arrays: PairArrays, max_eval_pairs: int = 100_000) -> float:
    if len(pair_arrays.chosen_idx) == 0:
        return math.nan
    if len(pair_arrays.chosen_idx) > max_eval_pairs:
        indices = np.linspace(0, len(pair_arrays.chosen_idx) - 1, max_eval_pairs, dtype=np.int64)
    else:
        indices = np.arange(len(pair_arrays.chosen_idx), dtype=np.int64)
    diff = x_all[pair_arrays.chosen_idx[indices]] - x_all[pair_arrays.rejected_idx[indices]]
    margins = diff @ coef
    weights = pair_arrays.weights[indices]
    losses = np.logaddexp(0.0, -margins)
    return float(np.sum(losses * weights) / max(float(np.sum(weights)), EPS))


def group_gradient(
    x_all: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    group_targets: list[GroupTarget],
    group_loss_weight: float,
) -> tuple[np.ndarray, float, float]:
    grad_w = np.zeros_like(coef)
    grad_b = 0.0
    loss_total = 0.0
    weight_total = 0.0
    for target in group_targets:
        scores = x_all[target.indices] @ coef + intercept
        probs = softmax(scores)
        delta = (probs - target.target) * target.weight
        grad_w += x_all[target.indices].T @ delta
        grad_b += float(delta.sum())
        loss_total += target.weight * float(-np.sum(target.target * np.log(probs + EPS)))
        weight_total += target.weight
    if weight_total <= 0:
        return grad_w, grad_b, math.nan
    scale = group_loss_weight / weight_total
    return grad_w * scale, grad_b * scale, loss_total / weight_total


def group_objective_loss(
    x_all: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    group_targets: list[GroupTarget],
) -> float:
    if not group_targets:
        return math.nan
    _, _, loss = group_gradient(
        x_all=x_all,
        coef=coef,
        intercept=intercept,
        group_targets=group_targets,
        group_loss_weight=1.0,
    )
    return loss


def adam_step(
    coef: np.ndarray,
    intercept: float,
    grad_w: np.ndarray,
    grad_b: float,
    state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> tuple[np.ndarray, float]:
    state.t += 1
    state.m_w = beta1 * state.m_w + (1.0 - beta1) * grad_w
    state.v_w = beta2 * state.v_w + (1.0 - beta2) * (grad_w * grad_w)
    state.m_b = beta1 * state.m_b + (1.0 - beta1) * grad_b
    state.v_b = beta2 * state.v_b + (1.0 - beta2) * (grad_b * grad_b)
    m_w_hat = state.m_w / (1.0 - beta1**state.t)
    v_w_hat = state.v_w / (1.0 - beta2**state.t)
    m_b_hat = state.m_b / (1.0 - beta1**state.t)
    v_b_hat = state.v_b / (1.0 - beta2**state.t)
    coef = coef - learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
    intercept = intercept - learning_rate * m_b_hat / (math.sqrt(v_b_hat) + 1e-8)
    return coef, intercept


def build_importance_rows(coef: np.ndarray) -> list[dict[str, Any]]:
    rows = [
        {
            "feature": feature,
            "coefficient": float(value),
            "abs_coefficient": float(abs(value)),
        }
        for feature, value in zip(baseline.FEATURE_COLUMNS, coef)
    ]
    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)


def build_model_payload(
    coef: np.ndarray,
    intercept: float,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    target_scope: str,
    train_split: str,
    dev_split: str,
    objective: str,
    pair_weights: dict[str, float],
    max_pairs_per_type: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    random_state: int,
    group_loss_weight: float,
    group_scope: str,
    group_target: str,
    group_temperature: float,
    group_success_weight: float,
    group_spl_weight: float,
    group_fail_reward: float,
    group_late_penalty: float,
    group_final_success_bonus: float,
    score_transform: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target_scope": target_scope,
        "train_split": train_split,
        "dev_split": dev_split,
        "feature_columns": baseline.FEATURE_COLUMNS,
        "model": {
            "type": "linear_preference_ranker",
            "objective": objective,
            "pair_loss": "bradley_terry_logistic",
            "group_loss": "episode_softmax_cross_entropy",
            "coef": [float(value) for value in coef],
            "intercept": float(intercept),
            "score_transform": score_transform,
        },
        "pair_weights": pair_weights,
        "max_pairs_per_type": max_pairs_per_type,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2": l2,
            "random_state": random_state,
            "group_loss_weight": group_loss_weight,
            "group_scope": group_scope,
            "group_target": group_target,
            "group_temperature": group_temperature,
            "group_success_weight": group_success_weight,
            "group_spl_weight": group_spl_weight,
            "group_fail_reward": group_fail_reward,
            "group_late_penalty": group_late_penalty,
            "group_final_success_bonus": group_final_success_bonus,
        },
        "imputer": {
            "type": "SimpleImputer",
            "strategy": "median",
            "statistics": [baseline.none_if_nan(value) for value in imputer.statistics_],
        },
        "scaler": {
            "type": "StandardScaler",
            "mean": [baseline.none_if_nan(value) for value in scaler.mean_],
            "scale": [baseline.none_if_nan(value) for value in scaler.scale_],
        },
    }


def write_report(
    path: Path,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    pair_training_summary: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    training_curve: list[dict[str, Any]],
    eval_manifest: dict[str, Any] | None,
) -> None:
    eval_rows = baseline.read_eval_summary(eval_manifest)
    final_curve = training_curve[-1] if training_curve else {}
    lines = [
        "# Endpoint Preference Ranker Report",
        "",
        "This report is generated by `scripts/analysis/train_endpoint_preference_ranker.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- train_split: `{manifest['train_split']}`",
        f"- dev_split: `{manifest['dev_split']}`",
        f"- objective: `{manifest['objective']}`",
        f"- pair_weights: `{format_pair_weights(manifest['pair_weights'])}`",
        f"- group_loss_weight: `{manifest['training']['group_loss_weight']}`",
        f"- group_scope: `{manifest['training']['group_scope']}`",
        f"- score_transform: `{manifest['training']['score_transform']}`",
        f"- gate_score_csv: `{manifest['gate_score_csv']}`",
        "",
        "## Training",
        "",
        baseline.markdown_table(
            ["epochs", "pairwise_loss", "group_loss", "objective_loss", "training_pairs", "group_targets"],
            [
                [
                    manifest["training"]["epochs"],
                    baseline.fmt(final_curve.get("pairwise_loss")),
                    baseline.fmt(final_curve.get("group_loss")),
                    baseline.fmt(final_curve.get("objective_loss")),
                    manifest["counts"]["training_pairs"],
                    manifest["counts"]["group_targets"],
                ]
            ],
        ),
        "",
        "## Candidate Metrics",
        "",
        baseline.markdown_table(
            [
                "split",
                "candidates",
                "episodes",
                "positive_rate",
                "roc_auc",
                "average_precision",
                "top1_success",
                "top1_harm",
            ],
            [
                [
                    row["split_filter"],
                    row["items"],
                    row["episodes"],
                    baseline.pct(row["positive_rate"]),
                    baseline.fmt(row["roc_auc"]),
                    baseline.fmt(row["average_precision"]),
                    baseline.pct(row["top1_success_rate"]),
                    baseline.pct(row["top1_harm_rate"]),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Pair Agreement",
        "",
        baseline.markdown_table(
            ["split", "pair_type", "pairs", "accuracy", "strict_accuracy", "mean_margin"],
            [
                [
                    row["split_filter"],
                    row["pair_type"],
                    row["pairs"],
                    baseline.pct(row["accuracy"]),
                    baseline.pct(row["strict_accuracy"]),
                    baseline.fmt(row["mean_score_margin"]),
                ]
                for row in pair_summary_rows
                if row["split_filter"] in {manifest["dev_split"], manifest["train_split"]}
            ],
        ),
        "",
        "## Pair Sampling",
        "",
        baseline.markdown_table(
            ["pair_type", "available", "sampled", "usable", "weight"],
            [
                [
                    row["pair_type"],
                    row["available_pairs"],
                    row["sampled_pairs"],
                    row["usable_pairs"],
                    baseline.fmt(row["weight"]),
                ]
                for row in pair_training_summary
            ],
        ),
        "",
        "## Evaluator Bridge",
        "",
        baseline.markdown_table(
            [
                "gate",
                "tau",
                "allow",
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
                    baseline.fmt(row["gate_threshold"]),
                    baseline.fmt(row["tau"]),
                    str(row["allow_change_final"]).lower(),
                    baseline.pct(row["SR"]),
                    baseline.pct(row["delta_SR"]),
                    baseline.pct(row["SPL"]),
                    baseline.pct(row["delta_SPL"]),
                    baseline.pct(row["recovery_rate"]),
                    baseline.pct(row["harm_rate"]),
                    baseline.pct(row["changed_endpoint_rate"]),
                ]
                for row in eval_rows[:10]
            ],
        ),
        "",
        "## Top Coefficients",
        "",
        baseline.markdown_table(
            ["feature", "coef", "abs_coef"],
            [
                [row["feature"], baseline.fmt(row["coefficient"]), baseline.fmt(row["abs_coefficient"])]
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


def resolve_group_loss_weight(objective: str, group_loss_weight: float | None) -> float:
    if group_loss_weight is not None:
        return float(group_loss_weight)
    if objective == "pairwise":
        return 0.0
    return 1.0


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


def format_loss(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(parsed):
        return "nan"
    return f"{parsed:.4f}"


def format_pair_summary(summary: list[dict[str, Any]]) -> str:
    if not summary:
        return "no pair summary"
    parts = []
    for row in summary:
        pair_type = row.get("pair_type")
        usable = row.get("usable_pairs")
        weight = row.get("weight")
        parts.append(f"{pair_type}:{usable}@w{weight}")
    return ", ".join(parts)


def transform_scores(raw_scores: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return raw_scores
    if transform == "sigmoid":
        return sigmoid(raw_scores)
    if transform == "minmax":
        min_value = float(np.min(raw_scores))
        max_value = float(np.max(raw_scores))
        if max_value - min_value <= EPS:
            return np.zeros_like(raw_scores)
        return (raw_scores - min_value) / (max_value - min_value)
    raise ValueError(f"Unsupported score_transform={transform!r}")


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, -1e9)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / max(float(np.sum(exp_values)), EPS)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def nan_to_zero(value: float) -> float:
    return 0.0 if value is None or not math.isfinite(float(value)) else float(value)


def parse_pair_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not text:
        return weights
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"Pair weight must be pair_type=weight, got {stripped!r}")
        key, value = stripped.split("=", 1)
        weights[key.strip()] = float(value)
    return weights


def format_pair_weights(weights: dict[str, float]) -> str:
    return ",".join(f"{key}={value:g}" for key, value in weights.items())


def resolve_path(value: str | None, default: Path) -> Path:
    return Path(value).resolve() if value else default.resolve()


if __name__ == "__main__":
    main()
