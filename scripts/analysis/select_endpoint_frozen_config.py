#!/usr/bin/env python3
"""Select a frozen endpoint gate + ranker inference config on dev only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_endpoint_reranker as reranker_eval  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_frozen_gate_ranker.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_OUTPUT_NAME = "frozen_gate_ranker"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_SELECTION_SPLIT = "dev"
DEFAULT_DIAGNOSTIC_SPLITS = ("train", "dev")
DEFAULT_GATE_THRESHOLDS = (
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
)
DEFAULT_TAUS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
DEFAULT_ALLOW_CHANGE_FINAL = (True, False)
EPS = 1e-12

SELECTION_GRID_COLUMNS = [
    "selection_rank",
    "selected",
    "eligible",
    "selection_pool",
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "gate_threshold",
    "tau",
    "allow_change_final",
    "items",
    "final_SR",
    "SR",
    "delta_SR",
    "final_SPL",
    "SPL",
    "delta_SPL",
    "recovery_rate",
    "harm_rate",
    "net_recovery_rate",
    "changed_endpoint_rate",
    "gate_pass_rate",
    "gate_precision",
    "gate_recall",
    "overshoot_items",
    "overshoot_recovery_rate",
    "final_success_harm_rate",
]

FAILURE_ITEM_COLUMNS = [
    "episode_id",
    "internal_item_id",
    "saved_instr_id",
    "final_failure_bucket",
    "should_rerank",
    "final_success",
    "selected_success",
    "oracle_success",
    "recovered",
    "harmed",
    "gate_score",
    "gate_passed",
    "selection_reason",
    "failure_slice",
    "final_candidate_id",
    "selected_candidate_id",
    "best_candidate_id",
    "final_step",
    "selected_step",
    "best_step",
    "selected_changed",
    "final_score",
    "selected_score",
    "best_score",
    "score_margin_over_final",
    "best_success",
    "final_spl",
    "selected_spl",
    "best_spl",
    "nearest_endpoint_spl",
    "final_distance_m",
    "selected_distance_m",
    "best_distance_m",
]

SLICE_SUMMARY_COLUMNS = [
    "failure_slice",
    "items",
    "rate",
    "should_rerank_items",
    "final_success_items",
    "recovered_items",
    "harmed_items",
    "changed_items",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a gate + ranker config using train/dev-only endpoint learning outputs.",
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
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <endpoint-learning-dir>/frozen_gate_ranker.",
    )
    parser.add_argument(
        "--gate-model-json",
        default=None,
        help="Gate model JSON metadata. Defaults to <endpoint-learning-dir>/gate_baseline/gate_model.json.",
    )
    parser.add_argument(
        "--ranker-model-json",
        default=None,
        help="Ranker model JSON metadata. Defaults to <endpoint-learning-dir>/ranker_baseline/ranker_model.json.",
    )
    parser.add_argument(
        "--gate-model-joblib",
        default=None,
        help="Gate model joblib path. Defaults to <endpoint-learning-dir>/gate_baseline/gate_model.joblib.",
    )
    parser.add_argument(
        "--ranker-model-joblib",
        default=None,
        help="Ranker model joblib path. Defaults to <endpoint-learning-dir>/ranker_baseline/ranker_model.joblib.",
    )
    parser.add_argument(
        "--target-scope",
        default=DEFAULT_TARGET_SCOPE,
        help="Target scope to evaluate. Defaults to official.",
    )
    parser.add_argument(
        "--selection-split",
        default=DEFAULT_SELECTION_SPLIT,
        help="Split used for config selection. Defaults to dev.",
    )
    parser.add_argument(
        "--diagnostic-splits",
        default=",".join(DEFAULT_DIAGNOSTIC_SPLITS),
        help="Comma-separated train/dev diagnostics for the selected config. Defaults to train,dev.",
    )
    parser.add_argument(
        "--gate-thresholds",
        default=",".join(format_float(value) for value in DEFAULT_GATE_THRESHOLDS),
        help="Comma-separated gate thresholds to scan on the selection split.",
    )
    parser.add_argument(
        "--taus",
        default=",".join(format_float(value) for value in DEFAULT_TAUS),
        help="Comma-separated final-stay tau values to scan on the selection split.",
    )
    parser.add_argument(
        "--allow-change-final",
        default=",".join("true" if value else "false" for value in DEFAULT_ALLOW_CHANGE_FINAL),
        help="true, false, or comma-separated true,false grid.",
    )
    parser.add_argument(
        "--min-delta-sr",
        type=float,
        default=0.0,
        help="Minimum dev delta_SR required for an eligible config. Defaults to 0.",
    )
    parser.add_argument(
        "--min-delta-spl",
        type=float,
        default=0.0,
        help="Minimum dev delta_SPL required for an eligible config. Defaults to 0.",
    )
    parser.add_argument(
        "--max-harm-rate",
        type=float,
        default=0.01,
        help="Maximum dev harm_rate for an eligible config. Defaults to 0.01.",
    )
    parser.add_argument(
        "--allow-frozen-final-fallback",
        action="store_true",
        help=(
            "Permit allow_change_final=false to be selected when no change-enabled config "
            "passes the constraints."
        ),
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
    output_dir = resolve_path(args.output_dir, endpoint_learning_dir / DEFAULT_OUTPUT_NAME)
    gate_model_json = resolve_path(
        args.gate_model_json,
        endpoint_learning_dir / "gate_baseline" / "gate_model.json",
    )
    ranker_model_json = resolve_path(
        args.ranker_model_json,
        endpoint_learning_dir / "ranker_baseline" / "ranker_model.json",
    )
    gate_model_joblib = resolve_path(
        args.gate_model_joblib,
        endpoint_learning_dir / "gate_baseline" / "gate_model.joblib",
    )
    ranker_model_joblib = resolve_path(
        args.ranker_model_joblib,
        endpoint_learning_dir / "ranker_baseline" / "ranker_model.joblib",
    )

    manifest = select_endpoint_frozen_config(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=score_csv,
        output_dir=output_dir,
        gate_model_json=gate_model_json,
        ranker_model_json=ranker_model_json,
        gate_model_joblib=gate_model_joblib,
        ranker_model_joblib=ranker_model_joblib,
        target_scope=args.target_scope,
        selection_split=args.selection_split,
        diagnostic_splits=parse_string_list(args.diagnostic_splits),
        gate_thresholds=tuple(parse_float_list(args.gate_thresholds)),
        taus=tuple(parse_float_list(args.taus)),
        allow_change_final_values=tuple(parse_bool_list(args.allow_change_final)),
        min_delta_sr=args.min_delta_sr,
        min_delta_spl=args.min_delta_spl,
        max_harm_rate=args.max_harm_rate,
        allow_frozen_final_fallback=args.allow_frozen_final_fallback,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def select_endpoint_frozen_config(
    candidate_csv: Path,
    episode_csv: Path,
    score_csv: Path,
    output_dir: Path,
    gate_model_json: Path,
    ranker_model_json: Path,
    gate_model_joblib: Path,
    ranker_model_joblib: Path,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    selection_split: str = DEFAULT_SELECTION_SPLIT,
    diagnostic_splits: tuple[str, ...] = DEFAULT_DIAGNOSTIC_SPLITS,
    gate_thresholds: tuple[float, ...] = DEFAULT_GATE_THRESHOLDS,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    allow_change_final_values: tuple[bool, ...] = DEFAULT_ALLOW_CHANGE_FINAL,
    min_delta_sr: float = 0.0,
    min_delta_spl: float = 0.0,
    max_harm_rate: float = 0.01,
    allow_frozen_final_fallback: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_grid_dir = output_dir / "dev_grid_eval_protocol"
    dev_eval_manifest = reranker_eval.evaluate_endpoint_reranker(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=score_csv,
        output_dir=dev_grid_dir,
        target_scope=target_scope,
        split_filters=(selection_split,),
        gate_thresholds=gate_thresholds,
        taus=taus,
        allow_change_final_values=allow_change_final_values,
        candidate_score_column="candidate_score",
        gate_score_column="gate_score",
        default_gate_score=1.0,
    )

    summary = pd.read_csv(resolve_manifest_path(dev_eval_manifest["files"]["summary_csv"]))
    if summary.empty:
        raise ValueError("Dev selection grid produced no summary rows")
    summary = normalize_summary_frame(summary)
    selection_grid, selected_row = choose_config(
        summary=summary,
        min_delta_sr=min_delta_sr,
        min_delta_spl=min_delta_spl,
        max_harm_rate=max_harm_rate,
        allow_frozen_final_fallback=allow_frozen_final_fallback,
    )

    selected_config = {
        "gate_threshold": float(selected_row["gate_threshold"]),
        "tau": float(selected_row["tau"]),
        "allow_change_final": bool(selected_row["allow_change_final"]),
    }

    dev_items = pd.read_csv(resolve_manifest_path(dev_eval_manifest["files"]["items_csv"]))
    selected_items = filter_items_for_config(dev_items, selected_config)
    selected_items = add_failure_slice_columns(selected_items, candidate_csv)
    slice_summary = build_slice_summary(selected_items)

    frozen_eval_dir = output_dir / "frozen_eval_protocol"
    frozen_eval_manifest = reranker_eval.evaluate_endpoint_reranker(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=score_csv,
        output_dir=frozen_eval_dir,
        target_scope=target_scope,
        split_filters=diagnostic_splits,
        gate_thresholds=(selected_config["gate_threshold"],),
        taus=(selected_config["tau"],),
        allow_change_final_values=(selected_config["allow_change_final"],),
        candidate_score_column="candidate_score",
        gate_score_column="gate_score",
        default_gate_score=1.0,
    )
    frozen_summary = pd.read_csv(resolve_manifest_path(frozen_eval_manifest["files"]["summary_csv"]))

    selection_grid_csv = output_dir / "dev_selection_grid.csv"
    selected_items_csv = output_dir / "dev_selected_items.csv"
    slice_summary_csv = output_dir / "failure_slice_summary.csv"
    frozen_config_json = output_dir / "frozen_config.json"
    report_md = output_dir / "dev_selection_report.md"
    manifest_json = output_dir / "manifest.json"

    selection_grid.to_csv(selection_grid_csv, index=False, columns=SELECTION_GRID_COLUMNS)
    selected_items.to_csv(selected_items_csv, index=False, columns=FAILURE_ITEM_COLUMNS)
    slice_summary.to_csv(slice_summary_csv, index=False, columns=SLICE_SUMMARY_COLUMNS)

    frozen_config = build_frozen_config(
        selected_row=selected_row,
        selected_config=selected_config,
        target_scope=target_scope,
        selection_split=selection_split,
        diagnostic_splits=diagnostic_splits,
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=score_csv,
        gate_model_json=gate_model_json,
        ranker_model_json=ranker_model_json,
        gate_model_joblib=gate_model_joblib,
        ranker_model_joblib=ranker_model_joblib,
        min_delta_sr=min_delta_sr,
        min_delta_spl=min_delta_spl,
        max_harm_rate=max_harm_rate,
        allow_frozen_final_fallback=allow_frozen_final_fallback,
    )
    write_json(frozen_config_json, frozen_config)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": target_scope,
        "selection_split": selection_split,
        "diagnostic_splits": list(diagnostic_splits),
        "selection_constraints": frozen_config["selection_constraints"],
        "selected_config": selected_config,
        "val_unseen_used_for_selection": False,
        "files": {
            "candidate_csv": path_to_string(candidate_csv),
            "episode_csv": path_to_string(episode_csv),
            "score_csv": path_to_string(score_csv),
            "dev_selection_grid_csv": path_to_string(selection_grid_csv),
            "dev_selected_items_csv": path_to_string(selected_items_csv),
            "failure_slice_summary_csv": path_to_string(slice_summary_csv),
            "frozen_config_json": path_to_string(frozen_config_json),
            "dev_selection_report_md": path_to_string(report_md),
            "manifest_json": path_to_string(manifest_json),
        },
        "dev_grid_eval": dev_eval_manifest,
        "frozen_eval": frozen_eval_manifest,
    }
    write_json(manifest_json, manifest)
    write_report(
        path=report_md,
        manifest=manifest,
        frozen_config=frozen_config,
        selection_grid=selection_grid,
        slice_summary=slice_summary,
        frozen_summary=frozen_summary,
    )
    return manifest


def choose_config(
    summary: pd.DataFrame,
    min_delta_sr: float,
    min_delta_spl: float,
    max_harm_rate: float,
    allow_frozen_final_fallback: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    grid = summary.copy()
    grid["eligible"] = (
        (grid["delta_SR"] >= min_delta_sr - EPS)
        & (grid["delta_SPL"] >= min_delta_spl - EPS)
        & (grid["harm_rate"] <= max_harm_rate + EPS)
    )
    grid["selection_pool"] = ""
    grid["selected"] = False
    grid["selection_rank"] = pd.NA

    eligible = grid[grid["eligible"]].copy()
    if eligible.empty:
        raise ValueError(
            "No config passed the selection constraints. "
            f"min_delta_sr={min_delta_sr}, min_delta_spl={min_delta_spl}, max_harm_rate={max_harm_rate}"
        )

    change_enabled = eligible[eligible["allow_change_final"] == True].copy()  # noqa: E712
    if not change_enabled.empty:
        pool = change_enabled
        pool_name = "eligible_change_enabled"
    elif allow_frozen_final_fallback:
        pool = eligible
        pool_name = "eligible_with_frozen_final_fallback"
    else:
        raise ValueError(
            "No allow_change_final=true config passed the constraints. "
            "Use --allow-frozen-final-fallback to permit selecting the frozen-final baseline."
        )

    pool = pool.sort_values(
        [
            "delta_SR",
            "net_recovery_rate",
            "harm_rate",
            "changed_endpoint_rate",
            "gate_pass_rate",
            "delta_SPL",
            "gate_threshold",
            "tau",
        ],
        ascending=[False, False, True, True, True, False, False, False],
        na_position="last",
    ).reset_index()
    pool["selection_rank"] = range(1, len(pool) + 1)

    rank_by_index = dict(zip(pool["index"], pool["selection_rank"]))
    grid.loc[pool["index"], "selection_pool"] = pool_name
    grid.loc[pool["index"], "selection_rank"] = grid.loc[pool["index"]].index.map(rank_by_index)

    selected_original_index = int(pool.iloc[0]["index"])
    grid.loc[selected_original_index, "selected"] = True

    grid = grid.sort_values(
        ["selected", "selection_rank", "delta_SR", "harm_rate", "changed_endpoint_rate"],
        ascending=[False, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    selected_row = grid[grid["selected"]].iloc[0]
    return grid, selected_row


def normalize_summary_frame(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["allow_change_final"] = frame["allow_change_final"].map(parse_bool_value)
    numeric_columns = [
        "gate_threshold",
        "tau",
        "items",
        "final_SR",
        "SR",
        "delta_SR",
        "final_SPL",
        "SPL",
        "delta_SPL",
        "recovery_rate",
        "harm_rate",
        "net_recovery_rate",
        "changed_endpoint_rate",
        "gate_pass_rate",
        "gate_precision",
        "gate_recall",
        "overshoot_items",
        "overshoot_recovery_rate",
        "final_success_harm_rate",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def filter_items_for_config(items: pd.DataFrame, selected_config: dict[str, Any]) -> pd.DataFrame:
    frame = items.copy()
    frame["allow_change_final"] = frame["allow_change_final"].map(parse_bool_value)
    frame["gate_threshold"] = pd.to_numeric(frame["gate_threshold"], errors="coerce")
    frame["tau"] = pd.to_numeric(frame["tau"], errors="coerce")
    mask = (
        (abs(frame["gate_threshold"] - selected_config["gate_threshold"]) <= EPS)
        & (abs(frame["tau"] - selected_config["tau"]) <= EPS)
        & (frame["allow_change_final"] == selected_config["allow_change_final"])
    )
    selected = frame[mask].copy()
    if selected.empty:
        raise ValueError(f"No item rows matched selected config: {selected_config}")
    return selected


def add_failure_slice_columns(items: pd.DataFrame, candidate_csv: Path) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_csv)
    candidates = candidates.set_index("candidate_id", drop=False)
    best_success_map = candidates["success_label"].map(parse_bool_value).to_dict()
    best_spl_map = pd.to_numeric(candidates["spl_at_candidate"], errors="coerce").to_dict()
    best_distance_map = pd.to_numeric(candidates["distance_to_goal_m"], errors="coerce").to_dict()

    frame = items.copy()
    for column in [
        "should_rerank",
        "final_success",
        "selected_success",
        "oracle_success",
        "recovered",
        "harmed",
        "gate_passed",
        "selected_changed",
    ]:
        frame[column] = frame[column].map(parse_bool_value)
    frame["best_success"] = frame["best_candidate_id"].map(best_success_map)
    frame["best_spl"] = frame["best_candidate_id"].map(best_spl_map)
    frame["best_distance_m"] = frame["best_candidate_id"].map(best_distance_map)
    frame["failure_slice"] = frame.apply(classify_failure_slice, axis=1)
    return frame.sort_values(["failure_slice", "episode_id"]).reset_index(drop=True)


def classify_failure_slice(row: pd.Series) -> str:
    if bool(row.get("recovered")):
        return "recovered"
    if bool(row.get("harmed")):
        return "final_success_harmed"
    if bool(row.get("should_rerank")):
        if not bool(row.get("gate_passed")):
            return "should_rerank_gate_rejected"
        if str(row.get("selection_reason")) == "final_within_tau":
            if bool(row.get("best_success")):
                return "should_rerank_tau_blocked_success"
            return "should_rerank_ranker_best_failed_or_final"
        if bool(row.get("selected_changed")) and not bool(row.get("selected_success")):
            return "should_rerank_ranker_best_failed"
        return "should_rerank_unrecovered_other"
    if bool(row.get("final_success")):
        if bool(row.get("selected_changed")):
            return "final_success_changed_safe"
        return "final_success_kept"
    if bool(row.get("selected_changed")):
        return "final_failure_changed_unrecovered"
    return "final_failure_kept"


def build_slice_summary(items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(items)
    for name, group in items.groupby("failure_slice", sort=True):
        rows.append(
            {
                "failure_slice": name,
                "items": int(len(group)),
                "rate": safe_divide(len(group), total),
                "should_rerank_items": int(group["should_rerank"].fillna(False).astype(bool).sum()),
                "final_success_items": int(group["final_success"].fillna(False).astype(bool).sum()),
                "recovered_items": int(group["recovered"].fillna(False).astype(bool).sum()),
                "harmed_items": int(group["harmed"].fillna(False).astype(bool).sum()),
                "changed_items": int(group["selected_changed"].fillna(False).astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows, columns=SLICE_SUMMARY_COLUMNS)


def build_frozen_config(
    selected_row: pd.Series,
    selected_config: dict[str, Any],
    target_scope: str,
    selection_split: str,
    diagnostic_splits: tuple[str, ...],
    candidate_csv: Path,
    episode_csv: Path,
    score_csv: Path,
    gate_model_json: Path,
    ranker_model_json: Path,
    gate_model_joblib: Path,
    ranker_model_joblib: Path,
    min_delta_sr: float,
    min_delta_spl: float,
    max_harm_rate: float,
    allow_frozen_final_fallback: bool,
) -> dict[str, Any]:
    gate_model_payload = read_json_if_exists(gate_model_json)
    ranker_model_payload = read_json_if_exists(ranker_model_json)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen_on_train_dev",
        "target_scope": target_scope,
        "selection_split": selection_split,
        "diagnostic_splits": list(diagnostic_splits),
        "val_unseen_used_for_selection": False,
        "selected_config": selected_config,
        "selection_constraints": {
            "min_delta_sr": min_delta_sr,
            "min_delta_spl": min_delta_spl,
            "max_harm_rate": max_harm_rate,
            "allow_frozen_final_fallback": allow_frozen_final_fallback,
            "selection_order": [
                "max delta_SR",
                "max net_recovery_rate",
                "min harm_rate",
                "min changed_endpoint_rate",
                "min gate_pass_rate",
                "max delta_SPL",
                "max gate_threshold",
                "max tau",
            ],
        },
        "dev_metrics": series_to_jsonable(selected_row),
        "models": {
            "gate": {
                "model_json": path_to_string(gate_model_json),
                "model_joblib": path_to_string(gate_model_joblib),
                "schema_version": gate_model_payload.get("schema_version"),
                "classifier": gate_model_payload.get("classifier", {}).get("type"),
                "train_split": gate_model_payload.get("train_split"),
                "dev_split": gate_model_payload.get("dev_split"),
            },
            "ranker": {
                "model_json": path_to_string(ranker_model_json),
                "model_joblib": path_to_string(ranker_model_joblib),
                "schema_version": ranker_model_payload.get("schema_version"),
                "classifier": ranker_model_payload.get("classifier", {}).get("type"),
                "loss": ranker_model_payload.get("classifier", {}).get("loss"),
                "train_split": ranker_model_payload.get("train_split"),
                "dev_split": ranker_model_payload.get("dev_split"),
            },
        },
        "inputs": {
            "candidate_csv": path_to_string(candidate_csv),
            "episode_csv": path_to_string(episode_csv),
            "score_csv": path_to_string(score_csv),
        },
        "inference_rule": (
            "if gate_score < gate_threshold choose final; otherwise choose argmax candidate_score; "
            "choose final when best_score <= final_score + tau"
        ),
    }


def write_report(
    path: Path,
    manifest: dict[str, Any],
    frozen_config: dict[str, Any],
    selection_grid: pd.DataFrame,
    slice_summary: pd.DataFrame,
    frozen_summary: pd.DataFrame,
) -> None:
    selected = frozen_config["selected_config"]
    dev_metrics = frozen_config["dev_metrics"]
    top_rows = selection_grid[selection_grid["selection_pool"].astype(str) != ""].head(12)
    lines = [
        "# Endpoint Gate + Ranker Frozen Config Report",
        "",
        "This report is generated by `scripts/analysis/select_endpoint_frozen_config.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- selection_split: `{manifest['selection_split']}`",
        f"- diagnostic_splits: `{','.join(manifest['diagnostic_splits'])}`",
        "- model training: no new training in this stage",
        "- selection data: train/dev only",
        "- val_unseen_used_for_selection: `false`",
        "",
        "## Frozen Config",
        "",
        f"- gate_threshold: `{fmt(selected['gate_threshold'])}`",
        f"- tau: `{fmt(selected['tau'])}`",
        f"- allow_change_final: `{str(selected['allow_change_final']).lower()}`",
        "",
        "## Dev Metrics",
        "",
        markdown_table(
            [
                "SR",
                "delta_SR",
                "SPL",
                "delta_SPL",
                "recovery",
                "harm",
                "changed",
                "gate_pass",
                "gate_precision",
                "gate_recall",
            ],
            [
                [
                    pct(dev_metrics.get("SR")),
                    pct(dev_metrics.get("delta_SR")),
                    pct(dev_metrics.get("SPL")),
                    pct(dev_metrics.get("delta_SPL")),
                    pct(dev_metrics.get("recovery_rate")),
                    pct(dev_metrics.get("harm_rate")),
                    pct(dev_metrics.get("changed_endpoint_rate")),
                    pct(dev_metrics.get("gate_pass_rate")),
                    pct(dev_metrics.get("gate_precision")),
                    pct(dev_metrics.get("gate_recall")),
                ]
            ],
        ),
        "",
        "## Top Eligible Configs",
        "",
        markdown_table(
            ["rank", "gate", "tau", "allow", "dSR", "dSPL", "recovery", "harm", "changed", "gate_pass"],
            [
                [
                    int(row["selection_rank"]),
                    fmt(row["gate_threshold"]),
                    fmt(row["tau"]),
                    str(bool(row["allow_change_final"])).lower(),
                    pct(row["delta_SR"]),
                    pct(row["delta_SPL"]),
                    pct(row["recovery_rate"]),
                    pct(row["harm_rate"]),
                    pct(row["changed_endpoint_rate"]),
                    pct(row["gate_pass_rate"]),
                ]
                for _, row in top_rows.iterrows()
                if not pd.isna(row["selection_rank"])
            ],
        ),
        "",
        "## Failure Slice",
        "",
        markdown_table(
            ["slice", "items", "rate", "should_rerank", "recovered", "harmed", "changed"],
            [
                [
                    row["failure_slice"],
                    row["items"],
                    pct(row["rate"]),
                    row["should_rerank_items"],
                    row["recovered_items"],
                    row["harmed_items"],
                    row["changed_items"],
                ]
                for _, row in slice_summary.iterrows()
            ],
        ),
        "",
        "## Frozen Train/Dev Diagnostics",
        "",
        markdown_table(
            ["split", "SR", "delta_SR", "SPL", "delta_SPL", "recovery", "harm", "changed"],
            [
                [
                    row["protocol_split"],
                    pct(row["SR"]),
                    pct(row["delta_SR"]),
                    pct(row["SPL"]),
                    pct(row["delta_SPL"]),
                    pct(row["recovery_rate"]),
                    pct(row["harm_rate"]),
                    pct(row["changed_endpoint_rate"]),
                ]
                for _, row in frozen_summary.iterrows()
            ],
        ),
        "",
        "## Files",
        "",
    ]
    for key, value in manifest["files"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def resolve_path(explicit: str | None, default: Path) -> Path:
    return Path(explicit).resolve() if explicit else default.resolve()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_string_list(value: str) -> tuple[str, ...]:
    value = value.replace("/", ",")
    return tuple(item.strip() for item in value.split(",") if item.strip())


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


def parse_bool_list(value: str) -> list[bool]:
    values: list[bool] = []
    for part in value.split(","):
        parsed = parse_bool_value(part)
        if parsed is None:
            if str(part).strip():
                raise ValueError(f"Could not parse boolean value: {part!r}")
            continue
        values.append(parsed)
    if not values:
        raise ValueError("Expected at least one boolean")
    return values


def parse_bool_value(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def series_to_jsonable(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        payload[key] = jsonable_value(value)
    return payload


def jsonable_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


def path_to_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def format_float(value: float) -> str:
    return format(value, ".12g")


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
