#!/usr/bin/env python3
"""Build actionable rerank labels from a fixed endpoint ranker score CSV."""

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
import train_endpoint_ranker_baseline as baseline  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_actionable_gate_labels.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_OUTPUT_NAME = "actionable_gate_labels"
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_SPLITS = ("train", "dev")
EPS = 1e-12

LABEL_COLUMNS = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "internal_item_id",
    "saved_instr_id",
    "should_rerank",
    "actionable_rerank",
    "criterion",
    "tau",
    "top_k",
    "final_success",
    "oracle_success",
    "final_candidate_id",
    "top1_candidate_id",
    "best_success_candidate_id",
    "top1_success",
    "best_success_rank",
    "final_score",
    "top1_score",
    "best_success_score",
    "top1_margin_over_final",
    "best_success_margin_over_final",
]

SUMMARY_COLUMNS = [
    "split_filter",
    "items",
    "should_rerank_items",
    "actionable_items",
    "actionable_rate",
    "actionable_over_should_rerank",
    "top1_success_over_should_rerank",
    "median_best_success_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build episode-level actionable_rerank labels for a fixed ranker.",
    )
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--endpoint-learning-dir", default=None)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--candidate-score-column", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-scope", default=DEFAULT_TARGET_SCOPE)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument(
        "--criterion",
        choices=("top1_success", "topk_success", "any_success_margin"),
        default="top1_success",
        help="Criterion used to define current-ranker actionable episodes.",
    )
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    endpoint_learning_dir = resolve_endpoint_learning_dir(experiment_dir, args.endpoint_learning_dir)
    candidate_csv = resolve_path(args.candidate_csv, endpoint_learning_dir / "candidate_groups" / "endpoint_candidates.csv")
    output_dir = resolve_path(args.output_dir, endpoint_learning_dir / DEFAULT_OUTPUT_NAME)
    manifest = build_actionable_gate_labels(
        candidate_csv=candidate_csv,
        score_csv=Path(args.score_csv).resolve(),
        output_dir=output_dir,
        target_scope=args.target_scope,
        splits=baseline.parse_string_list(args.splits),
        candidate_score_column=args.candidate_score_column,
        criterion=args.criterion,
        tau=args.tau,
        top_k=args.top_k,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def build_actionable_gate_labels(
    candidate_csv: Path,
    score_csv: Path,
    output_dir: Path,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    candidate_score_column: str | None = None,
    criterion: str = "top1_success",
    tau: float = 0.0,
    top_k: int = 3,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_csv, low_memory=False)
    candidates = candidates[candidates["target_scope"] == target_scope].copy()
    candidates = candidates[select_split_mask(candidates, splits)].copy()
    if candidates.empty:
        raise ValueError(f"No candidates matched target_scope={target_scope!r} and splits={splits!r}")

    score_frame = pd.read_csv(score_csv, low_memory=False)
    score_column = resolve_score_column(score_frame.columns.tolist(), candidate_score_column)
    score_map = score_frame.set_index("candidate_id")[score_column].map(parse_float).to_dict()
    candidates["candidate_score"] = candidates["candidate_id"].map(score_map)
    if candidates["candidate_score"].isna().any():
        missing = candidates.loc[candidates["candidate_score"].isna(), "candidate_id"].iloc[0]
        raise ValueError(f"Missing candidate score for candidate_id={missing}")

    rows = [
        build_one_label_row(
            group=group,
            criterion=criterion,
            tau=tau,
            top_k=top_k,
        )
        for _, group in candidates.groupby("episode_id", sort=False)
    ]
    summary_rows = build_summary_rows(rows, splits)

    labels_csv = output_dir / "actionable_gate_labels.csv"
    summary_csv = output_dir / "actionable_gate_label_summary.csv"
    report_md = output_dir / "actionable_gate_label_report.md"
    manifest_json = output_dir / "manifest.json"
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(labels_csv, index=False)
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_csv, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_scope": target_scope,
        "splits": list(splits),
        "criterion": criterion,
        "tau": tau,
        "top_k": top_k,
        "candidate_score_column": score_column,
        "files": {
            "candidate_csv": baseline.path_to_string(candidate_csv),
            "score_csv": baseline.path_to_string(score_csv),
            "actionable_gate_labels_csv": baseline.path_to_string(labels_csv),
            "actionable_gate_label_summary_csv": baseline.path_to_string(summary_csv),
            "actionable_gate_label_report_md": baseline.path_to_string(report_md),
            "manifest_json": baseline.path_to_string(manifest_json),
        },
        "counts": {
            "episodes": len(rows),
            "candidates": int(len(candidates)),
        },
    }
    baseline.write_json(manifest_json, manifest)
    write_report(report_md, manifest, summary_rows)
    return manifest


def build_one_label_row(group: pd.DataFrame, criterion: str, tau: float, top_k: int) -> dict[str, Any]:
    group = group.sort_values(["candidate_score", "candidate_step"], ascending=[False, False]).copy()
    first = group.iloc[0]
    final_rows = group[group["is_final"].map(parse_bool)]
    final = final_rows.iloc[0] if not final_rows.empty else group.sort_values("candidate_step").iloc[-1]
    top1 = group.iloc[0]
    success_rows = group[group["success_label"].map(parse_bool)].copy()
    final_success = parse_bool(first.get("final_success"))
    oracle_success = parse_bool(first.get("oracle_success"))
    should_rerank = parse_bool(first.get("should_rerank"))

    final_score = parse_float(final.get("candidate_score"))
    top1_score = parse_float(top1.get("candidate_score"))
    top1_success = parse_bool(top1.get("success_label"))
    best_success_candidate_id = None
    best_success_rank = math.nan
    best_success_score = math.nan
    if not success_rows.empty:
        success_rows = success_rows.sort_values(["candidate_score", "candidate_step"], ascending=[False, False])
        best_success = success_rows.iloc[0]
        best_success_candidate_id = best_success.get("candidate_id")
        best_success_score = parse_float(best_success.get("candidate_score"))
        ranked_ids = group["candidate_id"].astype(str).tolist()
        best_success_rank = ranked_ids.index(str(best_success_candidate_id)) + 1

    margin_top1 = subtract(top1_score, final_score)
    margin_best_success = subtract(best_success_score, final_score)
    actionable = False
    if should_rerank and final_success is False and oracle_success is True:
        if criterion == "top1_success":
            actionable = top1_success is True and finite_gt(margin_top1, tau)
        elif criterion == "topk_success":
            actionable = (
                math.isfinite(best_success_rank)
                and best_success_rank <= top_k
                and finite_gt(margin_best_success, tau)
            )
        elif criterion == "any_success_margin":
            actionable = finite_gt(margin_best_success, tau)
        else:
            raise ValueError(f"Unsupported criterion={criterion!r}")

    return {
        "experiment_id": first.get("experiment_id"),
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "protocol_split": first.get("protocol_split"),
        "target_scope": first.get("target_scope"),
        "episode_id": first.get("episode_id"),
        "internal_item_id": first.get("internal_item_id"),
        "saved_instr_id": first.get("saved_instr_id"),
        "should_rerank": should_rerank,
        "actionable_rerank": actionable,
        "criterion": criterion,
        "tau": tau,
        "top_k": top_k,
        "final_success": final_success,
        "oracle_success": oracle_success,
        "final_candidate_id": final.get("candidate_id"),
        "top1_candidate_id": top1.get("candidate_id"),
        "best_success_candidate_id": best_success_candidate_id,
        "top1_success": top1_success,
        "best_success_rank": best_success_rank,
        "final_score": final_score,
        "top1_score": top1_score,
        "best_success_score": best_success_score,
        "top1_margin_over_final": margin_top1,
        "best_success_margin_over_final": margin_best_success,
    }


def build_summary_rows(rows: list[dict[str, Any]], splits: tuple[str, ...]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for split_filter in splits:
        subset = frame[(frame["protocol_split"] == split_filter) | (frame["split"] == split_filter)]
        if subset.empty:
            continue
        should = subset["should_rerank"].astype(bool)
        actionable = subset["actionable_rerank"].astype(bool)
        should_subset = subset[should]
        ranks = pd.to_numeric(should_subset["best_success_rank"], errors="coerce").dropna()
        summary.append(
            {
                "split_filter": split_filter,
                "items": int(len(subset)),
                "should_rerank_items": int(should.sum()),
                "actionable_items": int(actionable.sum()),
                "actionable_rate": safe_divide(int(actionable.sum()), len(subset)),
                "actionable_over_should_rerank": safe_divide(int(actionable.sum()), int(should.sum())),
                "top1_success_over_should_rerank": safe_divide(
                    int((should_subset["top1_success"].astype(bool)).sum()),
                    int(should.sum()),
                ),
                "median_best_success_rank": float(ranks.median()) if not ranks.empty else math.nan,
            }
        )
    return summary


def write_report(path: Path, manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Endpoint Actionable Gate Label Report",
        "",
        "This report is generated by `scripts/analysis/build_endpoint_actionable_gate_labels.py`.",
        "",
        "## Protocol",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- target_scope: `{manifest['target_scope']}`",
        f"- criterion: `{manifest['criterion']}`",
        f"- tau: `{manifest['tau']}`",
        f"- top_k: `{manifest['top_k']}`",
        "",
        "## Summary",
        "",
        baseline.markdown_table(
            [
                "split",
                "items",
                "should_rerank",
                "actionable",
                "actionable_rate",
                "actionable/should",
                "top1_success/should",
                "median_best_success_rank",
            ],
            [
                [
                    row["split_filter"],
                    row["items"],
                    row["should_rerank_items"],
                    row["actionable_items"],
                    baseline.pct(row["actionable_rate"]),
                    baseline.pct(row["actionable_over_should_rerank"]),
                    baseline.pct(row["top1_success_over_should_rerank"]),
                    baseline.fmt(row["median_best_success_rank"]),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Files",
        "",
    ]
    for key, value in manifest["files"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_score_column(columns: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Missing explicit candidate score column {explicit!r}")
        return explicit
    for alias in reranker_eval.CANDIDATE_SCORE_ALIASES:
        if alias in columns:
            return alias
    raise ValueError(f"Could not resolve candidate score column from {columns}")


def select_split_mask(frame: pd.DataFrame, splits: tuple[str, ...]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for split_filter in splits:
        mask = mask | (frame["protocol_split"] == split_filter) | (frame["split"] == split_filter)
    return mask


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def subtract(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return math.nan
    return left - right


def finite_gt(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value > threshold + EPS


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator / denominator)


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def resolve_path(value: str | None, default: Path) -> Path:
    return Path(value).resolve() if value else default.resolve()


if __name__ == "__main__":
    main()
