#!/usr/bin/env python3
"""Evaluate offline endpoint reranker scores over fixed candidate groups."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_endpoint_candidate_groups as candidate_groups  # noqa: E402
import build_endpoint_heuristic_report as endpoint_heuristic  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "endpoint_reranker_eval.v1"
DEFAULT_ENDPOINT_LEARNING_DIR = "endpoint_learning"
DEFAULT_GATE_THRESHOLDS = (0.5,)
DEFAULT_TAUS = (0.0,)
DEFAULT_TARGET_SCOPE = "official"
DEFAULT_SPLIT = "dev"
DEFAULT_ALLOW_CHANGE_FINAL = (True,)
EPS = 1e-9

CANDIDATE_SCORE_ALIASES = (
    "candidate_score",
    "endpoint_score",
    "score",
    "logit",
    "model_score",
)
GATE_SCORE_ALIASES = (
    "gate_score",
    "should_rerank_score",
    "rerank_score",
    "episode_score",
)

ITEM_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "internal_item_id",
    "saved_instr_id",
    "gate_threshold",
    "tau",
    "allow_change_final",
    "gate_score",
    "gate_passed",
    "selection_reason",
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
    "final_viewpoint",
    "selected_viewpoint",
    "best_viewpoint",
    "final_success",
    "selected_success",
    "oracle_success",
    "nearest_endpoint_success",
    "should_rerank",
    "final_failure_bucket",
    "recovered",
    "harmed",
    "final_spl",
    "selected_spl",
    "nearest_endpoint_spl",
    "final_distance_m",
    "selected_distance_m",
    "final_path_length_m",
    "selected_path_length_m",
]

SUMMARY_FIELDNAMES = [
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
    "oracle_success_rate",
    "nearest_endpoint_success_rate",
    "gap_capture_rate",
    "final_SPL",
    "SPL",
    "delta_SPL",
    "nearest_endpoint_SPL",
    "recovery_rate",
    "harm_rate",
    "net_recovery_rate",
    "changed_endpoint_rate",
    "gate_pass_rate",
    "gate_precision",
    "gate_recall",
    "gate_auc",
    "overshoot_items",
    "overshoot_recovery_rate",
    "final_success_items",
    "final_success_harm_rate",
    "mean_gate_score",
    "mean_score_margin_over_final",
]

TAU_CURVE_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "gate_threshold",
    "tau",
    "allow_change_final",
    "items",
    "SR",
    "SPL",
    "delta_SR",
    "delta_SPL",
    "gap_capture_rate",
    "changed_endpoint_rate",
]

RECOVERY_HARM_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "gate_threshold",
    "tau",
    "allow_change_final",
    "items",
    "recovery_rate",
    "harm_rate",
    "net_recovery_rate",
    "overshoot_recovery_rate",
    "final_success_harm_rate",
    "gate_precision",
    "gate_recall",
    "gate_auc",
]


@dataclass(frozen=True)
class ScoreMaps:
    candidate_scores: dict[str, float]
    gate_scores: dict[str, float]
    candidate_score_column: str
    gate_score_column: str | None


@dataclass(frozen=True)
class EvalConfig:
    gate_threshold: float
    tau: float
    allow_change_final: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate endpoint reranker score CSVs on fixed candidate groups.",
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
        required=True,
        help=(
            "Score CSV with candidate_id plus candidate_score. gate_score is optional and may be "
            "repeated per candidate or provided per episode."
        ),
    )
    parser.add_argument(
        "--candidate-score-column",
        default=None,
        help="Candidate score column. Auto-detects candidate_score/endpoint_score/score/logit/model_score.",
    )
    parser.add_argument(
        "--gate-score-column",
        default=None,
        help="Gate score column. Auto-detects gate_score/should_rerank_score/rerank_score/episode_score.",
    )
    parser.add_argument(
        "--default-gate-score",
        type=float,
        default=1.0,
        help="Gate score used when the score CSV has no gate column. Defaults to 1.0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <endpoint-learning-dir>/eval_protocol.",
    )
    parser.add_argument(
        "--target-scope",
        default=DEFAULT_TARGET_SCOPE,
        help="Target scope to evaluate. Defaults to official.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=(
            "Comma-separated split filter. Values train/dev/test match protocol_split; "
            "dataset splits such as val_unseen match split."
        ),
    )
    parser.add_argument(
        "--gate-thresholds",
        default=",".join(format_float(value) for value in DEFAULT_GATE_THRESHOLDS),
        help="Comma-separated gate thresholds.",
    )
    parser.add_argument(
        "--taus",
        default=",".join(format_float(value) for value in DEFAULT_TAUS),
        help="Comma-separated final-stay tau values in candidate score units.",
    )
    parser.add_argument(
        "--allow-change-final",
        default="true",
        help="true, false, or comma-separated true,false grid. false gives the frozen-final baseline.",
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
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else endpoint_learning_dir / "eval_protocol"
    )

    manifest = evaluate_endpoint_reranker(
        candidate_csv=candidate_csv,
        episode_csv=episode_csv,
        score_csv=Path(args.score_csv).resolve(),
        output_dir=output_dir,
        target_scope=args.target_scope,
        split_filters=parse_string_list(args.split),
        gate_thresholds=tuple(parse_float_list(args.gate_thresholds)),
        taus=tuple(parse_float_list(args.taus)),
        allow_change_final_values=tuple(parse_bool_list(args.allow_change_final)),
        candidate_score_column=args.candidate_score_column,
        gate_score_column=args.gate_score_column,
        default_gate_score=args.default_gate_score,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def evaluate_endpoint_reranker(
    candidate_csv: Path,
    episode_csv: Path,
    score_csv: Path,
    output_dir: Path,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    split_filters: tuple[str, ...] = (DEFAULT_SPLIT,),
    gate_thresholds: tuple[float, ...] = DEFAULT_GATE_THRESHOLDS,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    allow_change_final_values: tuple[bool, ...] = DEFAULT_ALLOW_CHANGE_FINAL,
    candidate_score_column: str | None = None,
    gate_score_column: str | None = None,
    default_gate_score: float = 1.0,
) -> dict[str, Any]:
    validate_grid(gate_thresholds, taus, allow_change_final_values)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = read_csv(candidate_csv)
    episode_rows = read_csv(episode_csv)
    score_maps = load_score_maps(
        score_csv,
        candidate_score_column=candidate_score_column,
        gate_score_column=gate_score_column,
    )

    filtered_episodes = [
        row
        for row in episode_rows
        if row.get("target_scope") == target_scope and split_matches(row, split_filters)
    ]
    episode_ids = {str(row["episode_id"]) for row in filtered_episodes}
    filtered_candidates = [
        row
        for row in candidate_rows
        if str(row.get("episode_id")) in episode_ids
        and row.get("target_scope") == target_scope
        and split_matches(row, split_filters)
    ]
    if not filtered_episodes:
        raise ValueError(
            f"No episode groups matched target_scope={target_scope!r} and split={','.join(split_filters)!r}"
        )
    if not filtered_candidates:
        raise ValueError("No candidates matched the selected episode groups")

    candidates_by_episode = group_candidates_by_episode(filtered_candidates)
    missing_episode_ids = sorted(episode_ids.difference(candidates_by_episode))
    if missing_episode_ids:
        raise ValueError(f"Missing candidates for {len(missing_episode_ids)} episodes; first={missing_episode_ids[0]}")

    validate_candidate_scores(filtered_candidates, score_maps.candidate_scores)

    configs = [
        EvalConfig(gate_threshold=threshold, tau=tau, allow_change_final=allow_change_final)
        for threshold in gate_thresholds
        for tau in taus
        for allow_change_final in allow_change_final_values
    ]

    item_rows: list[dict[str, Any]] = []
    for group_row in filtered_episodes:
        episode_candidates = candidates_by_episode[str(group_row["episode_id"])]
        for config in configs:
            item_rows.append(
                evaluate_episode(
                    group_row=group_row,
                    candidate_rows=episode_candidates,
                    candidate_scores=score_maps.candidate_scores,
                    gate_scores=score_maps.gate_scores,
                    default_gate_score=default_gate_score,
                    config=config,
                )
            )

    summary_rows = build_summary_rows(item_rows)
    tau_rows = [{name: row.get(name) for name in TAU_CURVE_FIELDNAMES} for row in summary_rows]
    recovery_harm_rows = [
        {name: row.get(name) for name in RECOVERY_HARM_FIELDNAMES}
        for row in summary_rows
    ]

    items_csv = output_dir / "endpoint_learning_items.csv"
    summary_csv = output_dir / "endpoint_learning_summary.csv"
    tau_curve_csv = output_dir / "tau_curve.csv"
    recovery_harm_csv = output_dir / "recovery_harm_curve.csv"
    manifest_path = output_dir / "endpoint_reranker_eval_manifest.json"

    write_csv(items_csv, ITEM_FIELDNAMES, item_rows)
    write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(tau_curve_csv, TAU_CURVE_FIELDNAMES, tau_rows)
    write_csv(recovery_harm_csv, RECOVERY_HARM_FIELDNAMES, recovery_harm_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_csv": path_to_string(candidate_csv),
        "episode_csv": path_to_string(episode_csv),
        "score_csv": path_to_string(score_csv),
        "output_dir": path_to_string(output_dir),
        "target_scope": target_scope,
        "split_filters": list(split_filters),
        "gate_thresholds": list(gate_thresholds),
        "taus": list(taus),
        "allow_change_final_values": list(allow_change_final_values),
        "candidate_score_column": score_maps.candidate_score_column,
        "gate_score_column": score_maps.gate_score_column,
        "default_gate_score": default_gate_score,
        "inference_rule": (
            "if gate_score < gate_threshold choose final; otherwise choose argmax candidate_score; "
            "choose final when best_score <= final_score + tau"
        ),
        "files": {
            "items_csv": path_to_string(items_csv),
            "summary_csv": path_to_string(summary_csv),
            "tau_curve_csv": path_to_string(tau_curve_csv),
            "recovery_harm_curve_csv": path_to_string(recovery_harm_csv),
        },
        "counts": {
            "episodes": len(filtered_episodes),
            "candidates": len(filtered_candidates),
            "configs": len(configs),
            "item_rows": len(item_rows),
            "summary_rows": len(summary_rows),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def evaluate_episode(
    group_row: dict[str, str],
    candidate_rows: list[dict[str, str]],
    candidate_scores: dict[str, float],
    gate_scores: dict[str, float],
    default_gate_score: float,
    config: EvalConfig,
) -> dict[str, Any]:
    final_row = final_candidate(candidate_rows)
    final_candidate_id = str(final_row["candidate_id"])
    final_score = candidate_scores[final_candidate_id]
    gate_score = gate_scores.get(str(group_row["episode_id"]), default_gate_score)
    gate_passed = gate_score + EPS >= config.gate_threshold
    best_row = best_candidate(candidate_rows, candidate_scores)
    best_score = candidate_scores[str(best_row["candidate_id"])]

    if not config.allow_change_final:
        selected_row = final_row
        selection_reason = "change_disabled"
    elif not gate_passed:
        selected_row = final_row
        selection_reason = "gate_below_threshold"
    elif best_score <= final_score + config.tau + EPS:
        selected_row = final_row
        selection_reason = "final_within_tau"
    else:
        selected_row = best_row
        selection_reason = "ranker_best"

    selected_candidate_id = str(selected_row["candidate_id"])
    selected_score = candidate_scores[selected_candidate_id]
    final_success = parse_bool(group_row.get("final_success"))
    selected_success = parse_bool(selected_row.get("success_label"))
    recovered = bool(final_success is False and selected_success is True)
    harmed = bool(final_success is True and selected_success is False)

    return {
        "experiment_id": group_row.get("experiment_id"),
        "dataset": group_row.get("dataset"),
        "split": group_row.get("split"),
        "protocol_split": group_row.get("protocol_split"),
        "target_scope": group_row.get("target_scope"),
        "episode_id": group_row.get("episode_id"),
        "internal_item_id": group_row.get("internal_item_id"),
        "saved_instr_id": group_row.get("saved_instr_id"),
        "gate_threshold": config.gate_threshold,
        "tau": config.tau,
        "allow_change_final": config.allow_change_final,
        "gate_score": gate_score,
        "gate_passed": gate_passed,
        "selection_reason": selection_reason,
        "final_candidate_id": final_candidate_id,
        "selected_candidate_id": selected_candidate_id,
        "best_candidate_id": best_row.get("candidate_id"),
        "final_step": parse_int(final_row.get("candidate_step")),
        "selected_step": parse_int(selected_row.get("candidate_step")),
        "best_step": parse_int(best_row.get("candidate_step")),
        "selected_changed": selected_candidate_id != final_candidate_id,
        "final_score": final_score,
        "selected_score": selected_score,
        "best_score": best_score,
        "score_margin_over_final": best_score - final_score,
        "final_viewpoint": final_row.get("viewpoint"),
        "selected_viewpoint": selected_row.get("viewpoint"),
        "best_viewpoint": best_row.get("viewpoint"),
        "final_success": final_success,
        "selected_success": selected_success,
        "oracle_success": parse_bool(group_row.get("oracle_success")),
        "nearest_endpoint_success": parse_bool(group_row.get("nearest_endpoint_success")),
        "should_rerank": parse_bool(group_row.get("should_rerank")),
        "final_failure_bucket": group_row.get("final_failure_bucket"),
        "recovered": recovered,
        "harmed": harmed,
        "final_spl": parse_float(group_row.get("final_spl")),
        "selected_spl": parse_float(selected_row.get("spl_at_candidate")),
        "nearest_endpoint_spl": parse_float(group_row.get("nearest_endpoint_spl")),
        "final_distance_m": parse_float(group_row.get("final_distance_m")),
        "selected_distance_m": parse_float(selected_row.get("distance_to_goal_m")),
        "final_path_length_m": parse_float(group_row.get("final_path_length_m")),
        "selected_path_length_m": parse_float(selected_row.get("path_length_m")),
    }


def build_summary_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in item_rows:
        key = (
            row["experiment_id"],
            row["dataset"],
            row["split"],
            row["protocol_split"],
            row["target_scope"],
            row["gate_threshold"],
            row["tau"],
            row["allow_change_final"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=summary_sort_key):
        (
            experiment_id,
            dataset,
            split,
            protocol_split,
            target_scope,
            gate_threshold,
            tau,
            allow_change_final,
        ) = key
        rows = groups[key]
        final_sr = mean_bool(rows, "final_success")
        sr = mean_bool(rows, "selected_success")
        oracle_success_rate = mean_bool(rows, "oracle_success")
        final_spl = mean_float(rows, "final_spl")
        spl = mean_float(rows, "selected_spl")
        recovery_rate = mean_bool(rows, "recovered")
        harm_rate = mean_bool(rows, "harmed")
        overshoot_rows = [row for row in rows if row.get("final_failure_bucket") == "overshoot"]
        final_success_rows = [row for row in rows if row.get("final_success") is True]
        gate_pass_rows = [row for row in rows if row.get("gate_passed") is True]
        should_rows = [row for row in rows if row.get("should_rerank") is True]
        gate_true_pass_rows = [
            row
            for row in gate_pass_rows
            if row.get("should_rerank") is True
        ]

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "split": split,
                "protocol_split": protocol_split,
                "target_scope": target_scope,
                "gate_threshold": gate_threshold,
                "tau": tau,
                "allow_change_final": allow_change_final,
                "items": len(rows),
                "final_SR": final_sr,
                "SR": sr,
                "delta_SR": subtract_or_none(sr, final_sr),
                "oracle_success_rate": oracle_success_rate,
                "nearest_endpoint_success_rate": mean_bool(rows, "nearest_endpoint_success"),
                "gap_capture_rate": gap_capture_rate(sr, final_sr, oracle_success_rate),
                "final_SPL": final_spl,
                "SPL": spl,
                "delta_SPL": subtract_or_none(spl, final_spl),
                "nearest_endpoint_SPL": mean_float(rows, "nearest_endpoint_spl"),
                "recovery_rate": recovery_rate,
                "harm_rate": harm_rate,
                "net_recovery_rate": subtract_or_none(recovery_rate, harm_rate),
                "changed_endpoint_rate": mean_bool(rows, "selected_changed"),
                "gate_pass_rate": mean_bool(rows, "gate_passed"),
                "gate_precision": safe_divide(len(gate_true_pass_rows), len(gate_pass_rows)),
                "gate_recall": safe_divide(len(gate_true_pass_rows), len(should_rows)),
                "gate_auc": gate_auc(rows),
                "overshoot_items": len(overshoot_rows),
                "overshoot_recovery_rate": mean_bool(overshoot_rows, "recovered"),
                "final_success_items": len(final_success_rows),
                "final_success_harm_rate": mean_bool(final_success_rows, "harmed"),
                "mean_gate_score": mean_float(rows, "gate_score"),
                "mean_score_margin_over_final": mean_float(rows, "score_margin_over_final"),
            }
        )
    return summary_rows


def load_score_maps(
    score_csv: Path,
    candidate_score_column: str | None,
    gate_score_column: str | None,
) -> ScoreMaps:
    rows = read_csv(score_csv)
    if not rows:
        raise ValueError(f"Empty score CSV: {score_csv}")
    fieldnames = list(rows[0].keys())
    resolved_candidate_column = resolve_column(
        fieldnames,
        explicit=candidate_score_column,
        aliases=CANDIDATE_SCORE_ALIASES,
        required=True,
        label="candidate score",
    )
    resolved_gate_column = resolve_column(
        fieldnames,
        explicit=gate_score_column,
        aliases=GATE_SCORE_ALIASES,
        required=False,
        label="gate score",
    )

    candidate_scores: dict[str, float] = {}
    gate_scores: dict[str, float] = {}
    for row in rows:
        candidate_id = score_row_candidate_id(row)
        if candidate_id:
            candidate_score = parse_required_float(
                row.get(resolved_candidate_column),
                f"{resolved_candidate_column} for candidate_id={candidate_id}",
            )
            previous = candidate_scores.get(candidate_id)
            if previous is not None and abs(previous - candidate_score) > EPS:
                raise ValueError(f"Conflicting scores for candidate_id={candidate_id}")
            candidate_scores[candidate_id] = candidate_score

        if resolved_gate_column is not None:
            episode_id = score_row_episode_id(row, candidate_id)
            if episode_id:
                gate_score = parse_required_float(
                    row.get(resolved_gate_column),
                    f"{resolved_gate_column} for episode_id={episode_id}",
                )
                previous_gate = gate_scores.get(episode_id)
                if previous_gate is not None and abs(previous_gate - gate_score) > EPS:
                    raise ValueError(f"Conflicting gate scores for episode_id={episode_id}")
                gate_scores[episode_id] = gate_score

    if not candidate_scores:
        raise ValueError("Score CSV did not contain any candidate scores keyed by candidate_id or episode_id+candidate_step")
    return ScoreMaps(
        candidate_scores=candidate_scores,
        gate_scores=gate_scores,
        candidate_score_column=resolved_candidate_column,
        gate_score_column=resolved_gate_column,
    )


def score_row_candidate_id(row: dict[str, str]) -> str | None:
    candidate_id = as_nonempty_string(row.get("candidate_id"))
    if candidate_id is not None:
        return candidate_id
    episode_id = as_nonempty_string(row.get("episode_id"))
    candidate_step = parse_int(row.get("candidate_step"))
    if episode_id is None or candidate_step is None:
        return None
    return candidate_groups.make_candidate_id(episode_id, candidate_step)


def score_row_episode_id(row: dict[str, str], candidate_id: str | None) -> str | None:
    episode_id = as_nonempty_string(row.get("episode_id"))
    if episode_id is not None:
        return episode_id
    if candidate_id and ":step_" in candidate_id:
        return candidate_id.rsplit(":step_", 1)[0]
    return None


def resolve_column(
    fieldnames: list[str],
    explicit: str | None,
    aliases: tuple[str, ...],
    required: bool,
    label: str,
) -> str | None:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Requested {label} column {explicit!r} is not in score CSV")
        return explicit
    for alias in aliases:
        if alias in fieldnames:
            return alias
    if required:
        raise ValueError(f"Could not auto-detect {label} column. Available columns: {', '.join(fieldnames)}")
    return None


def validate_candidate_scores(candidate_rows: list[dict[str, str]], candidate_scores: dict[str, float]) -> None:
    missing = [
        str(row["candidate_id"])
        for row in candidate_rows
        if str(row["candidate_id"]) not in candidate_scores
    ]
    if missing:
        raise ValueError(f"Missing scores for {len(missing)} selected candidates; first={missing[0]}")


def final_candidate(candidate_rows: list[dict[str, str]]) -> dict[str, str]:
    final_rows = [row for row in candidate_rows if parse_bool(row.get("is_final")) is True]
    if len(final_rows) != 1:
        episode_id = candidate_rows[0].get("episode_id") if candidate_rows else "<empty>"
        raise ValueError(f"Expected exactly one final candidate for episode_id={episode_id}, got {len(final_rows)}")
    return final_rows[0]


def best_candidate(candidate_rows: list[dict[str, str]], candidate_scores: dict[str, float]) -> dict[str, str]:
    return max(
        candidate_rows,
        key=lambda row: (
            candidate_scores[str(row["candidate_id"])],
            parse_int(row.get("candidate_step")) or -1,
        ),
    )


def group_candidates_by_episode(candidate_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: parse_int(row.get("candidate_step")) or -1)
    return grouped


def split_matches(row: dict[str, str], filters: tuple[str, ...]) -> bool:
    if not filters or "all" in filters:
        return True
    split = row.get("split")
    protocol_split = row.get("protocol_split")
    return split in filters or protocol_split in filters


def gate_auc(rows: list[dict[str, Any]]) -> float | None:
    scored = [
        (parse_float(row.get("gate_score")), bool(row.get("should_rerank")))
        for row in rows
        if parse_float(row.get("gate_score")) is not None and row.get("should_rerank") is not None
    ]
    positives = sum(1 for _, label in scored if label)
    negatives = sum(1 for _, label in scored if not label)
    if positives == 0 or negatives == 0:
        return None

    scored.sort(key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(scored):
        next_index = index + 1
        while next_index < len(scored) and abs((scored[next_index][0] or 0.0) - (scored[index][0] or 0.0)) <= EPS:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        rank_sum_positive += average_rank * sum(1 for _, label in scored[index:next_index] if label)
        index = next_index
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def gap_capture_rate(sr: float | None, final_sr: float | None, oracle_success_rate: float | None) -> float | None:
    if sr is None or final_sr is None or oracle_success_rate is None:
        return None
    gap = oracle_success_rate - final_sr
    if gap <= EPS:
        return None
    return (sr - final_sr) / gap


def summary_sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    _, dataset, split, protocol_split, target_scope, gate_threshold, tau, allow_change_final = key
    dataset_order = {"R2R": 0, "REVERIE": 1, "SOON": 2, "CVDN": 3}
    protocol_order = {"train": 0, "dev": 1, "test": 2}
    scope_order = {"official": 0, "goal": 1, "region": 2, "region_threshold": 3}
    return (
        dataset_order.get(dataset, 99),
        dataset,
        split,
        protocol_order.get(protocol_split, 99),
        protocol_split,
        scope_order.get(target_scope, 99),
        target_scope,
        float(gate_threshold),
        float(tau),
        0 if allow_change_final else 1,
    )


def resolve_endpoint_learning_dir(experiment_dir: Path | None, endpoint_learning_dir: str | None) -> Path:
    if endpoint_learning_dir:
        return Path(endpoint_learning_dir).resolve()
    if experiment_dir is None:
        raise ValueError("Either --experiment-dir or --endpoint-learning-dir is required")
    return (experiment_dir / DEFAULT_ENDPOINT_LEARNING_DIR).resolve()


def validate_grid(
    gate_thresholds: tuple[float, ...],
    taus: tuple[float, ...],
    allow_change_final_values: tuple[bool, ...],
) -> None:
    if not gate_thresholds:
        raise ValueError("At least one gate threshold is required")
    if not taus:
        raise ValueError("At least one tau is required")
    if not allow_change_final_values:
        raise ValueError("At least one allow-change-final value is required")
    for value in (*gate_thresholds, *taus):
        if not math.isfinite(value):
            raise ValueError("Gate thresholds and taus must be finite")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_string_list(value: str) -> tuple[str, ...]:
    value = value.replace("/", ",")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_list(value: str) -> list[float]:
    values = [parse_float(item.strip()) for item in value.split(",") if item.strip()]
    parsed = [item for item in values if item is not None]
    if len(parsed) != len(values):
        raise ValueError(f"Could not parse float list: {value!r}")
    return parsed


def parse_bool_list(value: str) -> list[bool]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = parse_bool(item)
        if parsed is None:
            raise ValueError(f"Could not parse boolean value: {item!r}")
        values.append(parsed)
    return values


def parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_float(value: Any) -> float | None:
    return endpoint_heuristic.parse_float(value)


def parse_required_float(value: Any, label: str) -> float:
    parsed = parse_float(value)
    if parsed is None:
        raise ValueError(f"Missing or invalid float for {label}")
    return parsed


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def as_nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1.0 for value in values if bool(value)) / len(values)


def mean_float(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [parse_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None
    return sum(values) / len(values)


def subtract_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_float(value: float) -> str:
    return format(value, ".12g")


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".12g")
    return str(value)


def path_to_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
