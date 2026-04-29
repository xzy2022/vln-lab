#!/usr/bin/env python3
"""Build offline endpoint candidate groups and preference pairs from SAME eval outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_endpoint_heuristic_report as endpoint_heuristic  # noqa: E402
import build_oracle_gap_report as oracle_gap  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_NAME = "endpoint_learning"
SCHEMA_VERSION = "endpoint_candidate_groups.v1"
DEFAULT_TARGET_SCOPES = ("official",)
DEFAULT_DEV_RATIO = 0.2
DEFAULT_DEV_SALT = "endpoint_learning.v1"
DEFAULT_LAST_K = 5
DEFAULT_LOOP_WINDOW = 10
DEFAULT_REWARD_ALPHA = 1.0
EPS = 1e-9

CANDIDATE_FEATURE_COLUMNS = [
    "is_final",
    "has_decision_trace",
    "step_frac",
    "path_length_m",
    "stop_prob",
    "stop_margin_prob",
    "selected_prob",
    "top1_top2_margin",
    "moe_router_entropy",
    "fuse_weight",
    "is_revisit",
    "is_loop_region",
    "is_last_k",
]

CANDIDATE_TRACE_METADATA_COLUMNS = [
    "decision_trace_index",
    "is_route_intermediate",
    "is_route_expanded_without_decision",
    "route_step_offset",
    "route_step_count",
]

LABEL_COLUMNS = [
    "success_label",
    "spl_at_candidate",
    "distance_to_goal_m",
    "reward",
    "is_best_success_candidate",
    "is_nearest_candidate",
]

PAIR_TYPES = [
    "success_gt_fail",
    "better_spl_success_gt_lower_spl_success",
    "final_success_final_gt_failed_nonfinal",
]

CANDIDATE_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "candidate_id",
    "internal_item_id",
    "saved_instr_id",
    "success_mode",
    "distance_key",
    "success_threshold_m",
    "trace_available",
    "trajectory_step_count",
    "candidate_step",
    "viewpoint",
    "has_decision_trace",
    "decision_trace_index",
    "is_route_intermediate",
    "is_route_expanded_without_decision",
    "route_step_offset",
    "route_step_count",
    "is_final",
    "step_frac",
    "path_length_m",
    "stop_prob",
    "stop_margin_prob",
    "selected_prob",
    "top1_top2_margin",
    "moe_router_entropy",
    "fuse_weight",
    "is_revisit",
    "is_loop_region",
    "is_last_k",
    "final_step",
    "final_success",
    "oracle_success",
    "should_rerank",
    "final_failure_bucket",
    "success_label",
    "spl_at_candidate",
    "distance_to_goal_m",
    "reward",
    "is_best_success_candidate",
    "is_nearest_candidate",
]

EPISODE_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "internal_item_id",
    "saved_instr_id",
    "success_mode",
    "distance_key",
    "success_threshold_m",
    "trace_available",
    "trajectory_step_count",
    "decision_trace_step_count",
    "candidate_count",
    "candidates_with_decision_trace",
    "route_intermediate_candidates",
    "route_expanded_without_decision_candidates",
    "pair_count",
    "final_step",
    "first_success_step",
    "best_distance_step",
    "final_viewpoint",
    "final_success",
    "oracle_success",
    "nearest_endpoint_success",
    "should_rerank",
    "final_failure_bucket",
    "final_distance_m",
    "best_distance_m",
    "final_path_length_m",
    "shortest_path_length_m",
    "final_spl",
    "nearest_endpoint_spl",
    "success_candidate_count",
    "failed_candidate_count",
]

PAIR_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episode_id",
    "pair_id",
    "pair_type",
    "chosen_candidate_id",
    "rejected_candidate_id",
    "chosen_step",
    "rejected_step",
    "chosen_viewpoint",
    "rejected_viewpoint",
    "chosen_success_label",
    "rejected_success_label",
    "chosen_spl",
    "rejected_spl",
    "chosen_reward",
    "rejected_reward",
    "chosen_is_final",
    "rejected_is_final",
    "should_rerank",
]

PROTOCOL_SUMMARY_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "protocol_split",
    "target_scope",
    "episodes",
    "candidates",
    "pairs",
    "should_rerank_episodes",
    "should_rerank_rate",
    "final_success_rate",
    "oracle_success_rate",
    "success_candidates",
    "failed_candidates",
    "decision_trace_candidates",
    "decision_trace_candidate_rate",
    "route_intermediate_candidates",
    "route_intermediate_candidate_rate",
    "route_expanded_without_decision_candidates",
    "route_expanded_without_decision_candidate_rate",
    "success_gt_fail_pairs",
    "better_spl_success_gt_lower_spl_success_pairs",
    "final_success_final_gt_failed_nonfinal_pairs",
]


@dataclass(frozen=True)
class EpisodeArtifacts:
    group_row: dict[str, Any]
    candidate_rows: list[dict[str, Any]]
    pair_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class TraceAlignment:
    raw_step: dict[str, Any] | None
    decision: endpoint_heuristic.DecisionStep | None
    decision_trace_index: int | None
    is_route_intermediate: bool
    is_route_expanded_without_decision: bool
    route_step_offset: int | None
    route_step_count: int | None


@dataclass(frozen=True)
class AlignedTrace:
    trace_available: bool
    raw_step_count: int
    decision_steps: tuple[endpoint_heuristic.DecisionStep, ...]
    by_step: tuple[TraceAlignment, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed offline endpoint candidate groups and preference pairs.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment output directory, such as experiment_outputs/<experiment_id>.",
    )
    parser.add_argument(
        "--fine-metrics-dir",
        default=None,
        help="Fine metrics directory. Defaults to <experiment-dir>/fine_metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory. Defaults to <experiment-dir>/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--target-scopes",
        default=",".join(DEFAULT_TARGET_SCOPES),
        help="Comma-separated target scopes to build. Defaults to official.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_DEV_RATIO,
        help="Hash split ratio used to mark val_train_seen episodes as dev.",
    )
    parser.add_argument(
        "--dev-salt",
        default=DEFAULT_DEV_SALT,
        help="Salt for stable train/dev hashing.",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=DEFAULT_LAST_K,
        help="K used for the is_last_k candidate feature.",
    )
    parser.add_argument(
        "--loop-window",
        type=int,
        default=DEFAULT_LOOP_WINDOW,
        help="Maximum revisit window used for the is_loop_region feature.",
    )
    parser.add_argument(
        "--reward-alpha",
        type=float,
        default=DEFAULT_REWARD_ALPHA,
        help="Reward label: success * (1 + alpha * SPL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    fine_metrics_dir = Path(args.fine_metrics_dir) if args.fine_metrics_dir else experiment_dir / "fine_metrics"
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / DEFAULT_OUTPUT_NAME
    target_scopes = tuple(scope.strip() for scope in args.target_scopes.split(",") if scope.strip())

    build_endpoint_candidate_groups(
        experiment_dir=experiment_dir,
        fine_metrics_dir=fine_metrics_dir,
        output_dir=output_dir,
        target_scopes=target_scopes,
        dev_ratio=args.dev_ratio,
        dev_salt=args.dev_salt,
        last_k=args.last_k,
        loop_window=args.loop_window,
        reward_alpha=args.reward_alpha,
    )


def build_endpoint_candidate_groups(
    experiment_dir: Path,
    fine_metrics_dir: Path,
    output_dir: Path,
    target_scopes: tuple[str, ...] = DEFAULT_TARGET_SCOPES,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    dev_salt: str = DEFAULT_DEV_SALT,
    last_k: int = DEFAULT_LAST_K,
    loop_window: int = DEFAULT_LOOP_WINDOW,
    reward_alpha: float = DEFAULT_REWARD_ALPHA,
) -> dict[str, Any]:
    if not 0.0 <= dev_ratio <= 1.0:
        raise ValueError(f"--dev-ratio must be in [0, 1], got {dev_ratio}")
    if last_k <= 0:
        raise ValueError(f"--last-k must be positive, got {last_k}")
    if loop_window <= 0:
        raise ValueError(f"--loop-window must be positive, got {loop_window}")

    experiment_dir = experiment_dir.resolve()
    fine_metrics_dir = fine_metrics_dir.resolve()
    output_dir = output_dir.resolve()
    eval_items_dir = experiment_dir / "eval_items"

    sources = oracle_gap.discover_eval_item_sources(eval_items_dir)
    if not sources:
        raise FileNotFoundError(f"No eval_items contexts found in {eval_items_dir}")

    fine_rows = oracle_gap.load_fine_metrics_wide(fine_metrics_dir / "tables" / "fine_metrics_wide.csv")

    candidate_dir = output_dir / "candidate_groups"
    pair_dir = output_dir / "preference_pairs"
    protocol_dir = output_dir / "eval_protocol"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    group_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for source in sources:
        for item in oracle_gap.read_jsonl(source.items_path):
            identity = item.get("identity", {})
            key = (
                source.dataset,
                source.split,
                str(identity.get("internal_item_id")),
            )
            fine_row = fine_rows.get(key)
            if fine_row is None:
                raise KeyError(f"Missing fine_metrics row for {key}")

            protocol_split = protocol_split_for(
                dataset=source.dataset,
                split=source.split,
                internal_item_id=str(identity.get("internal_item_id")),
                dev_ratio=dev_ratio,
                salt=dev_salt,
            )
            for scope_name in target_scopes:
                target_scope = oracle_gap.build_target_scope(item, source.dataset, scope_name)
                if target_scope is None:
                    continue
                artifacts = build_episode_artifacts(
                    item=item,
                    source=source,
                    fine_row=fine_row,
                    target_scope=target_scope,
                    protocol_split=protocol_split,
                    last_k=last_k,
                    loop_window=loop_window,
                    reward_alpha=reward_alpha,
                )
                group_rows.append(artifacts.group_row)
                candidate_rows.extend(artifacts.candidate_rows)
                pair_rows.extend(artifacts.pair_rows)

    groups_csv = candidate_dir / "episode_groups.csv"
    candidates_csv = candidate_dir / "endpoint_candidates.csv"
    pairs_csv = pair_dir / "preference_pairs.csv"
    summary_csv = protocol_dir / "eval_protocol_summary.csv"
    protocol_json = protocol_dir / "protocol.json"
    manifest_path = output_dir / "manifest.json"

    summary_rows = build_protocol_summary(group_rows, candidate_rows, pair_rows)
    write_csv(groups_csv, EPISODE_FIELDNAMES, group_rows)
    write_csv(candidates_csv, CANDIDATE_FIELDNAMES, candidate_rows)
    write_csv(pairs_csv, PAIR_FIELDNAMES, pair_rows)
    write_csv(summary_csv, PROTOCOL_SUMMARY_FIELDNAMES, summary_rows)

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "candidate_protocol": "one episode = one group; one trajectory step = one candidate; no viewpoint deduplication",
        "should_rerank": "final_success == false AND oracle_success == true",
        "pair_types": PAIR_TYPES,
        "target_scopes": list(target_scopes),
        "default_target_scope": "official",
        "success_threshold_source": "eval_context.run_context.success_threshold_m",
        "train_dev_protocol": {
            "val_train_seen": f"stable hash split with dev_ratio={dev_ratio}",
            "val_unseen": "test/report only",
            "salt": dev_salt,
        },
        "candidate_feature_columns": CANDIDATE_FEATURE_COLUMNS,
        "candidate_trace_metadata_columns": CANDIDATE_TRACE_METADATA_COLUMNS,
        "label_columns_do_not_use_for_training_or_inference": LABEL_COLUMNS,
        "trace_alignment": (
            "decision_trace.steps are aligned to prediction.trajectory by current_viewpoint and "
            "route_viewpoints; route-expanded intermediate trajectory steps keep label columns "
            "but have has_decision_trace=false"
        ),
        "last_k": last_k,
        "loop_window": loop_window,
        "reward": f"success * (1 + {reward_alpha:g} * SPL)",
    }
    write_json(protocol_json, protocol)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": experiment_dir.name,
        "experiment_dir": path_to_string(experiment_dir),
        "fine_metrics_dir": path_to_string(fine_metrics_dir),
        "output_dir": path_to_string(output_dir),
        "target_scopes": list(target_scopes),
        "dev_ratio": dev_ratio,
        "dev_salt": dev_salt,
        "last_k": last_k,
        "loop_window": loop_window,
        "reward_alpha": reward_alpha,
        "files": {
            "episode_groups_csv": path_to_string(groups_csv),
            "endpoint_candidates_csv": path_to_string(candidates_csv),
            "preference_pairs_csv": path_to_string(pairs_csv),
            "eval_protocol_summary_csv": path_to_string(summary_csv),
            "protocol_json": path_to_string(protocol_json),
        },
        "counts": {
            "eval_item_sources": len(sources),
            "episode_groups": len(group_rows),
            "endpoint_candidates": len(candidate_rows),
            "preference_pairs": len(pair_rows),
            "summary_rows": len(summary_rows),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def build_episode_artifacts(
    item: dict[str, Any],
    source: oracle_gap.EvalItemSource,
    fine_row: dict[str, str],
    target_scope: oracle_gap.TargetScope,
    protocol_split: str,
    last_k: int = DEFAULT_LAST_K,
    loop_window: int = DEFAULT_LOOP_WINDOW,
    reward_alpha: float = DEFAULT_REWARD_ALPHA,
) -> EpisodeArtifacts:
    identity = item.get("identity", {})
    internal_item_id = str(identity.get("internal_item_id"))
    saved_instr_id = identity.get("saved_instr_id")
    experiment_id = source.items_path.parents[1].name
    episode_id = make_episode_id(source.dataset, source.split, target_scope.name, internal_item_id)

    trajectory = list(item.get("prediction", {}).get("trajectory") or [])
    raw_steps = (item.get("prediction", {}).get("decision_trace") or {}).get("steps") or []
    aligned_trace = build_aligned_trace(trajectory, raw_steps)
    cumulative_lengths = oracle_gap.as_float_list(
        item.get("primitives", {}).get("trajectory_cumulative_lengths_m") or []
    )
    distances = oracle_gap.distance_series(item, target_scope)

    final_step = len(trajectory) - 1 if trajectory else None
    first_success_step = oracle_gap.first_success_index(
        trajectory=trajectory,
        distances=distances,
        threshold=source.success_threshold_m,
        target_scope=target_scope,
    )
    best_distance_step = oracle_gap.first_best_distance_index(distances)
    final_success = oracle_gap.success_at_step(
        trajectory,
        distances,
        final_step,
        source.success_threshold_m,
        target_scope,
    )
    oracle_success = first_success_step is not None
    nearest_endpoint_success = oracle_gap.success_at_step(
        trajectory,
        distances,
        best_distance_step,
        source.success_threshold_m,
        target_scope,
    )
    final_distance_m = oracle_gap.value_at(distances, final_step)
    best_distance_m = oracle_gap.value_at(distances, best_distance_step)
    final_is_best_distance = (
        final_distance_m is not None
        and best_distance_m is not None
        and float(final_distance_m) <= float(best_distance_m) + EPS
    )
    final_failure_bucket = endpoint_heuristic.final_failure_bucket(
        final_success=final_success,
        oracle_success=oracle_success,
        final_is_best_distance=final_is_best_distance,
    )
    should_rerank = bool(final_success is False and oracle_success is True)

    shortest_path_length_m = endpoint_heuristic.parse_float(
        fine_row.get(f"{target_scope.metric_group}.shortest_path_length_m")
    )
    final_path_length_m = oracle_gap.value_at(cumulative_lengths, final_step)
    best_endpoint_path_length_m = oracle_gap.value_at(cumulative_lengths, best_distance_step)
    final_spl = endpoint_heuristic.parse_float(fine_row.get(f"{target_scope.metric_group}.spl"))
    if final_spl is None:
        final_spl = oracle_gap.spl_at(final_success, final_path_length_m, shortest_path_length_m)
    nearest_endpoint_spl = oracle_gap.spl_at(
        nearest_endpoint_success,
        best_endpoint_path_length_m,
        shortest_path_length_m,
    )

    loop_region = loop_region_indices(trajectory, aligned_trace.decision_steps, window=loop_window)
    candidate_rows: list[dict[str, Any]] = []
    seen_viewpoints: set[str] = set()
    last_k_start = max(0, len(trajectory) - last_k)

    for step_index, viewpoint in enumerate(trajectory):
        alignment = aligned_trace.by_step[step_index] if step_index < len(aligned_trace.by_step) else empty_alignment()
        decision = alignment.decision
        raw_step = alignment.raw_step or {}
        path_length_m = oracle_gap.value_at(cumulative_lengths, step_index)
        success_label = oracle_gap.success_at_step(
            trajectory,
            distances,
            step_index,
            source.success_threshold_m,
            target_scope,
        )
        spl_at_candidate = oracle_gap.spl_at(success_label, path_length_m, shortest_path_length_m)
        distance_to_goal_m = oracle_gap.value_at(distances, step_index)
        reward = (1.0 + reward_alpha * spl_at_candidate) if success_label is True else 0.0
        is_revisit = viewpoint in seen_viewpoints
        seen_viewpoints.add(viewpoint)

        candidate_rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": source.dataset,
                "split": source.split,
                "protocol_split": protocol_split,
                "target_scope": target_scope.name,
                "episode_id": episode_id,
                "candidate_id": make_candidate_id(episode_id, step_index),
                "internal_item_id": internal_item_id,
                "saved_instr_id": saved_instr_id,
                "success_mode": target_scope.success_mode,
                "distance_key": target_scope.distance_key,
                "success_threshold_m": source.success_threshold_m,
                "trace_available": aligned_trace.trace_available,
                "trajectory_step_count": len(trajectory),
                "candidate_step": step_index,
                "viewpoint": viewpoint,
                "has_decision_trace": decision is not None,
                "decision_trace_index": alignment.decision_trace_index,
                "is_route_intermediate": alignment.is_route_intermediate,
                "is_route_expanded_without_decision": alignment.is_route_expanded_without_decision,
                "route_step_offset": alignment.route_step_offset,
                "route_step_count": alignment.route_step_count,
                "is_final": step_index == final_step,
                "step_frac": endpoint_heuristic.step_fraction(step_index, len(trajectory)),
                "path_length_m": path_length_m,
                "stop_prob": decision.stop_prob if decision else None,
                "stop_margin_prob": decision.stop_margin_prob if decision else None,
                "selected_prob": decision.selected_prob if decision else None,
                "top1_top2_margin": top1_top2_margin(raw_step),
                "moe_router_entropy": decision.router_entropy if decision else None,
                "fuse_weight": endpoint_heuristic.parse_float(raw_step.get("fuse_weight")),
                "is_revisit": is_revisit,
                "is_loop_region": step_index in loop_region,
                "is_last_k": step_index >= last_k_start,
                "final_step": final_step,
                "final_success": final_success,
                "oracle_success": oracle_success,
                "should_rerank": should_rerank,
                "final_failure_bucket": final_failure_bucket,
                "success_label": success_label,
                "spl_at_candidate": spl_at_candidate,
                "distance_to_goal_m": distance_to_goal_m,
                "reward": reward,
                "is_best_success_candidate": False,
                "is_nearest_candidate": False,
            }
        )

    mark_label_ties(candidate_rows)
    pair_rows = build_pair_rows(candidate_rows)

    group_row = {
        "experiment_id": experiment_id,
        "dataset": source.dataset,
        "split": source.split,
        "protocol_split": protocol_split,
        "target_scope": target_scope.name,
        "episode_id": episode_id,
        "internal_item_id": internal_item_id,
        "saved_instr_id": saved_instr_id,
        "success_mode": target_scope.success_mode,
        "distance_key": target_scope.distance_key,
        "success_threshold_m": source.success_threshold_m,
        "trace_available": aligned_trace.trace_available,
        "trajectory_step_count": len(trajectory),
        "decision_trace_step_count": aligned_trace.raw_step_count,
        "candidate_count": len(candidate_rows),
        "candidates_with_decision_trace": sum(1 for row in candidate_rows if row["has_decision_trace"] is True),
        "route_intermediate_candidates": sum(1 for row in candidate_rows if row["is_route_intermediate"] is True),
        "route_expanded_without_decision_candidates": sum(
            1 for row in candidate_rows if row["is_route_expanded_without_decision"] is True
        ),
        "pair_count": len(pair_rows),
        "final_step": final_step,
        "first_success_step": first_success_step,
        "best_distance_step": best_distance_step,
        "final_viewpoint": oracle_gap.value_at(trajectory, final_step),
        "final_success": final_success,
        "oracle_success": oracle_success,
        "nearest_endpoint_success": nearest_endpoint_success,
        "should_rerank": should_rerank,
        "final_failure_bucket": final_failure_bucket,
        "final_distance_m": final_distance_m,
        "best_distance_m": best_distance_m,
        "final_path_length_m": final_path_length_m,
        "shortest_path_length_m": shortest_path_length_m,
        "final_spl": final_spl,
        "nearest_endpoint_spl": nearest_endpoint_spl,
        "success_candidate_count": sum(1 for row in candidate_rows if row["success_label"] is True),
        "failed_candidate_count": sum(1 for row in candidate_rows if row["success_label"] is False),
    }
    return EpisodeArtifacts(group_row=group_row, candidate_rows=candidate_rows, pair_rows=pair_rows)


def build_aligned_trace(trajectory: list[str], raw_steps: list[Any]) -> AlignedTrace:
    slots: list[dict[str, Any]] = [{} for _ in trajectory]
    decision_steps: list[endpoint_heuristic.DecisionStep] = []
    pointer = 0
    valid_raw_steps = [step for step in raw_steps if isinstance(step, dict)]

    for trace_index, raw_step in enumerate(valid_raw_steps):
        start = find_trace_start(trajectory, raw_step, pointer)
        if start is None:
            continue

        route = normalized_route(raw_step, trajectory[start])
        route_count = matched_route_count(trajectory, start, route)
        if route_count <= 0:
            route_count = 1

        for offset in range(route_count):
            step_index = start + offset
            if step_index >= len(slots):
                break
            slot = slots[step_index]
            slot.setdefault("route_step_offset", offset)
            slot.setdefault("route_step_count", route_count)
            if 0 < offset < route_count - 1 and "decision" not in slot:
                slot["is_route_intermediate"] = True

        decision = endpoint_heuristic.build_decision_step(start, trajectory[start], raw_step)
        slots[start].update(
            {
                "raw_step": raw_step,
                "decision": decision,
                "decision_trace_index": trace_index,
                "is_route_intermediate": False,
                "route_step_offset": 0,
                "route_step_count": route_count,
            }
        )
        decision_steps.append(decision)
        pointer = min(len(trajectory), start + max(route_count - 1, 0))

    alignments = tuple(slot_to_alignment(slot) for slot in slots)
    return AlignedTrace(
        trace_available=bool(valid_raw_steps),
        raw_step_count=len(valid_raw_steps),
        decision_steps=tuple(decision_steps),
        by_step=alignments,
    )


def slot_to_alignment(slot: dict[str, Any]) -> TraceAlignment:
    route_step_offset = slot.get("route_step_offset")
    decision = slot.get("decision")
    return TraceAlignment(
        raw_step=slot.get("raw_step"),
        decision=decision,
        decision_trace_index=slot.get("decision_trace_index"),
        is_route_intermediate=bool(slot.get("is_route_intermediate")),
        is_route_expanded_without_decision=bool(
            decision is None
            and route_step_offset is not None
            and int(route_step_offset) > 0
        ),
        route_step_offset=route_step_offset,
        route_step_count=slot.get("route_step_count"),
    )


def empty_alignment() -> TraceAlignment:
    return TraceAlignment(
        raw_step=None,
        decision=None,
        decision_trace_index=None,
        is_route_intermediate=False,
        is_route_expanded_without_decision=False,
        route_step_offset=None,
        route_step_count=None,
    )


def find_trace_start(trajectory: list[str], raw_step: dict[str, Any], pointer: int) -> int | None:
    if not trajectory:
        return None
    current = endpoint_heuristic.as_optional_string(raw_step.get("current_viewpoint"))
    if current is None:
        return None
    pointer = min(max(pointer, 0), len(trajectory) - 1)
    route = normalized_route(raw_step, current)

    if trajectory[pointer] == current and route_matches(trajectory, pointer, route):
        return pointer

    for index in range(pointer, len(trajectory)):
        if trajectory[index] == current and route_matches(trajectory, index, route):
            return index

    if trajectory[pointer] == current:
        return pointer

    for index in range(pointer, len(trajectory)):
        if trajectory[index] == current:
            return index
    return None


def normalized_route(raw_step: dict[str, Any], current_viewpoint: str | None) -> list[str]:
    route = raw_step.get("route_viewpoints")
    if not isinstance(route, list):
        route = []
    normalized = [str(viewpoint) for viewpoint in route if viewpoint is not None]
    if current_viewpoint is None:
        return normalized
    if not normalized:
        return [current_viewpoint]
    if normalized[0] != current_viewpoint:
        return [current_viewpoint] + normalized
    return normalized


def route_matches(trajectory: list[str], start: int, route: list[str]) -> bool:
    if start < 0 or start >= len(trajectory):
        return False
    if not route:
        return True
    if start + len(route) > len(trajectory):
        return False
    return all(trajectory[start + offset] == viewpoint for offset, viewpoint in enumerate(route))


def matched_route_count(trajectory: list[str], start: int, route: list[str]) -> int:
    if start < 0 or start >= len(trajectory):
        return 0
    if not route:
        return 1
    count = 0
    for offset, viewpoint in enumerate(route):
        step_index = start + offset
        if step_index >= len(trajectory) or trajectory[step_index] != viewpoint:
            break
        count += 1
    return max(count, 1)


def mark_label_ties(candidate_rows: list[dict[str, Any]]) -> None:
    success_spls = [
        endpoint_heuristic.parse_float(row.get("spl_at_candidate"))
        for row in candidate_rows
        if row.get("success_label") is True
    ]
    success_spls = [value for value in success_spls if value is not None]
    best_success_spl = max(success_spls) if success_spls else None

    distances = [
        endpoint_heuristic.parse_float(row.get("distance_to_goal_m"))
        for row in candidate_rows
    ]
    distances = [value for value in distances if value is not None]
    nearest_distance = min(distances) if distances else None

    for row in candidate_rows:
        spl = endpoint_heuristic.parse_float(row.get("spl_at_candidate"))
        distance = endpoint_heuristic.parse_float(row.get("distance_to_goal_m"))
        row["is_best_success_candidate"] = bool(
            row.get("success_label") is True
            and best_success_spl is not None
            and spl is not None
            and spl >= best_success_spl - EPS
        )
        row["is_nearest_candidate"] = bool(
            nearest_distance is not None
            and distance is not None
            and distance <= nearest_distance + EPS
        )


def build_pair_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    success_rows = [row for row in candidate_rows if row.get("success_label") is True]
    failed_rows = [row for row in candidate_rows if row.get("success_label") is False]

    for chosen in success_rows:
        for rejected in failed_rows:
            pair_rows.append(make_pair_row(chosen, rejected, "success_gt_fail", len(pair_rows)))

    for chosen in success_rows:
        chosen_spl = endpoint_heuristic.parse_float(chosen.get("spl_at_candidate")) or 0.0
        for rejected in success_rows:
            if chosen is rejected:
                continue
            rejected_spl = endpoint_heuristic.parse_float(rejected.get("spl_at_candidate")) or 0.0
            if chosen_spl > rejected_spl + EPS:
                pair_rows.append(
                    make_pair_row(
                        chosen,
                        rejected,
                        "better_spl_success_gt_lower_spl_success",
                        len(pair_rows),
                    )
                )

    final_rows = [row for row in candidate_rows if row.get("is_final")]
    final_row = final_rows[0] if final_rows else None
    if final_row is not None and final_row.get("final_success") is True:
        for rejected in failed_rows:
            if rejected.get("is_final"):
                continue
            pair_rows.append(
                make_pair_row(
                    final_row,
                    rejected,
                    "final_success_final_gt_failed_nonfinal",
                    len(pair_rows),
                )
            )
    return pair_rows


def make_pair_row(
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    pair_index: int,
) -> dict[str, Any]:
    episode_id = str(chosen["episode_id"])
    return {
        "experiment_id": chosen["experiment_id"],
        "dataset": chosen["dataset"],
        "split": chosen["split"],
        "protocol_split": chosen["protocol_split"],
        "target_scope": chosen["target_scope"],
        "episode_id": episode_id,
        "pair_id": f"{episode_id}:pair_{pair_index:06d}",
        "pair_type": pair_type,
        "chosen_candidate_id": chosen["candidate_id"],
        "rejected_candidate_id": rejected["candidate_id"],
        "chosen_step": chosen["candidate_step"],
        "rejected_step": rejected["candidate_step"],
        "chosen_viewpoint": chosen["viewpoint"],
        "rejected_viewpoint": rejected["viewpoint"],
        "chosen_success_label": chosen["success_label"],
        "rejected_success_label": rejected["success_label"],
        "chosen_spl": chosen["spl_at_candidate"],
        "rejected_spl": rejected["spl_at_candidate"],
        "chosen_reward": chosen["reward"],
        "rejected_reward": rejected["reward"],
        "chosen_is_final": chosen["is_final"],
        "rejected_is_final": rejected["is_final"],
        "should_rerank": chosen["should_rerank"],
    }


def top1_top2_margin(raw_step: dict[str, Any]) -> float | None:
    if not raw_step:
        return None
    score_name, candidates = endpoint_heuristic.score_space(raw_step)
    probs = [
        endpoint_heuristic.candidate_score(candidate, score_name, "prob")
        for candidate in candidates
    ]
    probs = [prob for prob in probs if prob is not None]
    if len(probs) < 2:
        return None
    probs.sort(reverse=True)
    return probs[0] - probs[1]


def loop_region_indices(
    trajectory: list[str],
    decision_steps: tuple[endpoint_heuristic.DecisionStep, ...],
    window: int,
) -> set[int]:
    loop_indices: set[int] = set()
    last_seen: dict[str, int] = {}
    for index, viewpoint in enumerate(trajectory):
        previous = last_seen.get(viewpoint)
        if previous is not None and 2 <= index - previous <= window:
            loop_indices.update(range(previous, index + 1))
        last_seen[viewpoint] = index
    for step in decision_steps:
        if step.route_is_backtrack:
            loop_indices.add(step.index)
    return loop_indices


def protocol_split_for(
    dataset: str,
    split: str,
    internal_item_id: str,
    dev_ratio: float,
    salt: str,
) -> str:
    if split == "val_train_seen":
        fraction = stable_fraction(f"{salt}:{dataset}:{internal_item_id}")
        return "dev" if fraction < dev_ratio else "train"
    if split in {"train", "train_eval", "train_seen"}:
        return "train"
    if split in {"val_seen", "dev", "validation"}:
        return "dev"
    if split == "val_unseen":
        return "test"
    return split


def stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12)


def make_episode_id(dataset: str, split: str, target_scope: str, internal_item_id: str) -> str:
    return f"{dataset}:{split}:{target_scope}:{internal_item_id}"


def make_candidate_id(episode_id: str, step_index: int) -> str:
    return f"{episode_id}:step_{step_index:03d}"


def build_protocol_summary(
    group_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_map: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    candidate_map: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    pair_map: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in group_rows:
        group_map.setdefault(summary_key(row), []).append(row)
    for row in candidate_rows:
        candidate_map.setdefault(summary_key(row), []).append(row)
    for row in pair_rows:
        pair_map.setdefault(summary_key(row), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(set(group_map) | set(candidate_map) | set(pair_map), key=summary_sort_key):
        experiment_id, dataset, split, protocol_split, target_scope = key
        groups = group_map.get(key, [])
        candidates = candidate_map.get(key, [])
        pairs = pair_map.get(key, [])
        pair_type_counts = {
            pair_type: sum(1 for row in pairs if row.get("pair_type") == pair_type)
            for pair_type in PAIR_TYPES
        }
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "split": split,
                "protocol_split": protocol_split,
                "target_scope": target_scope,
                "episodes": len(groups),
                "candidates": len(candidates),
                "pairs": len(pairs),
                "should_rerank_episodes": sum(1 for row in groups if row.get("should_rerank") is True),
                "should_rerank_rate": mean_bool(groups, "should_rerank"),
                "final_success_rate": mean_bool(groups, "final_success"),
                "oracle_success_rate": mean_bool(groups, "oracle_success"),
                "success_candidates": sum(1 for row in candidates if row.get("success_label") is True),
                "failed_candidates": sum(1 for row in candidates if row.get("success_label") is False),
                "decision_trace_candidates": sum(1 for row in candidates if row.get("has_decision_trace") is True),
                "decision_trace_candidate_rate": mean_bool(candidates, "has_decision_trace"),
                "route_intermediate_candidates": sum(
                    1 for row in candidates if row.get("is_route_intermediate") is True
                ),
                "route_intermediate_candidate_rate": mean_bool(candidates, "is_route_intermediate"),
                "route_expanded_without_decision_candidates": sum(
                    1 for row in candidates if row.get("is_route_expanded_without_decision") is True
                ),
                "route_expanded_without_decision_candidate_rate": mean_bool(
                    candidates,
                    "is_route_expanded_without_decision",
                ),
                "success_gt_fail_pairs": pair_type_counts["success_gt_fail"],
                "better_spl_success_gt_lower_spl_success_pairs": pair_type_counts[
                    "better_spl_success_gt_lower_spl_success"
                ],
                "final_success_final_gt_failed_nonfinal_pairs": pair_type_counts[
                    "final_success_final_gt_failed_nonfinal"
                ],
            }
        )
    return summary_rows


def summary_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["experiment_id"]),
        str(row["dataset"]),
        str(row["split"]),
        str(row["protocol_split"]),
        str(row["target_scope"]),
    )


def summary_sort_key(key: tuple[str, str, str, str, str]) -> tuple[int, str, str, int, str]:
    _, dataset, split, protocol_split, target_scope = key
    dataset_order = {"R2R": 0, "REVERIE": 1, "SOON": 2, "CVDN": 3}
    split_order = {"val_train_seen": 0, "train": 1, "val_seen": 2, "val_unseen": 3}
    protocol_order = {"train": 0, "dev": 1, "test": 2}
    return (
        dataset_order.get(dataset, 99),
        split,
        protocol_split,
        split_order.get(split, 99),
        f"{protocol_order.get(protocol_split, 99)}:{target_scope}",
    )


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1.0 for value in values if bool(value)) / len(values)


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
