#!/usr/bin/env python3
"""Build no-training endpoint heuristic reranking reports from SAME eval outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_oracle_gap_report as oracle_gap  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_NAME = "endpoint_heuristic_rerank"
SCHEMA_VERSION = "endpoint_heuristic_report.v1"
EPS = 1e-9

DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_LAST_K_VALUES = (3, 5, 7)
DEFAULT_SCOPES = oracle_gap.DEFAULT_SCOPES

ITEM_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "target_scope",
    "heuristic",
    "heuristic_params",
    "internal_item_id",
    "saved_instr_id",
    "success_mode",
    "distance_key",
    "success_threshold_m",
    "trace_available",
    "trajectory_step_count",
    "final_step",
    "selected_step",
    "selected_step_delta",
    "selected_step_frac",
    "selected_reason",
    "selected_changed",
    "final_viewpoint",
    "selected_viewpoint",
    "final_failure_bucket",
    "final_success",
    "heuristic_success",
    "oracle_success",
    "nearest_endpoint_success",
    "recovered_by_heuristic",
    "harmed_by_heuristic",
    "recovered_by_nearest_endpoint",
    "final_distance_m",
    "selected_distance_m",
    "best_distance_m",
    "selected_minus_best_distance_m",
    "final_path_length_m",
    "selected_path_length_m",
    "shortest_path_length_m",
    "final_spl",
    "heuristic_spl",
    "nearest_endpoint_spl",
    "final_stop_prob",
    "selected_stop_prob",
    "final_stop_margin_prob",
    "selected_stop_margin_prob",
    "final_router_entropy",
    "selected_router_entropy",
]

SUMMARY_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "target_scope",
    "heuristic",
    "heuristic_params",
    "items",
    "trace_available_rate",
    "changed_endpoint_rate",
    "final_success_rate",
    "heuristic_success_rate",
    "nearest_endpoint_success_rate",
    "delta_success_rate",
    "gap_capture_rate",
    "recovery_rate",
    "harm_rate",
    "net_recovery_rate",
    "final_spl",
    "heuristic_spl",
    "nearest_endpoint_spl",
    "delta_spl",
    "overshoot_items",
    "overshoot_recovery_rate",
    "overshoot_harm_rate",
    "stop_too_early_proxy_items",
    "stop_too_early_proxy_recovery_rate",
    "stop_too_early_proxy_harm_rate",
    "never_reached_items",
    "never_reached_recovery_rate",
    "never_reached_harm_rate",
    "mean_selected_step_delta",
    "mean_final_stop_prob",
    "mean_selected_stop_prob",
]


@dataclass(frozen=True)
class DecisionStep:
    index: int
    viewpoint: str | None
    stop_prob: float | None
    stop_logit: float | None
    best_move_prob: float | None
    best_move_logit: float | None
    stop_margin_prob: float | None
    stop_margin_logit: float | None
    selected_prob: float | None
    selected_kind: str | None
    selected_viewpoint: str | None
    executed_viewpoint: str | None
    router_entropy: float | None
    route_len: int
    route_is_backtrack: bool


@dataclass(frozen=True)
class SanitizedEpisode:
    trajectory: tuple[str, ...]
    decision_steps: tuple[DecisionStep, ...]
    trace_available: bool

    @property
    def final_step(self) -> int | None:
        if not self.trajectory:
            return None
        return len(self.trajectory) - 1


@dataclass(frozen=True)
class Selection:
    step: int | None
    reason: str


@dataclass(frozen=True)
class HeuristicSpec:
    name: str
    params: str
    selector: Callable[[SanitizedEpisode], Selection]

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return f"{self.name}:{self.params}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build no-training endpoint heuristic reranking reports from SAME eval_items.",
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
        "--scopes",
        default=",".join(DEFAULT_SCOPES),
        help="Comma-separated scopes to report: official,goal,region,region_threshold.",
    )
    parser.add_argument(
        "--heuristics",
        default="all",
        help=(
            "Comma-separated heuristic families, or 'all'. Families: final,max_stop_prob,"
            "max_stop_margin,last_k_max_stop,last_k_max_stop_margin,first_stop_threshold,"
            "last_stop_threshold,last_high_stop_before_move,loop_guard,conservative_rerank."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(format_float(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated stop-prob thresholds used by threshold heuristics.",
    )
    parser.add_argument(
        "--last-k-values",
        default=",".join(str(value) for value in DEFAULT_LAST_K_VALUES),
        help="Comma-separated K values used by last-k heuristics.",
    )
    parser.add_argument(
        "--loop-window",
        type=int,
        default=10,
        help="Maximum revisit window for loop_guard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    fine_metrics_dir = Path(args.fine_metrics_dir) if args.fine_metrics_dir else experiment_dir / "fine_metrics"
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / DEFAULT_OUTPUT_NAME
    scopes = tuple(scope.strip() for scope in args.scopes.split(",") if scope.strip())
    thresholds = tuple(parse_float_list(args.thresholds))
    last_k_values = tuple(parse_int_list(args.last_k_values))
    heuristic_families = tuple(name.strip() for name in args.heuristics.split(",") if name.strip())

    build_endpoint_heuristic_report(
        experiment_dir=experiment_dir,
        fine_metrics_dir=fine_metrics_dir,
        output_dir=output_dir,
        scopes=scopes,
        heuristic_families=heuristic_families,
        thresholds=thresholds,
        last_k_values=last_k_values,
        loop_window=args.loop_window,
    )


def build_endpoint_heuristic_report(
    experiment_dir: Path,
    fine_metrics_dir: Path,
    output_dir: Path,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    heuristic_families: tuple[str, ...] = ("all",),
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    last_k_values: tuple[int, ...] = DEFAULT_LAST_K_VALUES,
    loop_window: int = 10,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    fine_metrics_dir = fine_metrics_dir.resolve()
    output_dir = output_dir.resolve()
    eval_items_dir = experiment_dir / "eval_items"

    sources = oracle_gap.discover_eval_item_sources(eval_items_dir)
    if not sources:
        raise FileNotFoundError(f"No eval_items contexts found in {eval_items_dir}")

    fine_rows = oracle_gap.load_fine_metrics_wide(fine_metrics_dir / "tables" / "fine_metrics_wide.csv")
    heuristics = build_heuristic_specs(
        heuristic_families=heuristic_families,
        thresholds=thresholds,
        last_k_values=last_k_values,
        loop_window=loop_window,
    )
    if not heuristics:
        raise ValueError("No heuristics selected")

    output_dir.mkdir(parents=True, exist_ok=True)

    item_rows: list[dict[str, Any]] = []
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
            episode = build_sanitized_episode(item)
            for scope_name in scopes:
                target_scope = oracle_gap.build_target_scope(item, source.dataset, scope_name)
                if target_scope is None:
                    continue
                eval_context = build_eval_context(item, source, fine_row, target_scope)
                for heuristic in heuristics:
                    selection = normalize_selection(heuristic.selector(episode), episode)
                    item_rows.append(
                        build_item_row(
                            item=item,
                            source=source,
                            target_scope=target_scope,
                            eval_context=eval_context,
                            episode=episode,
                            heuristic=heuristic,
                            selection=selection,
                        )
                    )

    summary_rows = build_summary_rows(item_rows)

    item_csv_path = output_dir / "endpoint_heuristic_items.csv"
    summary_csv_path = output_dir / "endpoint_heuristic_summary.csv"
    report_path = output_dir / "endpoint_heuristic_report.md"
    manifest_path = output_dir / "manifest.json"

    write_csv(item_csv_path, ITEM_FIELDNAMES, item_rows)
    write_csv(summary_csv_path, SUMMARY_FIELDNAMES, summary_rows)
    report_path.write_text(
        build_markdown_report(experiment_dir.name, item_rows, summary_rows),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": experiment_dir.name,
        "experiment_dir": path_to_string(experiment_dir),
        "fine_metrics_dir": path_to_string(fine_metrics_dir),
        "output_dir": path_to_string(output_dir),
        "scopes": list(scopes),
        "heuristics": [{"name": spec.name, "params": spec.params, "label": spec.label} for spec in heuristics],
        "thresholds": list(thresholds),
        "last_k_values": list(last_k_values),
        "loop_window": loop_window,
        "files": {
            "items_csv": path_to_string(item_csv_path),
            "summary_csv": path_to_string(summary_csv_path),
            "report_md": path_to_string(report_path),
        },
        "counts": {
            "eval_item_sources": len(sources),
            "heuristics": len(heuristics),
            "item_scope_heuristic_rows": len(item_rows),
            "summary_rows": len(summary_rows),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_sanitized_episode(item: dict[str, Any]) -> SanitizedEpisode:
    prediction = item.get("prediction", {})
    trajectory = tuple(str(viewpoint) for viewpoint in (prediction.get("trajectory") or []))
    raw_steps = (prediction.get("decision_trace") or {}).get("steps") or []
    decision_steps: list[DecisionStep] = []
    for index, viewpoint in enumerate(trajectory):
        raw_step = raw_steps[index] if index < len(raw_steps) and isinstance(raw_steps[index], dict) else {}
        decision_steps.append(build_decision_step(index, viewpoint, raw_step))
    return SanitizedEpisode(
        trajectory=trajectory,
        decision_steps=tuple(decision_steps),
        trace_available=bool(raw_steps),
    )


def build_decision_step(index: int, trajectory_viewpoint: str | None, raw_step: dict[str, Any]) -> DecisionStep:
    stop_prob = parse_float(raw_step.get("stop_prob"))
    score_name, candidates = score_space(raw_step)
    stop_candidate = first_candidate(candidates, viewpoint_is_none=True)
    if stop_prob is None:
        stop_prob = candidate_score(stop_candidate, score_name, "prob")
    stop_logit = candidate_score(stop_candidate, score_name, "logit")
    best_move_prob, best_move_logit = best_move_score(candidates, score_name)
    stop_margin_prob = subtract_or_none(stop_prob, best_move_prob)
    stop_margin_logit = subtract_or_none(stop_logit, best_move_logit)

    selected = raw_step.get("selected") or {}
    route = raw_step.get("route_viewpoints") or []
    moe = raw_step.get("moe") or {}
    return DecisionStep(
        index=index,
        viewpoint=str(raw_step.get("current_viewpoint") or trajectory_viewpoint) if raw_step or trajectory_viewpoint else None,
        stop_prob=stop_prob,
        stop_logit=stop_logit,
        best_move_prob=best_move_prob,
        best_move_logit=best_move_logit,
        stop_margin_prob=stop_margin_prob,
        stop_margin_logit=stop_margin_logit,
        selected_prob=parse_float(selected.get("prob")),
        selected_kind=as_optional_string(selected.get("selection_kind")),
        selected_viewpoint=as_optional_string(selected.get("viewpoint")),
        executed_viewpoint=as_optional_string(selected.get("executed_viewpoint")),
        router_entropy=parse_float(moe.get("router_entropy")),
        route_len=len(route) if isinstance(route, list) else 0,
        route_is_backtrack=bool(isinstance(route, list) and len(route) > 2),
    )


def score_space(raw_step: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    fusion = raw_step.get("fusion")
    if fusion == "local":
        return "local", list(raw_step.get("local_candidates") or [])
    if fusion == "global":
        return "global", list(raw_step.get("gmap_candidates") or [])
    return "fused", list(raw_step.get("gmap_candidates") or [])


def first_candidate(candidates: list[dict[str, Any]], viewpoint_is_none: bool) -> dict[str, Any] | None:
    for candidate in candidates:
        viewpoint = candidate.get("viewpoint")
        if viewpoint_is_none and viewpoint is None:
            return candidate
        if not viewpoint_is_none and viewpoint is not None:
            return candidate
    return None


def candidate_score(candidate: dict[str, Any] | None, score_name: str, field: str) -> float | None:
    if candidate is None:
        return None
    return parse_float((candidate.get(score_name) or {}).get(field))


def best_move_score(candidates: list[dict[str, Any]], score_name: str) -> tuple[float | None, float | None]:
    best_prob: float | None = None
    best_logit: float | None = None
    for candidate in candidates:
        if candidate.get("viewpoint") is None:
            continue
        if candidate.get("valid") is False or candidate.get("actionable") is False:
            continue
        prob = candidate_score(candidate, score_name, "prob")
        logit = candidate_score(candidate, score_name, "logit")
        if prob is not None and (best_prob is None or prob > best_prob):
            best_prob = prob
        if logit is not None and (best_logit is None or logit > best_logit):
            best_logit = logit
    return best_prob, best_logit


def build_heuristic_specs(
    heuristic_families: tuple[str, ...],
    thresholds: tuple[float, ...],
    last_k_values: tuple[int, ...],
    loop_window: int,
) -> list[HeuristicSpec]:
    all_families = {
        "final",
        "max_stop_prob",
        "max_stop_margin",
        "last_k_max_stop",
        "last_k_max_stop_margin",
        "first_stop_threshold",
        "last_stop_threshold",
        "last_high_stop_before_move",
        "loop_guard",
        "conservative_rerank",
    }
    selected = set(all_families if "all" in heuristic_families else heuristic_families)
    unknown = selected.difference(all_families)
    if unknown:
        raise ValueError(f"Unknown heuristic families: {', '.join(sorted(unknown))}")

    specs: list[HeuristicSpec] = []
    if "final" in selected:
        specs.append(HeuristicSpec("final", "", select_final))
    if "max_stop_prob" in selected:
        specs.append(HeuristicSpec("max_stop_prob", "", select_max_stop_prob))
    if "max_stop_margin" in selected:
        specs.append(HeuristicSpec("max_stop_margin", "score=prob_margin", select_max_stop_margin))
    if "last_k_max_stop" in selected:
        for k_value in last_k_values:
            specs.append(
                HeuristicSpec(
                    "last_k_max_stop",
                    f"k={k_value}",
                    lambda episode, k_value=k_value: select_last_k_max_stop(episode, k_value),
                )
            )
    if "last_k_max_stop_margin" in selected:
        for k_value in last_k_values:
            specs.append(
                HeuristicSpec(
                    "last_k_max_stop_margin",
                    f"k={k_value},score=prob_margin",
                    lambda episode, k_value=k_value: select_last_k_max_stop_margin(episode, k_value),
                )
            )
    if "first_stop_threshold" in selected:
        for threshold in thresholds:
            specs.append(
                HeuristicSpec(
                    "first_stop_threshold",
                    f"threshold={format_float(threshold)}",
                    lambda episode, threshold=threshold: select_threshold_stop(episode, threshold, first=True),
                )
            )
    if "last_stop_threshold" in selected:
        for threshold in thresholds:
            specs.append(
                HeuristicSpec(
                    "last_stop_threshold",
                    f"threshold={format_float(threshold)}",
                    lambda episode, threshold=threshold: select_threshold_stop(episode, threshold, first=False),
                )
            )
    if "last_high_stop_before_move" in selected:
        for threshold in thresholds:
            specs.append(
                HeuristicSpec(
                    "last_high_stop_before_move",
                    f"threshold={format_float(threshold)}",
                    lambda episode, threshold=threshold: select_last_high_stop_before_move(episode, threshold),
                )
            )
    if "loop_guard" in selected:
        for threshold in (0.1, 0.2, 0.3):
            specs.append(
                HeuristicSpec(
                    "loop_guard",
                    f"threshold={format_float(threshold)},window={loop_window}",
                    lambda episode, threshold=threshold: select_loop_guard(
                        episode,
                        threshold=threshold,
                        window=loop_window,
                    ),
                )
            )
    if "conservative_rerank" in selected:
        for min_stop, delta in ((0.2, 0.05), (0.3, 0.05), (0.3, 0.1)):
            specs.append(
                HeuristicSpec(
                    "conservative_rerank",
                    f"k=5,min_stop={format_float(min_stop)},delta={format_float(delta)},window={loop_window}",
                    lambda episode, min_stop=min_stop, delta=delta: select_conservative_rerank(
                        episode,
                        k_value=5,
                        min_stop=min_stop,
                        delta=delta,
                        loop_window=loop_window,
                    ),
                )
            )
    return specs


def normalize_selection(selection: Selection, episode: SanitizedEpisode) -> Selection:
    final_step = episode.final_step
    if final_step is None:
        return Selection(None, "empty_trajectory")
    if selection.step is None or selection.step < 0 or selection.step >= len(episode.trajectory):
        return Selection(final_step, f"{selection.reason}|fallback_final")
    return selection


def select_final(episode: SanitizedEpisode) -> Selection:
    return Selection(episode.final_step, "final_endpoint")


def select_max_stop_prob(episode: SanitizedEpisode) -> Selection:
    return select_max_by_feature(
        episode,
        start=0,
        end=len(episode.decision_steps),
        feature=lambda step: step.stop_prob,
        reason="max_stop_prob",
    )


def select_max_stop_margin(episode: SanitizedEpisode) -> Selection:
    return select_max_by_feature(
        episode,
        start=0,
        end=len(episode.decision_steps),
        feature=lambda step: step.stop_margin_prob,
        reason="max_stop_margin_prob",
    )


def select_last_k_max_stop(episode: SanitizedEpisode, k_value: int) -> Selection:
    end = len(episode.decision_steps)
    start = max(0, end - max(k_value, 1))
    return select_max_by_feature(
        episode,
        start=start,
        end=end,
        feature=lambda step: step.stop_prob,
        reason=f"last_{k_value}_max_stop_prob",
    )


def select_last_k_max_stop_margin(episode: SanitizedEpisode, k_value: int) -> Selection:
    end = len(episode.decision_steps)
    start = max(0, end - max(k_value, 1))
    return select_max_by_feature(
        episode,
        start=start,
        end=end,
        feature=lambda step: step.stop_margin_prob,
        reason=f"last_{k_value}_max_stop_margin_prob",
    )


def select_threshold_stop(episode: SanitizedEpisode, threshold: float, first: bool) -> Selection:
    candidates = [
        step
        for step in episode.decision_steps
        if step.stop_prob is not None and step.stop_prob >= threshold
    ]
    if not candidates:
        return Selection(episode.final_step, f"no_stop_prob_ge_{format_float(threshold)}")
    selected = candidates[0] if first else candidates[-1]
    prefix = "first" if first else "last"
    return Selection(selected.index, f"{prefix}_stop_prob_ge_{format_float(threshold)}")


def select_last_high_stop_before_move(episode: SanitizedEpisode, threshold: float) -> Selection:
    candidates = [
        step
        for step in episode.decision_steps
        if step.index != episode.final_step
        and step.stop_prob is not None
        and step.stop_prob >= threshold
        and step.selected_kind != "stop"
    ]
    if not candidates:
        return Selection(episode.final_step, f"no_high_stop_before_move_ge_{format_float(threshold)}")
    selected = max(candidates, key=lambda step: (step.stop_prob or -math.inf, step.index))
    return Selection(selected.index, f"last_high_stop_before_move_ge_{format_float(threshold)}")


def select_loop_guard(episode: SanitizedEpisode, threshold: float = 0.1, window: int = 10) -> Selection:
    loop_index = first_loop_index(episode, window=window)
    if loop_index is None:
        return Selection(episode.final_step, "no_loop")

    start = max(0, loop_index - max(window, 1))
    end = min(len(episode.decision_steps), loop_index + 1)
    candidates = [
        step
        for step in episode.decision_steps[start:end]
        if step.stop_prob is not None and step.stop_prob >= threshold
    ]
    if not candidates:
        return Selection(episode.final_step, f"loop_without_stop_ge_{format_float(threshold)}")
    selected = max(candidates, key=lambda step: (step.stop_prob or -math.inf, step.index))
    return Selection(selected.index, f"loop_guard_at_{loop_index}")


def select_conservative_rerank(
    episode: SanitizedEpisode,
    k_value: int = 5,
    min_stop: float = 0.2,
    delta: float = 0.05,
    loop_window: int = 10,
) -> Selection:
    final_step = episode.final_step
    if final_step is None:
        return Selection(None, "empty_trajectory")

    final_decision = decision_at(episode, final_step)
    final_stop = final_decision.stop_prob if final_decision else None
    if final_stop is None:
        final_stop = 0.0

    candidates: list[tuple[DecisionStep, str]] = []
    last_k_selection = select_last_k_max_stop(episode, k_value)
    last_k_step = decision_at(episode, last_k_selection.step)
    if last_k_step is not None:
        candidates.append((last_k_step, "last_k"))

    high_move_selection = select_last_high_stop_before_move(episode, min_stop)
    high_move_step = decision_at(episode, high_move_selection.step)
    if high_move_step is not None and high_move_step.index != final_step:
        candidates.append((high_move_step, "high_stop_before_move"))

    loop_selection = select_loop_guard(episode, threshold=min_stop / 2.0, window=loop_window)
    loop_step = decision_at(episode, loop_selection.step)
    if loop_step is not None and loop_step.index != final_step:
        candidates.append((loop_step, "loop_guard"))

    if not candidates:
        return Selection(final_step, "no_candidate")

    min_index = max(0, int(math.floor(final_step * 0.35)))
    viable: list[tuple[float, int, str, DecisionStep]] = []
    for step, source in candidates:
        stop_prob = step.stop_prob
        if step.index == final_step or step.index < min_index or stop_prob is None:
            continue
        if stop_prob < min_stop:
            continue
        if stop_prob < final_stop + delta and source != "loop_guard":
            continue
        if source == "loop_guard" and stop_prob + EPS < final_stop:
            continue
        viable.append((stop_prob, step.index, source, step))

    if not viable:
        return Selection(final_step, "candidate_not_confident")

    _, _, source, selected = max(viable)
    return Selection(selected.index, f"conservative_{source}")


def select_max_by_feature(
    episode: SanitizedEpisode,
    start: int,
    end: int,
    feature: Callable[[DecisionStep], float | None],
    reason: str,
) -> Selection:
    values: list[tuple[float, int]] = []
    for step in episode.decision_steps[start:end]:
        value = feature(step)
        if value is not None and math.isfinite(value):
            values.append((value, step.index))
    if not values:
        return Selection(episode.final_step, f"{reason}|no_score")
    _, index = max(values)
    return Selection(index, reason)


def first_loop_index(episode: SanitizedEpisode, window: int) -> int | None:
    last_seen: dict[str, int] = {}
    for index, viewpoint in enumerate(episode.trajectory):
        previous = last_seen.get(viewpoint)
        if previous is not None and index - previous >= 2 and index - previous <= window:
            return index
        last_seen[viewpoint] = index
    for step in episode.decision_steps:
        if step.route_is_backtrack:
            return step.index
    return None


def decision_at(episode: SanitizedEpisode, step: int | None) -> DecisionStep | None:
    if step is None or step < 0 or step >= len(episode.decision_steps):
        return None
    return episode.decision_steps[step]


def build_eval_context(
    item: dict[str, Any],
    source: oracle_gap.EvalItemSource,
    fine_row: dict[str, str],
    target_scope: oracle_gap.TargetScope,
) -> dict[str, Any]:
    trajectory = item.get("prediction", {}).get("trajectory") or []
    distances = oracle_gap.distance_series(item, target_scope)
    cumulative_lengths = oracle_gap.as_float_list(
        item.get("primitives", {}).get("trajectory_cumulative_lengths_m") or []
    )
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
        and final_distance_m <= best_distance_m + EPS
    )
    failure_bucket = final_failure_bucket(
        final_success=final_success,
        oracle_success=oracle_success,
        final_is_best_distance=final_is_best_distance,
    )
    shortest_path_length_m = parse_float(fine_row.get(f"{target_scope.metric_group}.shortest_path_length_m"))
    final_path_length_m = oracle_gap.value_at(cumulative_lengths, final_step)
    best_endpoint_path_length_m = oracle_gap.value_at(cumulative_lengths, best_distance_step)
    final_spl = parse_float(fine_row.get(f"{target_scope.metric_group}.spl"))
    if final_spl is None:
        final_spl = oracle_gap.spl_at(final_success, final_path_length_m, shortest_path_length_m)
    return {
        "trajectory": trajectory,
        "distances": distances,
        "cumulative_lengths": cumulative_lengths,
        "final_step": final_step,
        "best_distance_step": best_distance_step,
        "final_success": final_success,
        "oracle_success": oracle_success,
        "nearest_endpoint_success": nearest_endpoint_success,
        "final_failure_bucket": failure_bucket,
        "final_distance_m": final_distance_m,
        "best_distance_m": best_distance_m,
        "final_path_length_m": final_path_length_m,
        "best_endpoint_path_length_m": best_endpoint_path_length_m,
        "shortest_path_length_m": shortest_path_length_m,
        "final_spl": final_spl,
        "nearest_endpoint_spl": oracle_gap.spl_at(
            nearest_endpoint_success,
            best_endpoint_path_length_m,
            shortest_path_length_m,
        ),
    }


def build_item_row(
    item: dict[str, Any],
    source: oracle_gap.EvalItemSource,
    target_scope: oracle_gap.TargetScope,
    eval_context: dict[str, Any],
    episode: SanitizedEpisode,
    heuristic: HeuristicSpec,
    selection: Selection,
) -> dict[str, Any]:
    identity = item.get("identity", {})
    trajectory = eval_context["trajectory"]
    distances = eval_context["distances"]
    cumulative_lengths = eval_context["cumulative_lengths"]
    final_step = eval_context["final_step"]
    selected_step = selection.step
    final_decision = decision_at(episode, final_step)
    selected_decision = decision_at(episode, selected_step)

    heuristic_success = oracle_gap.success_at_step(
        trajectory,
        distances,
        selected_step,
        source.success_threshold_m,
        target_scope,
    )
    selected_distance_m = oracle_gap.value_at(distances, selected_step)
    selected_path_length_m = oracle_gap.value_at(cumulative_lengths, selected_step)
    heuristic_spl = oracle_gap.spl_at(
        heuristic_success,
        selected_path_length_m,
        eval_context["shortest_path_length_m"],
    )
    final_success = eval_context["final_success"]
    selected_step_delta = subtract_or_none(selected_step, final_step)

    return {
        "experiment_id": source.items_path.parents[1].name,
        "dataset": source.dataset,
        "split": source.split,
        "target_scope": target_scope.name,
        "heuristic": heuristic.name,
        "heuristic_params": heuristic.params,
        "internal_item_id": identity.get("internal_item_id"),
        "saved_instr_id": identity.get("saved_instr_id"),
        "success_mode": target_scope.success_mode,
        "distance_key": target_scope.distance_key,
        "success_threshold_m": source.success_threshold_m,
        "trace_available": episode.trace_available,
        "trajectory_step_count": len(trajectory),
        "final_step": final_step,
        "selected_step": selected_step,
        "selected_step_delta": selected_step_delta,
        "selected_step_frac": step_fraction(selected_step, len(trajectory)),
        "selected_reason": selection.reason,
        "selected_changed": bool(selected_step is not None and final_step is not None and selected_step != final_step),
        "final_viewpoint": oracle_gap.value_at(trajectory, final_step),
        "selected_viewpoint": oracle_gap.value_at(trajectory, selected_step),
        "final_failure_bucket": eval_context["final_failure_bucket"],
        "final_success": final_success,
        "heuristic_success": heuristic_success,
        "oracle_success": eval_context["oracle_success"],
        "nearest_endpoint_success": eval_context["nearest_endpoint_success"],
        "recovered_by_heuristic": bool(final_success is False and heuristic_success is True),
        "harmed_by_heuristic": bool(final_success is True and heuristic_success is False),
        "recovered_by_nearest_endpoint": bool(
            final_success is False and eval_context["nearest_endpoint_success"] is True
        ),
        "final_distance_m": eval_context["final_distance_m"],
        "selected_distance_m": selected_distance_m,
        "best_distance_m": eval_context["best_distance_m"],
        "selected_minus_best_distance_m": subtract_or_none(
            parse_float(selected_distance_m),
            parse_float(eval_context["best_distance_m"]),
        ),
        "final_path_length_m": eval_context["final_path_length_m"],
        "selected_path_length_m": selected_path_length_m,
        "shortest_path_length_m": eval_context["shortest_path_length_m"],
        "final_spl": eval_context["final_spl"],
        "heuristic_spl": heuristic_spl,
        "nearest_endpoint_spl": eval_context["nearest_endpoint_spl"],
        "final_stop_prob": final_decision.stop_prob if final_decision else None,
        "selected_stop_prob": selected_decision.stop_prob if selected_decision else None,
        "final_stop_margin_prob": final_decision.stop_margin_prob if final_decision else None,
        "selected_stop_margin_prob": selected_decision.stop_margin_prob if selected_decision else None,
        "final_router_entropy": final_decision.router_entropy if final_decision else None,
        "selected_router_entropy": selected_decision.router_entropy if selected_decision else None,
    }


def final_failure_bucket(
    final_success: bool | None,
    oracle_success: bool,
    final_is_best_distance: bool,
) -> str:
    if final_success is True:
        return "final_success"
    if oracle_success:
        return "overshoot"
    if final_is_best_distance:
        return "stop_too_early_proxy"
    return "never_reached"


def build_summary_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in item_rows:
        key = (
            row["experiment_id"],
            row["dataset"],
            row["split"],
            row["target_scope"],
            row["heuristic"],
            row["heuristic_params"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=summary_sort_key):
        experiment_id, dataset, split, target_scope, heuristic, params = key
        final_rate = mean_bool(rows, "final_success")
        heuristic_rate = mean_bool(rows, "heuristic_success")
        nearest_rate = mean_bool(rows, "nearest_endpoint_success")
        final_spl = mean_float(rows, "final_spl")
        heuristic_spl = mean_float(rows, "heuristic_spl")
        bucket_rows = {
            bucket: [row for row in rows if row.get("final_failure_bucket") == bucket]
            for bucket in ("overshoot", "stop_too_early_proxy", "never_reached")
        }
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "split": split,
                "target_scope": target_scope,
                "heuristic": heuristic,
                "heuristic_params": params,
                "items": len(rows),
                "trace_available_rate": mean_bool(rows, "trace_available"),
                "changed_endpoint_rate": mean_bool(rows, "selected_changed"),
                "final_success_rate": final_rate,
                "heuristic_success_rate": heuristic_rate,
                "nearest_endpoint_success_rate": nearest_rate,
                "delta_success_rate": subtract_or_none(heuristic_rate, final_rate),
                "gap_capture_rate": gap_capture_rate(heuristic_rate, final_rate, nearest_rate),
                "recovery_rate": mean_bool(rows, "recovered_by_heuristic"),
                "harm_rate": mean_bool(rows, "harmed_by_heuristic"),
                "net_recovery_rate": subtract_or_none(
                    mean_bool(rows, "recovered_by_heuristic"),
                    mean_bool(rows, "harmed_by_heuristic"),
                ),
                "final_spl": final_spl,
                "heuristic_spl": heuristic_spl,
                "nearest_endpoint_spl": mean_float(rows, "nearest_endpoint_spl"),
                "delta_spl": subtract_or_none(heuristic_spl, final_spl),
                "overshoot_items": len(bucket_rows["overshoot"]),
                "overshoot_recovery_rate": mean_bool(bucket_rows["overshoot"], "recovered_by_heuristic"),
                "overshoot_harm_rate": mean_bool(bucket_rows["overshoot"], "harmed_by_heuristic"),
                "stop_too_early_proxy_items": len(bucket_rows["stop_too_early_proxy"]),
                "stop_too_early_proxy_recovery_rate": mean_bool(
                    bucket_rows["stop_too_early_proxy"],
                    "recovered_by_heuristic",
                ),
                "stop_too_early_proxy_harm_rate": mean_bool(
                    bucket_rows["stop_too_early_proxy"],
                    "harmed_by_heuristic",
                ),
                "never_reached_items": len(bucket_rows["never_reached"]),
                "never_reached_recovery_rate": mean_bool(bucket_rows["never_reached"], "recovered_by_heuristic"),
                "never_reached_harm_rate": mean_bool(bucket_rows["never_reached"], "harmed_by_heuristic"),
                "mean_selected_step_delta": mean_float(rows, "selected_step_delta"),
                "mean_final_stop_prob": mean_float(rows, "final_stop_prob"),
                "mean_selected_stop_prob": mean_float(rows, "selected_stop_prob"),
            }
        )
    return summary_rows


def summary_sort_key(
    item: tuple[tuple[str, str, str, str, str, str], list[dict[str, Any]]],
) -> tuple[int, str, str, int, str, str]:
    (_, dataset, split, target_scope, heuristic, params), _ = item
    dataset_order = {"R2R": 0, "REVERIE": 1, "SOON": 2, "CVDN": 3}
    scope_order = {"official": 0, "goal": 1, "region": 2, "region_threshold": 3}
    return (
        dataset_order.get(dataset, 99),
        dataset,
        split,
        scope_order.get(target_scope, 99),
        heuristic,
        params,
    )


def gap_capture_rate(
    heuristic_rate: float | None,
    final_rate: float | None,
    nearest_rate: float | None,
) -> float | None:
    if heuristic_rate is None or final_rate is None or nearest_rate is None:
        return None
    gap = nearest_rate - final_rate
    if gap <= EPS:
        return None
    return (heuristic_rate - final_rate) / gap


def build_markdown_report(
    experiment_id: str,
    item_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Endpoint Heuristic Reranking Report",
        "",
        f"- Experiment: `{experiment_id}`",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Item-scope-heuristic rows: `{len(item_rows)}`",
        "",
        "## Field Notes",
        "",
        "- Heuristics only use trajectory and decision_trace-derived features.",
        "- GT distances and success labels are used only after endpoint selection for scoring.",
        "- `gap_capture_rate`: `(heuristic SR - final SR) / (nearest endpoint SR - final SR)`.",
        "- `recovery_rate`: final failed but the heuristic endpoint succeeds.",
        "- `harm_rate`: final succeeded but the heuristic endpoint fails.",
        "",
        "## Official Scope",
        "",
    ]
    official_rows = [row for row in summary_rows if row["target_scope"] == "official"]
    lines.extend(markdown_table(top_summary_rows(official_rows)))
    lines.extend(["", "## All Target Scopes", ""])
    lines.extend(markdown_table(top_summary_rows(summary_rows)))
    lines.append("")
    return "\n".join(lines)


def top_summary_rows(rows: list[dict[str, Any]], limit_per_group: int = 12) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["split"], row["target_scope"])
        groups.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        final_rows = [
            row for row in group_rows if row["heuristic"] == "final" and row["heuristic_params"] == ""
        ]
        ranked = sorted(
            [row for row in group_rows if row not in final_rows],
            key=lambda row: (
                parse_float(row.get("delta_success_rate")) or -math.inf,
                parse_float(row.get("delta_spl")) or -math.inf,
                -(parse_float(row.get("harm_rate")) or math.inf),
            ),
            reverse=True,
        )
        selected.extend(final_rows[:1] + ranked[: max(limit_per_group - 1, 0)])
    return selected


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = [
        "dataset",
        "split",
        "scope",
        "heuristic",
        "params",
        "items",
        "final SR",
        "heur SR",
        "nearest SR",
        "delta SR",
        "capture",
        "recovery",
        "harm",
        "final SPL",
        "heur SPL",
        "changed",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["dataset"],
            row["split"],
            row["target_scope"],
            row["heuristic"],
            row["heuristic_params"],
            str(row["items"]),
            format_percent(row["final_success_rate"]),
            format_percent(row["heuristic_success_rate"]),
            format_percent(row["nearest_endpoint_success_rate"]),
            format_signed_percent(row["delta_success_rate"]),
            format_percent(row["gap_capture_rate"]),
            format_percent(row["recovery_rate"]),
            format_percent(row["harm_rate"]),
            format_percent(row["final_spl"]),
            format_percent(row["heuristic_spl"]),
            format_percent(row["changed_endpoint_rate"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = parse_float(chunk)
        if value is None:
            raise ValueError(f"Invalid float value: {chunk}")
        values.append(value)
    return values


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"Expected positive integer, got: {chunk}")
        values.append(value)
    return values


def as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def subtract_or_none(left: Any, right: Any) -> float | None:
    left_float = parse_float(left)
    right_float = parse_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def step_fraction(step: int | None, trajectory_length: int) -> float | None:
    if step is None or trajectory_length <= 1:
        return None
    return float(step) / float(trajectory_length - 1)


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1.0 for value in values if bool(value)) / len(values)


def mean_float(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [parse_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


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


def format_float(value: float) -> str:
    return format(value, ".12g")


def format_percent(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{parsed * 100.0:.2f}"


def format_signed_percent(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{parsed * 100.0:+.2f}"


def path_to_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
