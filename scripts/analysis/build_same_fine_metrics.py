#!/usr/bin/env python3
"""Build per-item SAME fine metrics from eval_items sidecars."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONNECTIVITY_DIR = REPO_ROOT / "data" / "same" / "simulator" / "connectivity"
SCHEMA_VERSION = "same_fine_metrics.v2"
EPS = 1e-9

COMMON_METRICS = [
    "action_step_count",
    "move_step_count",
    "instruction_token_count",
    "path_length_m",
    "path_edge_count",
]

ENDPOINT_METRICS = [
    "final_success",
    "oracle_success",
    "final_distance_to_goal_m",
    "final_distance_to_goal_edges",
    "path_length_ratio",
    "oracle_path_length_m",
    "oracle_path_edge_count",
    "oracle_path_length_ratio",
    "shortest_path_length_m",
    "shortest_path_edge_count",
]

OFFICIAL_METRICS = [
    "final_success",
    "oracle_success",
    "final_distance_to_goal_m",
    "oracle_distance_to_goal_m",
    "final_distance_to_goal_edges",
    "path_length_m",
    "path_length_ratio",
    "oracle_path_length_m",
    "oracle_path_edge_count",
    "oracle_path_length_ratio",
    "shortest_path_length_m",
    "shortest_path_edge_count",
    "spl",
    "oracle_plan_success",
    "oracle_plan_distance_m",
    "dist_to_end_reduction_m",
    "rgs",
    "rgspl",
    "det_success",
    "det_spl",
    "goal_progress_m",
    "heading_error",
    "elevation_error",
    "point_det_error",
]

GROUP_METRICS = {
    "common": COMMON_METRICS,
    "eval_end_goal": ENDPOINT_METRICS,
    "eval_end_region": ENDPOINT_METRICS,
    "eval_end_region_threshold": ENDPOINT_METRICS,
    "official": OFFICIAL_METRICS,
}

WIDE_FIELDNAMES = (
    ["experiment_id", "dataset", "split", "internal_item_id", "saved_instr_id"]
    + [
        f"{group_name}.{metric_name}"
        for group_name, metric_names in GROUP_METRICS.items()
        for metric_name in metric_names
    ]
)

LONG_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "internal_item_id",
    "metric_group",
    "metric_name",
    "value_num",
    "value_bool",
    "value_type",
]


@dataclass(frozen=True)
class EvalItemSource:
    dataset: str
    split: str
    context_path: Path
    items_path: Path
    success_threshold_m: float


@dataclass
class Graph:
    adjacency: dict[str, dict[str, float]]

    @classmethod
    def from_connectivity_file(cls, path: Path) -> "Graph":
        data = json.loads(path.read_text(encoding="utf-8"))
        adjacency: dict[str, dict[str, float]] = {}
        for i, item in enumerate(data):
            if not item.get("included"):
                continue
            source = item["image_id"]
            adjacency.setdefault(source, {})
            for j, connected in enumerate(item.get("unobstructed", [])):
                if not connected:
                    continue
                target_item = data[j]
                if not target_item.get("included"):
                    continue
                target = target_item["image_id"]
                weight = pose_distance_m(item, target_item)
                adjacency.setdefault(target, {})
                previous = adjacency[source].get(target)
                if previous is None or weight < previous:
                    adjacency[source][target] = weight
                    adjacency[target][source] = weight
        return cls(adjacency=adjacency)

    def shortest(self, source: str | None, target: str | None) -> tuple[float | None, int | None]:
        if source is None or target is None:
            return None, None
        if source == target:
            return 0.0, 0
        if source not in self.adjacency or target not in self.adjacency:
            return None, None

        heap: list[tuple[float, int, str]] = [(0.0, 0, source)]
        best: dict[str, tuple[float, int]] = {source: (0.0, 0)}
        while heap:
            distance, edge_count, node = heapq.heappop(heap)
            if not same_shortest_state((distance, edge_count), best[node]):
                continue
            if node == target:
                return distance, edge_count
            for neighbor, weight in self.adjacency[node].items():
                candidate = (distance + weight, edge_count + 1)
                previous = best.get(neighbor)
                if previous is None or is_better_shortest(candidate, previous):
                    best[neighbor] = candidate
                    heapq.heappush(heap, (candidate[0], candidate[1], neighbor))
        return None, None

    def nearest(self, source: str | None, targets: Iterable[str]) -> tuple[float | None, int | None, str | None]:
        best_distance: float | None = None
        best_edges: int | None = None
        best_target: str | None = None
        for target in sorted(set(targets)):
            distance, edge_count = self.shortest(source, target)
            if distance is None or edge_count is None:
                continue
            if (
                best_distance is None
                or distance < best_distance - EPS
                or (abs(distance - best_distance) <= EPS and edge_count < (best_edges or 0))
            ):
                best_distance = distance
                best_edges = edge_count
                best_target = target
        return best_distance, best_edges, best_target


def pose_distance_m(item_a: dict[str, Any], item_b: dict[str, Any]) -> float:
    pose_a = item_a["pose"]
    pose_b = item_b["pose"]
    return math.sqrt(
        (pose_a[3] - pose_b[3]) ** 2
        + (pose_a[7] - pose_b[7]) ** 2
        + (pose_a[11] - pose_b[11]) ** 2
    )


def same_shortest_state(left: tuple[float, int], right: tuple[float, int]) -> bool:
    return abs(left[0] - right[0]) <= EPS and left[1] == right[1]


def is_better_shortest(candidate: tuple[float, int], previous: tuple[float, int]) -> bool:
    return candidate[0] < previous[0] - EPS or (
        abs(candidate[0] - previous[0]) <= EPS and candidate[1] < previous[1]
    )


class GraphCache:
    def __init__(self, connectivity_dir: Path) -> None:
        self.connectivity_dir = connectivity_dir
        self._graphs: dict[str, Graph] = {}

    def get(self, scan: str) -> Graph:
        if scan not in self._graphs:
            path = self.connectivity_dir / f"{scan}_connectivity.json"
            if not path.exists():
                raise FileNotFoundError(f"Connectivity graph not found for scan {scan}: {path}")
            self._graphs[scan] = Graph.from_connectivity_file(path)
        return self._graphs[scan]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SAME fine metrics from eval_items sidecars.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment output directory.")
    parser.add_argument(
        "--connectivity-dir",
        default=str(DEFAULT_CONNECTIVITY_DIR),
        help="MatterSim connectivity directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <experiment-dir>/fine_metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "fine_metrics"
    build_fine_metrics(
        experiment_dir=experiment_dir,
        connectivity_dir=Path(args.connectivity_dir),
        output_dir=output_dir,
    )


def build_fine_metrics(experiment_dir: Path, connectivity_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    connectivity_dir = connectivity_dir.resolve()
    output_dir = (output_dir or experiment_dir / "fine_metrics").resolve()
    eval_items_dir = experiment_dir / "eval_items"
    sources = discover_eval_item_sources(eval_items_dir)
    if not sources:
        raise FileNotFoundError(f"No eval_items contexts found in {eval_items_dir}")

    jsonl_dir = output_dir / "jsonl"
    tables_dir = output_dir / "tables"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    graph_cache = GraphCache(connectivity_dir)
    all_rows: list[dict[str, Any]] = []
    output_jsonl_paths: list[Path] = []

    for source in sources:
        rows = [
            build_fine_metric_row(item, source, graph_cache)
            for item in read_jsonl(source.items_path)
        ]
        output_path = jsonl_dir / f"{source.dataset}_{source.split}_fine_metrics.jsonl"
        write_jsonl(output_path, rows)
        output_jsonl_paths.append(output_path)
        all_rows.extend(rows)

    wide_path = tables_dir / "fine_metrics_wide.csv"
    long_path = tables_dir / "fine_metrics_long.csv"
    summary_path = tables_dir / "fine_metrics_summary.json"
    manifest_path = output_dir / "manifest.json"

    write_wide_csv(wide_path, all_rows)
    write_long_csv(long_path, all_rows)
    summary = build_summary(all_rows)
    write_json(summary_path, summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_dir.name,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_eval_items_dir": path_to_string(eval_items_dir),
        "connectivity_dir": path_to_string(connectivity_dir),
        "output_dir": path_to_string(output_dir),
        "metric_groups": list(GROUP_METRICS),
        "files": {
            "jsonl": [path_to_string(path) for path in output_jsonl_paths],
            "wide_csv": path_to_string(wide_path),
            "long_csv": path_to_string(long_path),
            "summary_json": path_to_string(summary_path),
        },
        "counts": {
            "dataset_splits": len(sources),
            "items": len(all_rows),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def discover_eval_item_sources(eval_items_dir: Path) -> list[EvalItemSource]:
    sources: list[EvalItemSource] = []
    for context_path in sorted(eval_items_dir.glob("*_eval_context.json")):
        context = json.loads(context_path.read_text(encoding="utf-8"))
        run_context = context.get("run_context", {})
        dataset = run_context.get("dataset")
        split = run_context.get("split")
        items_name = context.get("files", {}).get("eval_items")
        if not dataset or not split or not items_name:
            raise ValueError(f"Invalid eval context: {context_path}")
        items_path = eval_items_dir / items_name
        if not items_path.exists():
            raise FileNotFoundError(f"Eval items file not found: {items_path}")
        sources.append(
            EvalItemSource(
                dataset=str(dataset),
                split=str(split),
                context_path=context_path,
                items_path=items_path,
                success_threshold_m=float(run_context.get("success_threshold_m", 3.0)),
            )
        )
    return sources


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def build_fine_metric_row(
    item: dict[str, Any],
    source: EvalItemSource,
    graph_cache: GraphCache,
) -> dict[str, Any]:
    annotation = item.get("annotation", {})
    scan = annotation.get("scan")
    if not scan:
        raise ValueError(f"Missing annotation.scan for {item.get('identity', {})}")
    graph = graph_cache.get(scan)
    common = build_common_metrics(item)
    eval_end_goal = build_goal_metrics(item, source, graph, common)
    eval_end_region = build_region_metrics(item, source, graph, common)
    eval_end_region_threshold = build_region_threshold_metrics(item, source, graph, common)
    official = build_official_metrics(item, source, graph, common)
    identity = item.get("identity", {})

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": source.items_path.parents[1].name,
        "dataset": source.dataset,
        "split": source.split,
        "identity": {
            "internal_item_id": identity.get("internal_item_id"),
            "saved_instr_id": identity.get("saved_instr_id"),
            "source_ids": identity.get("source_ids", {}),
        },
        "common": common,
        "eval_end_goal": eval_end_goal,
        "eval_end_region": eval_end_region,
        "eval_end_region_threshold": eval_end_region_threshold,
        "official": official,
    }


def build_common_metrics(item: dict[str, Any]) -> dict[str, Any]:
    prediction = item.get("prediction", {})
    trajectory = prediction.get("trajectory") or []
    pred_path_segments = prediction.get("pred_path_segments") or []
    primitives = item.get("primitives", {})
    official = item.get("official_item_scores", {})
    raw_same = official.get("raw_same", {})
    canonical = official.get("canonical", {})

    action_step_count = raw_same.get("action_steps")
    if action_step_count is None:
        action_step_count = max(len(pred_path_segments) - 1, 0)

    path_length_m = canonical.get("actual_length_m")
    if path_length_m is None:
        cumulative = primitives.get("trajectory_cumulative_lengths_m") or []
        if cumulative:
            path_length_m = cumulative[-1]
    if path_length_m is None:
        path_length_m = safe_sum(primitives.get("trajectory_edge_lengths_m") or [])

    return {
        "action_step_count": int(action_step_count) if action_step_count is not None else None,
        "move_step_count": count_moves(trajectory),
        "instruction_token_count": item.get("annotation", {})
        .get("instruction_meta", {})
        .get("encoding_len"),
        "path_length_m": float(path_length_m) if path_length_m is not None else None,
        "path_edge_count": count_moves(trajectory),
    }


def build_goal_metrics(
    item: dict[str, Any],
    source: EvalItemSource,
    graph: Graph,
    common: dict[str, Any],
) -> dict[str, Any]:
    annotation = item.get("annotation", {})
    primitives = item.get("primitives", {})
    trajectory = item.get("prediction", {}).get("trajectory") or []
    cumulative_lengths = primitives.get("trajectory_cumulative_lengths_m") or []
    distances_by_step = primitives.get("distance_to_nav_goal_by_step_m") or []
    start_viewpoint = annotation.get("start_viewpoint")
    final_viewpoint = trajectory[-1] if trajectory else primitives.get("final_viewpoint")
    goal_viewpoint = (annotation.get("gt_path") or [None])[-1] if source.dataset == "SOON" else annotation.get("nav_goal_viewpoint")

    final_distance_m = last_or_none(distances_by_step)
    if final_distance_m is None:
        final_distance_m, _ = graph.shortest(final_viewpoint, goal_viewpoint)

    final_distance_to_goal_edges = graph.shortest(final_viewpoint, goal_viewpoint)[1]
    shortest_path_length_m = item.get("official_item_scores", {}).get("canonical", {}).get("shortest_path_length_m")
    if shortest_path_length_m is None:
        shortest_path_length_m = primitives.get("shortest_start_to_nav_goal_distance_m")
    shortest_graph_distance, shortest_path_edge_count = graph.shortest(start_viewpoint, goal_viewpoint)
    if shortest_path_length_m is None:
        shortest_path_length_m = shortest_graph_distance

    oracle_index = first_index_below_threshold(distances_by_step, source.success_threshold_m)
    oracle_path_length_m = cumulative_at(cumulative_lengths, oracle_index)
    oracle_path_edge_count = move_count_until(trajectory, oracle_index)

    return {
        "final_success": bool(final_distance_m is not None and final_distance_m < source.success_threshold_m),
        "oracle_success": oracle_index is not None,
        "final_distance_to_goal_m": final_distance_m,
        "final_distance_to_goal_edges": final_distance_to_goal_edges,
        "path_length_ratio": ratio(common.get("path_length_m"), shortest_path_length_m),
        "oracle_path_length_m": oracle_path_length_m,
        "oracle_path_edge_count": oracle_path_edge_count,
        "oracle_path_length_ratio": ratio(oracle_path_length_m, shortest_path_length_m),
        "shortest_path_length_m": shortest_path_length_m,
        "shortest_path_edge_count": shortest_path_edge_count,
    }


def build_region_metrics(
    item: dict[str, Any],
    source: EvalItemSource,
    graph: Graph,
    common: dict[str, Any],
) -> dict[str, Any] | None:
    if source.dataset == "R2R":
        return None

    annotation = item.get("annotation", {})
    primitives = item.get("primitives", {})
    trajectory = item.get("prediction", {}).get("trajectory") or []
    cumulative_lengths = primitives.get("trajectory_cumulative_lengths_m") or []
    targets = region_targets(item, source.dataset)
    if not targets:
        return endpoint_null_metrics()

    start_viewpoint = annotation.get("start_viewpoint")
    final_viewpoint = trajectory[-1] if trajectory else primitives.get("final_viewpoint")
    final_distance_m, final_edges, _ = graph.nearest(final_viewpoint, targets)
    shortest_path_length_m, shortest_path_edge_count, _ = graph.nearest(start_viewpoint, targets)

    target_set = set(targets)
    oracle_index = first_index_in_targets(trajectory, target_set)
    oracle_path_length_m = cumulative_at(cumulative_lengths, oracle_index)
    oracle_path_edge_count = move_count_until(trajectory, oracle_index)

    return {
        "final_success": final_viewpoint in target_set,
        "oracle_success": oracle_index is not None,
        "final_distance_to_goal_m": final_distance_m,
        "final_distance_to_goal_edges": final_edges,
        "path_length_ratio": ratio(common.get("path_length_m"), shortest_path_length_m),
        "oracle_path_length_m": oracle_path_length_m,
        "oracle_path_edge_count": oracle_path_edge_count,
        "oracle_path_length_ratio": ratio(oracle_path_length_m, shortest_path_length_m),
        "shortest_path_length_m": shortest_path_length_m,
        "shortest_path_edge_count": shortest_path_edge_count,
    }


def build_region_threshold_metrics(
    item: dict[str, Any],
    source: EvalItemSource,
    graph: Graph,
    common: dict[str, Any],
) -> dict[str, Any] | None:
    if source.dataset == "R2R":
        return None

    annotation = item.get("annotation", {})
    primitives = item.get("primitives", {})
    trajectory = item.get("prediction", {}).get("trajectory") or []
    cumulative_lengths = primitives.get("trajectory_cumulative_lengths_m") or []
    targets = region_targets(item, source.dataset)
    if not targets:
        return endpoint_null_metrics()

    start_viewpoint = annotation.get("start_viewpoint")
    final_viewpoint = trajectory[-1] if trajectory else primitives.get("final_viewpoint")
    graph_final_distance_m, final_edges, _ = graph.nearest(final_viewpoint, targets)
    shortest_path_length_m, shortest_path_edge_count, _ = graph.nearest(start_viewpoint, targets)

    distances_by_step = primitives.get(region_distance_primitive_key(source.dataset)) or []
    final_distance_m = last_or_none(distances_by_step)
    if final_distance_m is None:
        final_distance_m = graph_final_distance_m
    if not distances_by_step:
        distances_by_step = [graph.nearest(viewpoint, targets)[0] for viewpoint in trajectory]
    oracle_index = first_index_below_threshold(distances_by_step, source.success_threshold_m)
    oracle_path_length_m = cumulative_at(cumulative_lengths, oracle_index)
    oracle_path_edge_count = move_count_until(trajectory, oracle_index)

    return {
        "final_success": bool(final_distance_m is not None and final_distance_m < source.success_threshold_m),
        "oracle_success": oracle_index is not None,
        "final_distance_to_goal_m": final_distance_m,
        "final_distance_to_goal_edges": final_edges,
        "path_length_ratio": ratio(common.get("path_length_m"), shortest_path_length_m),
        "oracle_path_length_m": oracle_path_length_m,
        "oracle_path_edge_count": oracle_path_edge_count,
        "oracle_path_length_ratio": ratio(oracle_path_length_m, shortest_path_length_m),
        "shortest_path_length_m": shortest_path_length_m,
        "shortest_path_edge_count": shortest_path_edge_count,
    }


def build_official_metrics(
    item: dict[str, Any],
    source: EvalItemSource,
    graph: Graph,
    common: dict[str, Any],
) -> dict[str, Any]:
    annotation = item.get("annotation", {})
    primitives = item.get("primitives", {})
    prediction = item.get("prediction", {})
    trajectory = prediction.get("trajectory") or []
    cumulative_lengths = primitives.get("trajectory_cumulative_lengths_m") or []
    official_scores = item.get("official_item_scores", {})
    raw_same = official_scores.get("raw_same", {})
    canonical = official_scores.get("canonical", {})

    final_viewpoint = trajectory[-1] if trajectory else primitives.get("final_viewpoint")
    start_viewpoint = annotation.get("start_viewpoint")
    distance_goal = official_distance_goal_viewpoint(source.dataset, annotation)
    final_distance_edges = graph.shortest(final_viewpoint, distance_goal)[1]
    shortest_distance_m = canonical.get("shortest_path_length_m")
    shortest_graph_distance, shortest_path_edge_count = graph.shortest(start_viewpoint, distance_goal)
    if shortest_distance_m is None:
        shortest_distance_m = shortest_graph_distance

    oracle_index = official_oracle_index(item, source)
    oracle_path_length_m = cumulative_at(cumulative_lengths, oracle_index)
    oracle_path_edge_count = move_count_until(trajectory, oracle_index)
    path_length_m = canonical.get("actual_length_m")
    if path_length_m is None:
        path_length_m = raw_same.get("trajectory_lengths")
    if path_length_m is None:
        path_length_m = common.get("path_length_m")

    oracle_plan_distance_m = raw_same.get("oracle_plan_errors")

    return {
        "final_success": bool_or_none(canonical.get("final_success", raw_same.get("success"))),
        "oracle_success": bool_or_none(canonical.get("oracle_success", raw_same.get("oracle_success"))),
        "final_distance_to_goal_m": canonical.get("final_distance_m"),
        "oracle_distance_to_goal_m": canonical.get("min_distance_along_trajectory_m"),
        "final_distance_to_goal_edges": final_distance_edges,
        "path_length_m": path_length_m,
        "path_length_ratio": ratio(path_length_m, shortest_distance_m),
        "oracle_path_length_m": oracle_path_length_m,
        "oracle_path_edge_count": oracle_path_edge_count,
        "oracle_path_length_ratio": ratio(oracle_path_length_m, shortest_distance_m),
        "shortest_path_length_m": shortest_distance_m,
        "shortest_path_edge_count": shortest_path_edge_count,
        "spl": raw_same.get("spl"),
        "oracle_plan_success": (
            None
            if oracle_plan_distance_m is None
            else bool(oracle_plan_distance_m < source.success_threshold_m)
        ),
        "oracle_plan_distance_m": oracle_plan_distance_m,
        "dist_to_end_reduction_m": raw_same.get("dist_to_end_reductions"),
        "rgs": raw_same.get("rgs"),
        "rgspl": raw_same.get("rgspl"),
        "det_success": bool_or_none(raw_same.get("det_success")),
        "det_spl": raw_same.get("det_spl"),
        "goal_progress_m": raw_same.get("goal_progress"),
        "heading_error": raw_same.get("heading_error"),
        "elevation_error": raw_same.get("elevation_error"),
        "point_det_error": raw_same.get("point_det_error"),
    }


def region_targets(item: dict[str, Any], dataset: str) -> list[str]:
    extras = item.get("dataset_extras", {})
    if dataset == "CVDN":
        return list(extras.get("cvdn", {}).get("end_panos") or [])
    if dataset == "REVERIE":
        return list(item.get("annotation", {}).get("success_target_viewpoints") or [])
    if dataset == "SOON":
        return list(extras.get("soon", {}).get("bbox_viewpoints") or [])
    return []


def region_distance_primitive_key(dataset: str) -> str:
    if dataset == "CVDN":
        return "distance_to_nearest_end_pano_by_step_m"
    if dataset == "REVERIE":
        return "distance_to_nearest_success_target_by_step_m"
    if dataset == "SOON":
        return "distance_to_nearest_bbox_viewpoint_by_step_m"
    return ""


def official_distance_goal_viewpoint(dataset: str, annotation: dict[str, Any]) -> str | None:
    if dataset == "SOON":
        return (annotation.get("gt_path") or [None])[-1]
    return annotation.get("nav_goal_viewpoint")


def official_oracle_index(item: dict[str, Any], source: EvalItemSource) -> int | None:
    trajectory = item.get("prediction", {}).get("trajectory") or []
    primitives = item.get("primitives", {})
    if source.dataset == "REVERIE":
        return first_index_in_targets(trajectory, set(region_targets(item, source.dataset)))
    distances_by_step = primitives.get("distance_to_nav_goal_by_step_m") or []
    return first_index_below_threshold(distances_by_step, source.success_threshold_m)


def endpoint_null_metrics() -> dict[str, Any]:
    return {name: None for name in ENDPOINT_METRICS}


def count_moves(trajectory: list[str]) -> int:
    return sum(1 for previous, current in zip(trajectory, trajectory[1:]) if previous != current)


def move_count_until(trajectory: list[str], index: int | None) -> int | None:
    if index is None:
        return None
    return count_moves(trajectory[: index + 1])


def first_index_below_threshold(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value is not None and value < threshold:
            return index
    return None


def first_index_in_targets(trajectory: list[str], targets: set[str]) -> int | None:
    for index, viewpoint in enumerate(trajectory):
        if viewpoint in targets:
            return index
    return None


def cumulative_at(cumulative_lengths: list[float], index: int | None) -> float | None:
    if index is None or index >= len(cumulative_lengths):
        return None
    return float(cumulative_lengths[index])


def last_or_none(values: list[Any]) -> Any:
    return values[-1] if values else None


def safe_sum(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))


def ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator = float(denominator)
    if abs(denominator) <= EPS:
        return None
    return float(numerator) / denominator


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_wide_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WIDE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_wide_row(row))


def flatten_wide_row(row: dict[str, Any]) -> dict[str, str]:
    identity = row.get("identity", {})
    output = {
        "experiment_id": csv_value(row.get("experiment_id")),
        "dataset": csv_value(row.get("dataset")),
        "split": csv_value(row.get("split")),
        "internal_item_id": csv_value(identity.get("internal_item_id")),
        "saved_instr_id": csv_value(identity.get("saved_instr_id")),
    }
    for group_name, metric_names in GROUP_METRICS.items():
        group = row.get(group_name) or {}
        for metric_name in metric_names:
            output[f"{group_name}.{metric_name}"] = csv_value(group.get(metric_name))
    return output


def write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            for long_row in iter_long_rows(row):
                writer.writerow(long_row)


def iter_long_rows(row: dict[str, Any]) -> Iterable[dict[str, str]]:
    identity = row.get("identity", {})
    base = {
        "experiment_id": csv_value(row.get("experiment_id")),
        "dataset": csv_value(row.get("dataset")),
        "split": csv_value(row.get("split")),
        "internal_item_id": csv_value(identity.get("internal_item_id")),
    }
    for group_name, metric_names in GROUP_METRICS.items():
        group = row.get(group_name)
        if group is None:
            continue
        for metric_name in metric_names:
            value = group.get(metric_name)
            value_num, value_bool, value_type = split_long_value(value)
            yield {
                **base,
                "metric_group": group_name,
                "metric_name": metric_name,
                "value_num": value_num,
                "value_bool": value_bool,
                "value_type": value_type,
            }


def split_long_value(value: Any) -> tuple[str, str, str]:
    if value is None:
        return "", "", "null"
    if isinstance(value, bool):
        return "", "true" if value else "false", "bool"
    if isinstance(value, (int, float)):
        return csv_value(value), "", "num"
    return "", csv_value(value), "str"


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


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_splits: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['dataset']}/{row['split']}"
        summary = dataset_splits.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "items": 0,
                "goal_final_successes": 0,
                "goal_oracle_successes": 0,
                "region_items": 0,
                "region_final_successes": 0,
                "region_oracle_successes": 0,
                "region_threshold_items": 0,
                "region_threshold_final_successes": 0,
                "region_threshold_oracle_successes": 0,
                "official_final_successes": 0,
                "official_oracle_successes": 0,
                "official_oracle_plan_successes": 0,
            },
        )
        summary["items"] += 1
        goal = row.get("eval_end_goal") or {}
        summary["goal_final_successes"] += int(bool(goal.get("final_success")))
        summary["goal_oracle_successes"] += int(bool(goal.get("oracle_success")))
        region = row.get("eval_end_region")
        if region is not None:
            summary["region_items"] += 1
            summary["region_final_successes"] += int(bool(region.get("final_success")))
            summary["region_oracle_successes"] += int(bool(region.get("oracle_success")))
        region_threshold = row.get("eval_end_region_threshold")
        if region_threshold is not None:
            summary["region_threshold_items"] += 1
            summary["region_threshold_final_successes"] += int(bool(region_threshold.get("final_success")))
            summary["region_threshold_oracle_successes"] += int(bool(region_threshold.get("oracle_success")))
        official = row.get("official") or {}
        summary["official_final_successes"] += int(bool(official.get("final_success")))
        summary["official_oracle_successes"] += int(bool(official.get("oracle_success")))
        summary["official_oracle_plan_successes"] += int(bool(official.get("oracle_plan_success")))
    return {
        "schema_version": SCHEMA_VERSION,
        "items": len(rows),
        "dataset_splits": dataset_splits,
    }


def path_to_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
