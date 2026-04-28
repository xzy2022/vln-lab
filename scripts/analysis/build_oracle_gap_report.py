#!/usr/bin/env python3
"""Build oracle-gap and endpoint upper-bound reports from SAME eval outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_NAME = "oracle_gap_for_rl_research"
SCHEMA_VERSION = "same_oracle_gap_report.v1"
EPS = 1e-9

DEFAULT_SCOPES = ("official", "goal", "region", "region_threshold")

ITEM_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "target_scope",
    "internal_item_id",
    "saved_instr_id",
    "success_mode",
    "distance_key",
    "success_threshold_m",
    "trajectory_step_count",
    "final_step",
    "first_success_step",
    "best_distance_step",
    "best_distance_step_last",
    "final_is_best_distance",
    "final_success",
    "oracle_success",
    "nearest_endpoint_success",
    "recovered_by_first_success",
    "recovered_by_nearest_endpoint",
    "overshoot",
    "stop_too_early_proxy",
    "never_reached",
    "final_distance_m",
    "best_distance_m",
    "final_minus_best_distance_m",
    "final_path_length_m",
    "first_success_path_length_m",
    "best_endpoint_path_length_m",
    "shortest_path_length_m",
    "final_spl",
    "first_success_oracle_spl",
    "nearest_endpoint_spl",
]

SUMMARY_FIELDNAMES = [
    "experiment_id",
    "dataset",
    "split",
    "target_scope",
    "items",
    "final_success_rate",
    "oracle_success_rate",
    "oracle_gap_rate",
    "nearest_endpoint_success_rate",
    "nearest_recovery_rate",
    "overshoot_rate",
    "stop_too_early_proxy_rate",
    "never_reached_rate",
    "final_spl",
    "first_success_oracle_spl",
    "nearest_endpoint_spl",
    "mean_final_distance_m",
    "mean_best_distance_m",
    "mean_final_minus_best_distance_m",
    "mean_first_success_step",
    "mean_best_distance_step",
]


@dataclass(frozen=True)
class EvalItemSource:
    dataset: str
    split: str
    context_path: Path
    items_path: Path
    success_threshold_m: float


@dataclass(frozen=True)
class TargetScope:
    name: str
    metric_group: str
    success_mode: str
    distance_key: str
    targets: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oracle-gap and no-training endpoint upper-bound reports from SAME eval_items.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    fine_metrics_dir = Path(args.fine_metrics_dir) if args.fine_metrics_dir else experiment_dir / "fine_metrics"
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / DEFAULT_OUTPUT_NAME
    scopes = tuple(scope.strip() for scope in args.scopes.split(",") if scope.strip())
    build_oracle_gap_report(
        experiment_dir=experiment_dir,
        fine_metrics_dir=fine_metrics_dir,
        output_dir=output_dir,
        scopes=scopes,
    )


def build_oracle_gap_report(
    experiment_dir: Path,
    fine_metrics_dir: Path,
    output_dir: Path,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    fine_metrics_dir = fine_metrics_dir.resolve()
    output_dir = output_dir.resolve()
    eval_items_dir = experiment_dir / "eval_items"

    sources = discover_eval_item_sources(eval_items_dir)
    if not sources:
        raise FileNotFoundError(f"No eval_items contexts found in {eval_items_dir}")

    fine_rows = load_fine_metrics_wide(fine_metrics_dir / "tables" / "fine_metrics_wide.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    item_rows: list[dict[str, Any]] = []
    for source in sources:
        for item in read_jsonl(source.items_path):
            identity = item.get("identity", {})
            key = (
                source.dataset,
                source.split,
                str(identity.get("internal_item_id")),
            )
            fine_row = fine_rows.get(key)
            if fine_row is None:
                raise KeyError(f"Missing fine_metrics row for {key}")
            for scope_name in scopes:
                target_scope = build_target_scope(item, source.dataset, scope_name)
                if target_scope is None:
                    continue
                item_rows.append(build_item_row(item, source, fine_row, target_scope))

    summary_rows = build_summary_rows(item_rows)

    item_csv_path = output_dir / "oracle_gap_items.csv"
    summary_csv_path = output_dir / "oracle_gap_summary.csv"
    report_path = output_dir / "oracle_gap_report.md"
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
        "files": {
            "items_csv": path_to_string(item_csv_path),
            "summary_csv": path_to_string(summary_csv_path),
            "report_md": path_to_string(report_path),
        },
        "counts": {
            "eval_item_sources": len(sources),
            "item_scope_rows": len(item_rows),
            "summary_rows": len(summary_rows),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def load_fine_metrics_wide(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"fine_metrics wide CSV not found: {path}")
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["dataset"], row["split"], row["internal_item_id"])
            rows[key] = row
    return rows


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


def build_target_scope(item: dict[str, Any], dataset: str, scope_name: str) -> TargetScope | None:
    if scope_name == "official":
        if dataset == "REVERIE":
            return TargetScope(
                name="official",
                metric_group="official",
                success_mode="exact_viewpoint",
                distance_key=region_distance_key(dataset),
                targets=tuple(region_targets(item, dataset)),
            )
        return TargetScope(
            name="official",
            metric_group="official",
            success_mode="distance_threshold",
            distance_key="distance_to_nav_goal_by_step_m",
        )

    if scope_name == "goal":
        return TargetScope(
            name="goal",
            metric_group="eval_end_goal",
            success_mode="distance_threshold",
            distance_key="distance_to_nav_goal_by_step_m",
        )

    if scope_name == "region":
        if dataset == "R2R":
            return None
        return TargetScope(
            name="region",
            metric_group="eval_end_region",
            success_mode="exact_viewpoint",
            distance_key=region_distance_key(dataset),
            targets=tuple(region_targets(item, dataset)),
        )

    if scope_name == "region_threshold":
        if dataset == "R2R":
            return None
        return TargetScope(
            name="region_threshold",
            metric_group="eval_end_region_threshold",
            success_mode="distance_threshold",
            distance_key=region_distance_key(dataset),
            targets=tuple(region_targets(item, dataset)),
        )

    raise ValueError(f"Unknown target scope: {scope_name}")


def build_item_row(
    item: dict[str, Any],
    source: EvalItemSource,
    fine_row: dict[str, str],
    target_scope: TargetScope,
) -> dict[str, Any]:
    identity = item.get("identity", {})
    trajectory = item.get("prediction", {}).get("trajectory") or []
    primitives = item.get("primitives", {})
    cumulative_lengths = as_float_list(primitives.get("trajectory_cumulative_lengths_m") or [])
    distances = distance_series(item, target_scope)

    final_step = len(trajectory) - 1 if trajectory else None
    first_success_step = first_success_index(
        trajectory=trajectory,
        distances=distances,
        threshold=source.success_threshold_m,
        target_scope=target_scope,
    )
    best_distance_step = first_best_distance_index(distances)
    best_distance_step_last = last_best_distance_index(distances)

    final_success = success_at_step(trajectory, distances, final_step, source.success_threshold_m, target_scope)
    oracle_success = first_success_step is not None
    nearest_endpoint_success = success_at_step(
        trajectory,
        distances,
        best_distance_step,
        source.success_threshold_m,
        target_scope,
    )

    final_distance_m = value_at(distances, final_step)
    best_distance_m = value_at(distances, best_distance_step)
    final_is_best_distance = (
        final_distance_m is not None
        and best_distance_m is not None
        and final_distance_m <= best_distance_m + EPS
    )

    shortest_path_length_m = parse_float(fine_row.get(f"{target_scope.metric_group}.shortest_path_length_m"))
    final_path_length_m = value_at(cumulative_lengths, final_step)
    first_success_path_length_m = value_at(cumulative_lengths, first_success_step)
    best_endpoint_path_length_m = value_at(cumulative_lengths, best_distance_step)

    overshoot = bool(
        final_success is False
        and oracle_success
        and first_success_step is not None
        and final_step is not None
        and first_success_step < final_step
    )
    stop_too_early_proxy = bool(final_success is False and not oracle_success and final_is_best_distance)
    never_reached = bool(final_success is False and not oracle_success and not stop_too_early_proxy)

    final_spl = parse_float(fine_row.get(f"{target_scope.metric_group}.spl"))
    if final_spl is None:
        final_spl = spl_at(final_success, final_path_length_m, shortest_path_length_m)

    return {
        "experiment_id": source.items_path.parents[1].name,
        "dataset": source.dataset,
        "split": source.split,
        "target_scope": target_scope.name,
        "internal_item_id": identity.get("internal_item_id"),
        "saved_instr_id": identity.get("saved_instr_id"),
        "success_mode": target_scope.success_mode,
        "distance_key": target_scope.distance_key,
        "success_threshold_m": source.success_threshold_m,
        "trajectory_step_count": len(trajectory),
        "final_step": final_step,
        "first_success_step": first_success_step,
        "best_distance_step": best_distance_step,
        "best_distance_step_last": best_distance_step_last,
        "final_is_best_distance": final_is_best_distance,
        "final_success": final_success,
        "oracle_success": oracle_success,
        "nearest_endpoint_success": nearest_endpoint_success,
        "recovered_by_first_success": bool(final_success is False and oracle_success),
        "recovered_by_nearest_endpoint": bool(final_success is False and nearest_endpoint_success),
        "overshoot": overshoot,
        "stop_too_early_proxy": stop_too_early_proxy,
        "never_reached": never_reached,
        "final_distance_m": final_distance_m,
        "best_distance_m": best_distance_m,
        "final_minus_best_distance_m": subtract_or_none(final_distance_m, best_distance_m),
        "final_path_length_m": final_path_length_m,
        "first_success_path_length_m": first_success_path_length_m,
        "best_endpoint_path_length_m": best_endpoint_path_length_m,
        "shortest_path_length_m": shortest_path_length_m,
        "final_spl": final_spl,
        "first_success_oracle_spl": spl_at(oracle_success, first_success_path_length_m, shortest_path_length_m),
        "nearest_endpoint_spl": spl_at(
            nearest_endpoint_success,
            best_endpoint_path_length_m,
            shortest_path_length_m,
        ),
    }


def distance_series(item: dict[str, Any], target_scope: TargetScope) -> list[float | None]:
    primitives = item.get("primitives", {})
    trajectory = item.get("prediction", {}).get("trajectory") or []
    raw_distances = primitives.get(target_scope.distance_key) or []
    distances = as_float_list(raw_distances)
    if distances:
        return distances
    if target_scope.success_mode == "exact_viewpoint":
        target_set = set(target_scope.targets)
        return [0.0 if viewpoint in target_set else 1.0 for viewpoint in trajectory]
    return []


def first_success_index(
    trajectory: list[str],
    distances: list[float | None],
    threshold: float,
    target_scope: TargetScope,
) -> int | None:
    for index in range(len(trajectory)):
        if success_at_step(trajectory, distances, index, threshold, target_scope):
            return index
    return None


def success_at_step(
    trajectory: list[str],
    distances: list[float | None],
    step: int | None,
    threshold: float,
    target_scope: TargetScope,
) -> bool | None:
    if step is None or step < 0 or step >= len(trajectory):
        return None
    if target_scope.success_mode == "exact_viewpoint":
        return trajectory[step] in set(target_scope.targets)
    distance = value_at(distances, step)
    if distance is None:
        return None
    return distance < threshold


def first_best_distance_index(distances: list[float | None]) -> int | None:
    best_index: int | None = None
    best_value: float | None = None
    for index, distance in enumerate(distances):
        if distance is None:
            continue
        if best_value is None or distance < best_value - EPS:
            best_value = distance
            best_index = index
    return best_index


def last_best_distance_index(distances: list[float | None]) -> int | None:
    best_index = first_best_distance_index(distances)
    if best_index is None:
        return None
    best_value = distances[best_index]
    if best_value is None:
        return None
    last_index = best_index
    for index, distance in enumerate(distances):
        if distance is not None and abs(distance - best_value) <= EPS:
            last_index = index
    return last_index


def region_targets(item: dict[str, Any], dataset: str) -> list[str]:
    if dataset == "CVDN":
        return list(item.get("dataset_extras", {}).get("cvdn", {}).get("end_panos") or [])
    if dataset == "REVERIE":
        return list(item.get("annotation", {}).get("success_target_viewpoints") or [])
    if dataset == "SOON":
        return list(item.get("dataset_extras", {}).get("soon", {}).get("bbox_viewpoints") or [])
    return []


def region_distance_key(dataset: str) -> str:
    if dataset == "CVDN":
        return "distance_to_nearest_end_pano_by_step_m"
    if dataset == "REVERIE":
        return "distance_to_nearest_success_target_by_step_m"
    if dataset == "SOON":
        return "distance_to_nearest_bbox_viewpoint_by_step_m"
    return ""


def build_summary_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in item_rows:
        key = (row["experiment_id"], row["dataset"], row["split"], row["target_scope"])
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (experiment_id, dataset, split, target_scope), rows in sorted(groups.items(), key=summary_sort_key):
        final_rate = mean_bool(rows, "final_success")
        oracle_rate = mean_bool(rows, "oracle_success")
        nearest_rate = mean_bool(rows, "nearest_endpoint_success")
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "split": split,
                "target_scope": target_scope,
                "items": len(rows),
                "final_success_rate": final_rate,
                "oracle_success_rate": oracle_rate,
                "oracle_gap_rate": subtract_or_none(oracle_rate, final_rate),
                "nearest_endpoint_success_rate": nearest_rate,
                "nearest_recovery_rate": mean_bool(rows, "recovered_by_nearest_endpoint"),
                "overshoot_rate": mean_bool(rows, "overshoot"),
                "stop_too_early_proxy_rate": mean_bool(rows, "stop_too_early_proxy"),
                "never_reached_rate": mean_bool(rows, "never_reached"),
                "final_spl": mean_float(rows, "final_spl"),
                "first_success_oracle_spl": mean_float(rows, "first_success_oracle_spl"),
                "nearest_endpoint_spl": mean_float(rows, "nearest_endpoint_spl"),
                "mean_final_distance_m": mean_float(rows, "final_distance_m"),
                "mean_best_distance_m": mean_float(rows, "best_distance_m"),
                "mean_final_minus_best_distance_m": mean_float(rows, "final_minus_best_distance_m"),
                "mean_first_success_step": mean_float(rows, "first_success_step"),
                "mean_best_distance_step": mean_float(rows, "best_distance_step"),
            }
        )
    return summary_rows


def summary_sort_key(item: tuple[tuple[str, str, str, str], list[dict[str, Any]]]) -> tuple[int, str, str, int]:
    (_, dataset, split, target_scope), _ = item
    dataset_order = {"R2R": 0, "REVERIE": 1, "SOON": 2, "CVDN": 3}
    scope_order = {"official": 0, "goal": 1, "region": 2, "region_threshold": 3}
    return (dataset_order.get(dataset, 99), dataset, split, scope_order.get(target_scope, 99))


def build_markdown_report(
    experiment_id: str,
    item_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Oracle Gap and Endpoint Upper-Bound Report",
        "",
        f"- Experiment: `{experiment_id}`",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Item-scope rows: `{len(item_rows)}`",
        "",
        "## Field Notes",
        "",
        "- `first_success_step`: first step that satisfies the target scope success rule.",
        "- `best_distance_step`: earliest visited step with the minimum target distance.",
        "- `overshoot`: final failed, but the trajectory had already entered a successful state.",
        "- `stop_too_early_proxy`: final failed, never succeeded, and the final step is also the closest visited step.",
        "- `nearest_endpoint_spl`: no-training upper bound from selecting the closest visited endpoint.",
        "",
        "## Official Scope",
        "",
    ]
    official_rows = [row for row in summary_rows if row["target_scope"] == "official"]
    lines.extend(markdown_table(official_rows))
    lines.extend(["", "## All Target Scopes", ""])
    lines.extend(markdown_table(summary_rows))
    lines.append("")
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = [
        "dataset",
        "split",
        "scope",
        "items",
        "final SR",
        "oracle SR",
        "gap",
        "nearest SR",
        "final SPL",
        "oracle-stop SPL",
        "nearest SPL",
        "overshoot",
        "stop-early proxy",
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
            str(row["items"]),
            format_percent(row["final_success_rate"]),
            format_percent(row["oracle_success_rate"]),
            format_percent(row["oracle_gap_rate"]),
            format_percent(row["nearest_endpoint_success_rate"]),
            format_percent(row["final_spl"]),
            format_percent(row["first_success_oracle_spl"]),
            format_percent(row["nearest_endpoint_spl"]),
            format_percent(row["overshoot_rate"]),
            format_percent(row["stop_too_early_proxy_rate"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def as_float_list(values: list[Any]) -> list[float | None]:
    return [parse_float(value) for value in values]


def value_at(values: list[Any], index: int | None) -> Any:
    if index is None or index < 0 or index >= len(values):
        return None
    return values[index]


def spl_at(success: bool | None, path_length_m: float | None, shortest_path_length_m: float | None) -> float:
    if not success:
        return 0.0
    if path_length_m is None or shortest_path_length_m is None:
        return 0.0
    if abs(shortest_path_length_m) <= EPS:
        return 1.0 if abs(path_length_m) <= EPS else 0.0
    return float(shortest_path_length_m) / max(float(shortest_path_length_m), float(path_length_m))


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


def subtract_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


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


def format_percent(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{parsed * 100.0:.2f}"


def path_to_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
