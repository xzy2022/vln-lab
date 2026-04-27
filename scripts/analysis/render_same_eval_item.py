#!/usr/bin/env python3
"""Render one SAME eval item trajectory by internal item id."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "experiment_outputs"
DEFAULT_DATASET_PATH = REPO_ROOT / "third_party" / "Matterport3DSimulator" / "data" / "v1" / "scans"
DEFAULT_NAV_GRAPH_PATH = REPO_ROOT / "third_party" / "Matterport3DSimulator" / "connectivity"
EPS = 1e-9


@dataclass(frozen=True)
class EvalItemMatch:
    item: dict[str, Any]
    eval_items_path: Path
    dataset: str
    split: str


@dataclass(frozen=True)
class TopologyNode:
    viewpoint: str
    x: float
    y: float
    z: float


@dataclass
class TopologyGraph:
    nodes: dict[str, TopologyNode]
    adjacency: dict[str, dict[str, float]]
    connectivity_path: Path

    @classmethod
    def from_connectivity_file(cls, path: Path) -> "TopologyGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes: dict[str, TopologyNode] = {}
        adjacency: dict[str, dict[str, float]] = {}

        for item in data:
            if not item.get("included"):
                continue
            pose = item["pose"]
            viewpoint = item["image_id"]
            nodes[viewpoint] = TopologyNode(
                viewpoint=viewpoint,
                x=float(pose[3]),
                y=float(pose[7]),
                z=float(pose[11]),
            )
            adjacency.setdefault(viewpoint, {})

        for i, item in enumerate(data):
            if not item.get("included"):
                continue
            source = item["image_id"]
            for j, connected in enumerate(item.get("unobstructed", [])):
                if not connected:
                    continue
                target_item = data[j]
                if not target_item.get("included"):
                    continue
                target = target_item["image_id"]
                if source == target:
                    continue
                weight = topology_pose_distance_m(item, target_item)
                previous = adjacency[source].get(target)
                if previous is None or weight < previous:
                    adjacency[source][target] = weight
                    adjacency[target][source] = weight

        return cls(nodes=nodes, adjacency=adjacency, connectivity_path=path)

    @property
    def edges(self) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        edges: list[tuple[str, str]] = []
        for source, neighbors in self.adjacency.items():
            for target in neighbors:
                edge = tuple(sorted((source, target)))
                if edge in seen:
                    continue
                seen.add(edge)
                edges.append((edge[0], edge[1]))
        return edges

    def shortest_path(self, source: str | None, target: str | None) -> tuple[list[str], float | None, int | None]:
        if source is None or target is None:
            return [], None, None
        if source not in self.nodes or target not in self.nodes:
            return [], None, None
        if source == target:
            return [source], 0.0, 0

        heap: list[tuple[float, int, str]] = [(0.0, 0, source)]
        best: dict[str, tuple[float, int]] = {source: (0.0, 0)}
        previous: dict[str, str] = {}

        while heap:
            distance, edge_count, node = heapq.heappop(heap)
            if not same_shortest_state((distance, edge_count), best[node]):
                continue
            if node == target:
                return (
                    reconstruct_path(previous, source, target),
                    distance,
                    edge_count,
                )
            for neighbor in sorted(self.adjacency.get(node, {})):
                weight = self.adjacency[node][neighbor]
                candidate = (distance + weight, edge_count + 1)
                current = best.get(neighbor)
                if current is None or is_better_shortest(candidate, current):
                    best[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(heap, (candidate[0], candidate[1], neighbor))

        return [], None, None


@dataclass(frozen=True)
class TopologyTargetSpec:
    target_type: str
    targets: list[str]
    fallback_used: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find one SAME eval_items sample by identity.internal_item_id or "
            "identity.saved_instr_id, then render its viewpoint-level trajectory."
        )
    )
    parser.add_argument(
        "item_id",
        help="identity.internal_item_id, for example r2r_5593_0",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment output directory. Defaults to the newest directory under experiment_outputs/.",
    )
    parser.add_argument(
        "--eval-items",
        type=Path,
        default=None,
        help="Optional direct path to one *_eval_items.jsonl file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter, for example R2R, REVERIE, CVDN, or SOON.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter, for example val_seen or val_unseen.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Matterport3D scans root, the directory that contains <scan_id>/.",
    )
    parser.add_argument(
        "--nav-graph-path",
        type=Path,
        default=DEFAULT_NAV_GRAPH_PATH,
        help="MatterSim connectivity directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <experiment-dir>/item_replays/<item_id>/.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--vfov-deg", type=float, default=60.0)
    parser.add_argument(
        "--show-topk",
        type=int,
        default=None,
        help="Only label the top-K visible candidates by decision probability. Defaults to all.",
    )
    parser.add_argument(
        "--annotation-font-size",
        type=int,
        default=28,
        help="Base font size for image overlays.",
    )
    parser.add_argument(
        "--header-font-size",
        type=int,
        default=None,
        help="Font size for the decision header. Defaults to --annotation-font-size.",
    )
    parser.add_argument(
        "--candidate-font-size",
        type=int,
        default=None,
        help="Font size for candidate probability labels. Defaults to --annotation-font-size.",
    )
    parser.add_argument(
        "--route-font-size",
        type=int,
        default=None,
        help="Font size for route annotations. Defaults to --annotation-font-size.",
    )
    parser.add_argument(
        "--download-missing-scan",
        action="store_true",
        help=(
            "If the scan directory is missing, download matterport_skybox_images.zip "
            "into the dataset root before preparing skybox_small files."
        ),
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=None,
        help=(
            "Base directory passed to scripts/setup/download_mp.py -o. "
            "Defaults to the parent of v1/ for --dataset-path."
        ),
    )
    parser.add_argument(
        "--skip-skybox-prepare",
        action="store_true",
        help="Skip automatic scripts/setup/prepare_mp3d_skybox.py for the target scan.",
    )
    parser.add_argument(
        "--skip-topology-map",
        action="store_true",
        help="Skip topology_full.png and topology_focus.png generation.",
    )
    parser.add_argument(
        "--topology-map-size",
        type=int,
        default=1200,
        help="Square PNG size in pixels for topology maps.",
    )
    parser.add_argument(
        "--topology-focus-hops",
        type=int,
        default=1,
        help="Neighbor hops around sample-related nodes for topology_focus.png.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = resolve_experiment_dir(args.experiment_dir)
    match = find_eval_item(
        item_id=args.item_id,
        experiment_dir=experiment_dir,
        eval_items_path=args.eval_items,
        dataset_filter=args.dataset,
        split_filter=args.split,
    )
    item = match.item
    annotation = item.get("annotation", {})
    prediction = item.get("prediction", {})
    identity = item.get("identity", {})
    scan = annotation.get("scan")
    trajectory = prediction.get("trajectory") or []
    if not scan:
        raise SystemExit(f"Matched item has no annotation.scan: {identity}")
    if not trajectory:
        raise SystemExit(f"Matched item has no prediction.trajectory: {identity}")

    dataset_path = args.dataset_path.expanduser().resolve()
    nav_graph_path = args.nav_graph_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else experiment_dir / "item_replays" / args.item_id
    )

    ensure_scan_ready(
        scan=scan,
        dataset_path=dataset_path,
        download_missing=args.download_missing_scan,
        download_root=args.download_root,
        skip_prepare=args.skip_skybox_prepare,
    )
    render_trajectory(
        item=item,
        eval_items_path=match.eval_items_path,
        item_id=args.item_id,
        dataset=match.dataset,
        dataset_path=dataset_path,
        nav_graph_path=nav_graph_path,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        vfov_deg=args.vfov_deg,
        show_topk=args.show_topk,
        annotation_font_size=args.annotation_font_size,
        header_font_size=args.header_font_size,
        candidate_font_size=args.candidate_font_size,
        route_font_size=args.route_font_size,
        skip_topology_map=args.skip_topology_map,
        topology_map_size=args.topology_map_size,
        topology_focus_hops=args.topology_focus_hops,
    )

    summary = {
        "item_id": args.item_id,
        "internal_item_id": identity.get("internal_item_id"),
        "saved_instr_id": identity.get("saved_instr_id"),
        "dataset": match.dataset,
        "split": match.split,
        "scan": scan,
        "trajectory_len": len(trajectory),
        "eval_items": str(match.eval_items_path),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def resolve_experiment_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        experiment_dir = explicit.expanduser().resolve()
        if not experiment_dir.is_dir():
            raise SystemExit(f"Experiment directory does not exist: {experiment_dir}")
        return experiment_dir

    if not DEFAULT_EXPERIMENTS_ROOT.is_dir():
        raise SystemExit(
            "Missing experiment_outputs/. Pass --experiment-dir or --eval-items explicitly."
        )
    candidates = [
        path for path in DEFAULT_EXPERIMENTS_ROOT.iterdir()
        if path.is_dir() and (path / "eval_items").is_dir()
    ]
    if not candidates:
        raise SystemExit("No experiment directory with eval_items/ found under experiment_outputs/.")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def find_eval_item(
    *,
    item_id: str,
    experiment_dir: Path,
    eval_items_path: Path | None,
    dataset_filter: str | None,
    split_filter: str | None,
) -> EvalItemMatch:
    matches: list[EvalItemMatch] = []
    for path in iter_eval_items_paths(experiment_dir, eval_items_path):
        dataset, split = dataset_split_from_eval_items_path(path)
        if dataset_filter and dataset.lower() != dataset_filter.lower():
            continue
        if split_filter and split != split_filter:
            continue
        for item in read_jsonl(path):
            identity = item.get("identity", {})
            if item_id in {
                str(identity.get("internal_item_id")),
                str(identity.get("saved_instr_id")),
            }:
                matches.append(EvalItemMatch(item=item, eval_items_path=path, dataset=dataset, split=split))

    if not matches:
        searched = eval_items_path or (experiment_dir / "eval_items")
        raise SystemExit(f"Item not found: {item_id}. Searched: {searched}")
    if len(matches) > 1:
        candidates = "\n".join(
            f"- {match.dataset} {match.split}: {match.eval_items_path}"
            for match in matches
        )
        raise SystemExit(
            f"Item id is ambiguous: {item_id}. Add --dataset/--split or --eval-items.\n{candidates}"
        )
    return matches[0]


def iter_eval_items_paths(experiment_dir: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Eval items file does not exist: {path}")
        return [path]
    paths = sorted((experiment_dir / "eval_items").glob("*_eval_items.jsonl"))
    if not paths:
        raise SystemExit(f"No *_eval_items.jsonl files found in {experiment_dir / 'eval_items'}")
    return paths


def dataset_split_from_eval_items_path(path: Path) -> tuple[str, str]:
    suffix = "_eval_items.jsonl"
    name = path.name
    if not name.endswith(suffix) or "_" not in name[: -len(suffix)]:
        return "UNKNOWN", "unknown"
    dataset, split = name[: -len(suffix)].split("_", 1)
    return dataset, split


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return items


def ensure_scan_ready(
    *,
    scan: str,
    dataset_path: Path,
    download_missing: bool,
    download_root: Path | None,
    skip_prepare: bool,
) -> None:
    scan_dir = dataset_path / scan
    skybox_dir = scan_dir / "matterport_skybox_images"
    archive_path = scan_dir / "matterport_skybox_images.zip"
    has_small_skybox = skybox_dir.is_dir() and any(skybox_dir.glob("*_skybox_small.jpg"))
    has_prepare_input = skybox_dir.is_dir() or archive_path.is_file()
    if not scan_dir.is_dir() or (download_missing and not has_small_skybox and not has_prepare_input):
        if not download_missing:
            raise SystemExit(
                f"Missing scan directory: {scan_dir}\n"
                "Add --download-missing-scan to download matterport_skybox_images for this scan."
            )
        root = resolve_download_root(dataset_path, download_root)
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "setup" / "download_mp.py"),
                "-o",
                str(root),
                "--id",
                scan,
                "--type",
                "matterport_skybox_images",
            ]
        )

    if skip_prepare:
        return
    skybox_dir = scan_dir / "matterport_skybox_images"
    has_small_skybox = skybox_dir.is_dir() and any(skybox_dir.glob("*_skybox_small.jpg"))
    if has_small_skybox:
        return
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "setup" / "prepare_mp3d_skybox.py"),
            "--scans-dir",
            str(dataset_path),
            "--scan-id",
            scan,
        ]
    )


def resolve_download_root(dataset_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if dataset_path.name == "scans" and dataset_path.parent.name == "v1":
        return dataset_path.parent.parent
    return dataset_path


def render_trajectory(
    *,
    item: dict[str, Any],
    eval_items_path: Path,
    item_id: str,
    dataset: str,
    dataset_path: Path,
    nav_graph_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    vfov_deg: float,
    show_topk: int | None,
    annotation_font_size: int,
    header_font_size: int | None,
    candidate_font_size: int | None,
    route_font_size: int | None,
    skip_topology_map: bool,
    topology_map_size: int,
    topology_focus_hops: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = item.get("annotation", {})
    prediction = item.get("prediction", {})
    identity = item.get("identity", {})
    scan = annotation["scan"]
    trajectory = prediction["trajectory"]
    decision_trace = prediction.get("decision_trace") or {}
    trace_steps = decision_trace.get("steps") or []

    sim = init_sim(
        dataset_path=dataset_path,
        nav_graph_path=nav_graph_path,
        width=width,
        height=height,
        vfov_deg=vfov_deg,
    )
    try:
        frames = build_replay_frames(trajectory, trace_steps)
        for frame_idx, frame in enumerate(frames):
            viewpoint = frame["viewpoint"]
            sheet = build_contact_sheet(
                sim=sim,
                scan=scan,
                viewpoint=viewpoint,
                width=width,
                height=height,
            )
            if frame.get("kind") == "decision":
                annotate_decision_sheet(
                    sheet=sheet,
                    frame=frame,
                    width=width,
                    height=height,
                    show_topk=show_topk,
                    header_font_size=header_font_size or annotation_font_size,
                    candidate_font_size=candidate_font_size or annotation_font_size,
                )
            elif frame.get("kind") == "route":
                annotate_route_sheet(
                    sheet=sheet,
                    frame=frame,
                    route_font_size=route_font_size or annotation_font_size,
                )
            sheet.save(output_dir / f"step_{frame_idx:03d}_{viewpoint}.jpg", quality=90)

        meta = {
            "item_id": item_id,
            "internal_item_id": identity.get("internal_item_id"),
            "saved_instr_id": identity.get("saved_instr_id"),
            "eval_items": str(eval_items_path),
            "scan": scan,
            "instruction": annotation.get("instruction"),
            "start_viewpoint": annotation.get("start_viewpoint"),
            "nav_goal_viewpoint": annotation.get("nav_goal_viewpoint"),
            "trajectory": trajectory,
            "decision_trace_used": bool(trace_steps),
            "frames": [
                {key: value for key, value in frame.items() if key != "decision"}
                for frame in frames
            ],
            "route_annotations": [
                frame for frame in frames if frame.get("kind") == "route"
            ],
            "overlay": {
                "annotation_font_size": annotation_font_size,
                "header_font_size": header_font_size or annotation_font_size,
                "candidate_font_size": candidate_font_size or annotation_font_size,
                "route_font_size": route_font_size or annotation_font_size,
            },
        }
        if skip_topology_map:
            meta["topology"] = {
                "enabled": False,
                "reason": "skipped_by_user",
            }
        else:
            meta["topology"] = render_topology_maps(
                item=item,
                dataset=dataset,
                nav_graph_path=nav_graph_path,
                output_dir=output_dir,
                map_size=topology_map_size,
                focus_hops=topology_focus_hops,
            )
        if trace_steps:
            meta["note"] = (
                "Frames are built from prediction.decision_trace. Decision frames show "
                "model-visible candidates and route frames show graph-path motion caused by "
                "one earlier global decision."
            )
        else:
            meta["note"] = (
                "This is a viewpoint-level replay: each step saves the 36 standard views "
                "for that viewpoint. It is not an exact SAME action-level replay because "
                "this eval_items file does not store per-step decision_trace."
            )
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        sim.close()


def render_topology_maps(
    *,
    item: dict[str, Any],
    dataset: str,
    nav_graph_path: Path,
    output_dir: Path,
    map_size: int,
    focus_hops: int,
) -> dict[str, Any]:
    annotation = item.get("annotation", {})
    prediction = item.get("prediction", {})
    scan = annotation["scan"]
    connectivity_path = nav_graph_path / f"{scan}_connectivity.json"
    graph = TopologyGraph.from_connectivity_file(connectivity_path)

    prediction_path = normalize_viewpoint_list(prediction.get("trajectory") or [])
    label_path = normalize_viewpoint_list(annotation.get("gt_path") or [])
    start_viewpoint = str(annotation.get("start_viewpoint") or prediction_path[0])
    final_viewpoint = prediction_path[-1] if prediction_path else None
    label_goal = label_path[-1] if label_path else None
    target_spec = topology_target_spec(item, dataset)
    optimal_path, optimal_distance, optimal_edges, optimal_target, path_errors = nearest_topology_path(
        graph=graph,
        source=start_viewpoint,
        targets=target_spec.targets,
    )

    map_size = max(int(map_size), 320)
    focus_hops = max(int(focus_hops), 0)
    full_map_path = output_dir / "topology_full.png"
    focus_map_path = output_dir / "topology_focus.png"
    missing_nodes = collect_missing_topology_nodes(
        graph=graph,
        paths={
            "prediction_path": prediction_path,
            "label_path": label_path,
            "optimal_path": optimal_path,
            "target_set": target_spec.targets,
            "start": [start_viewpoint],
            "final": [final_viewpoint] if final_viewpoint else [],
            "label_goal": [label_goal] if label_goal else [],
            "optimal_target": [optimal_target] if optimal_target else [],
        },
    )

    render_topology_map_image(
        graph=graph,
        output_path=full_map_path,
        title=f"{scan} full topology",
        map_size=map_size,
        visible_nodes=set(graph.nodes),
        prediction_path=prediction_path,
        label_path=label_path,
        optimal_path=optimal_path,
        start_viewpoint=start_viewpoint,
        final_viewpoint=final_viewpoint,
        label_goal=label_goal,
        optimal_target=optimal_target,
        targets=target_spec.targets,
    )
    focus_nodes = topology_focus_nodes(
        graph=graph,
        paths=[prediction_path, label_path, optimal_path, target_spec.targets],
        hops=focus_hops,
    )
    render_topology_map_image(
        graph=graph,
        output_path=focus_map_path,
        title=f"{scan} sample focus",
        map_size=map_size,
        visible_nodes=focus_nodes,
        prediction_path=prediction_path,
        label_path=label_path,
        optimal_path=optimal_path,
        start_viewpoint=start_viewpoint,
        final_viewpoint=final_viewpoint,
        label_goal=label_goal,
        optimal_target=optimal_target,
        targets=target_spec.targets,
    )

    return {
        "enabled": True,
        "full_map": str(full_map_path),
        "focus_map": str(focus_map_path),
        "connectivity_file": str(connectivity_path),
        "target_type": target_spec.target_type,
        "target_count": len(target_spec.targets),
        "target_fallback_used": target_spec.fallback_used,
        "optimal_target": optimal_target,
        "label_path": label_path,
        "optimal_path": optimal_path,
        "prediction_path": prediction_path,
        "optimal_distance_m": optimal_distance,
        "optimal_edge_count": optimal_edges,
        "missing_nodes": missing_nodes,
        "path_errors": path_errors,
    }


def topology_target_spec(item: dict[str, Any], dataset: str) -> TopologyTargetSpec:
    annotation = item.get("annotation", {})
    extras = item.get("dataset_extras", {})
    dataset_name = dataset.upper()
    label_path = normalize_viewpoint_list(annotation.get("gt_path") or [])

    if dataset_name == "CVDN":
        return TopologyTargetSpec(
            target_type="cvdn_end_panos",
            targets=unique_viewpoints(extras.get("cvdn", {}).get("end_panos") or []),
        )
    if dataset_name == "REVERIE":
        return TopologyTargetSpec(
            target_type="reverie_success_target_viewpoints",
            targets=unique_viewpoints(annotation.get("success_target_viewpoints") or []),
        )
    if dataset_name == "SOON":
        targets = unique_viewpoints(extras.get("soon", {}).get("bbox_viewpoints") or [])
        if targets:
            return TopologyTargetSpec(target_type="soon_bbox_viewpoints", targets=targets)
        return TopologyTargetSpec(
            target_type="soon_gt_path_goal_fallback",
            targets=unique_viewpoints(label_path[-1:]),
            fallback_used=True,
        )

    nav_goal = annotation.get("nav_goal_viewpoint")
    if nav_goal:
        return TopologyTargetSpec(
            target_type="r2r_nav_goal_viewpoint",
            targets=unique_viewpoints([nav_goal]),
        )
    return TopologyTargetSpec(
        target_type="r2r_gt_path_goal_fallback",
        targets=unique_viewpoints(label_path[-1:]),
        fallback_used=True,
    )


def nearest_topology_path(
    *,
    graph: TopologyGraph,
    source: str | None,
    targets: list[str],
) -> tuple[list[str], float | None, int | None, str | None, list[dict[str, Any]]]:
    best_path: list[str] = []
    best_distance: float | None = None
    best_edges: int | None = None
    best_target: str | None = None
    errors: list[dict[str, Any]] = []

    for target in sorted(set(targets)):
        path, distance, edge_count = graph.shortest_path(source, target)
        if not path or distance is None or edge_count is None:
            reason = "no_path_from_start"
            if source not in graph.nodes:
                reason = "missing_start_node"
            elif target not in graph.nodes:
                reason = "missing_target_node"
            errors.append({"target": target, "reason": reason})
            continue
        if (
            best_distance is None
            or distance < best_distance - EPS
            or (
                abs(distance - best_distance) <= EPS
                and (best_edges is None or edge_count < best_edges)
            )
            or (
                best_distance is not None
                and abs(distance - best_distance) <= EPS
                and edge_count == best_edges
                and (best_target is None or target < best_target)
            )
        ):
            best_path = path
            best_distance = distance
            best_edges = edge_count
            best_target = target

    return best_path, best_distance, best_edges, best_target, errors


def render_topology_map_image(
    *,
    graph: TopologyGraph,
    output_path: Path,
    title: str,
    map_size: int,
    visible_nodes: set[str],
    prediction_path: list[str],
    label_path: list[str],
    optimal_path: list[str],
    start_viewpoint: str | None,
    final_viewpoint: str | None,
    label_goal: str | None,
    optimal_target: str | None,
    targets: list[str],
) -> None:
    from PIL import Image, ImageDraw  # pylint: disable=import-outside-toplevel

    visible_nodes = {node for node in visible_nodes if node in graph.nodes}
    if not visible_nodes:
        visible_nodes = set(graph.nodes)

    image = Image.new("RGB", (map_size, map_size), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    font = load_overlay_font(max(12, map_size // 55))
    small_font = load_overlay_font(max(10, map_size // 80))
    bounds = topology_bounds(graph, visible_nodes)
    padding = max(54, map_size // 14)
    _, legend_height = topology_legend_dimensions(map_size)
    plot_box = (
        padding,
        min(map_size - padding * 2, legend_height + padding),
        map_size - padding,
        map_size - padding,
    )
    z_values = [node.z for node in graph.nodes.values()]
    min_z = min(z_values, default=0.0)
    max_z = max(z_values, default=0.0)

    def point(viewpoint: str) -> tuple[float, float]:
        node = graph.nodes[viewpoint]
        return project_topology_node(node, bounds, plot_box)

    edge_width = max(1, map_size // 520)
    node_radius = max(4, map_size // 145)
    for source, target in graph.edges:
        if source not in visible_nodes or target not in visible_nodes:
            continue
        draw.line((point(source), point(target)), fill=(196, 196, 190), width=edge_width)

    for viewpoint in sorted(visible_nodes):
        node = graph.nodes[viewpoint]
        x, y = point(viewpoint)
        fill = elevation_color(node.z, min_z, max_z)
        draw.ellipse(
            (x - node_radius, y - node_radius, x + node_radius, y + node_radius),
            fill=fill,
            outline=(98, 98, 92),
            width=max(1, map_size // 700),
        )

    path_width = max(4, map_size // 150)
    draw_topology_path(
        draw,
        graph,
        prediction_path,
        point,
        visible_nodes,
        fill=(39, 118, 219),
        width=path_width,
    )
    draw_topology_path(
        draw,
        graph,
        label_path,
        point,
        visible_nodes,
        fill=(39, 151, 92),
        width=max(3, path_width - 2),
    )
    draw_topology_path(
        draw,
        graph,
        optimal_path,
        point,
        visible_nodes,
        fill=(137, 76, 200),
        width=max(3, path_width - 3),
        dashed=True,
    )
    draw_prediction_step_labels(
        draw,
        graph,
        prediction_path,
        point,
        visible_nodes,
        font=small_font,
        map_size=map_size,
    )

    marker_radius = max(10, map_size // 58)
    for target in targets:
        if target in visible_nodes:
            draw_topology_marker(
                draw,
                point(target),
                kind="target",
                label="",
                color=(137, 76, 200),
                radius=max(8, marker_radius - 4),
                font=small_font,
            )
    if label_goal in visible_nodes:
        draw_topology_marker(
            draw,
            point(label_goal),
            kind="square",
            label="G",
            color=(39, 151, 92),
            radius=marker_radius,
            font=font,
        )
    if optimal_target in visible_nodes:
        draw_topology_marker(
            draw,
            point(optimal_target),
            kind="diamond",
            label="O",
            color=(137, 76, 200),
            radius=marker_radius,
            font=font,
        )
    if final_viewpoint in visible_nodes:
        draw_topology_marker(
            draw,
            point(final_viewpoint),
            kind="x",
            label="F",
            color=(208, 64, 64),
            radius=marker_radius,
            font=font,
        )
    if start_viewpoint in visible_nodes:
        draw_topology_marker(
            draw,
            point(start_viewpoint),
            kind="circle",
            label="S",
            color=(32, 32, 32),
            radius=marker_radius,
            font=font,
        )

    draw_topology_legend(
        draw,
        title,
        font,
        small_font,
        map_size,
        min_z=min_z,
        max_z=max_z,
    )
    image.save(output_path)


def draw_topology_path(
    draw: Any,
    graph: TopologyGraph,
    path: list[str],
    point: Any,
    visible_nodes: set[str],
    *,
    fill: tuple[int, int, int],
    width: int,
    dashed: bool = False,
) -> None:
    for source, target in zip(path, path[1:]):
        if source not in graph.nodes or target not in graph.nodes:
            continue
        if source not in visible_nodes or target not in visible_nodes:
            continue
        if source == target:
            continue
        start = point(source)
        end = point(target)
        if dashed:
            draw_dashed_line(
                draw,
                start,
                end,
                fill=fill,
                width=width,
                dash=max(12, width * 4),
                gap=max(8, width * 2),
            )
        else:
            draw.line((start, end), fill=fill, width=width)


def draw_topology_marker(
    draw: Any,
    center: tuple[float, float],
    *,
    kind: str,
    label: str,
    color: tuple[int, int, int],
    radius: int,
    font: Any,
) -> None:
    x, y = center
    if kind == "target":
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=max(2, radius // 4),
        )
        return
    if kind == "square":
        draw.rectangle(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=max(2, radius // 5),
        )
    elif kind == "diamond":
        draw.polygon(
            [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)],
            fill=color,
            outline=(255, 255, 255),
        )
    elif kind == "x":
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255),
            outline=color,
            width=max(2, radius // 4),
        )
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=max(2, radius // 4))
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=max(2, radius // 4))
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255),
            outline=color,
            width=max(2, radius // 4),
        )

    if label:
        bbox = text_bbox(draw, (0, 0), label, font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        fill = (255, 255, 255) if kind in {"square", "diamond"} else color
        draw.text((x - text_width / 2, y - text_height / 2), label, fill=fill, font=font)


def draw_prediction_step_labels(
    draw: Any,
    graph: TopologyGraph,
    path: list[str],
    point: Any,
    visible_nodes: set[str],
    *,
    font: Any,
    map_size: int,
) -> None:
    if not path:
        return

    radius = max(8, map_size // 80)
    leader = max(10, map_size // 70)
    offsets = [
        (0, -leader * 2),
        (leader * 2, -leader),
        (leader * 2, leader),
        (0, leader * 2),
        (-leader * 2, leader),
        (-leader * 2, -leader),
    ]
    for index, viewpoint in enumerate(path):
        if viewpoint not in graph.nodes or viewpoint not in visible_nodes:
            continue
        x, y = point(viewpoint)
        dx, dy = offsets[index % len(offsets)]
        cx = x + dx
        cy = y + dy
        draw.line((x, y, cx, cy), fill=(39, 118, 219), width=max(1, map_size // 700))
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(39, 118, 219),
            outline=(255, 255, 255),
            width=max(2, radius // 5),
        )
        text = str(index)
        bbox = text_bbox(draw, (0, 0), text, font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (cx - text_width / 2, cy - text_height / 2),
            text,
            fill=(255, 255, 255),
            font=font,
        )


def draw_topology_legend(
    draw: Any,
    title: str,
    font: Any,
    small_font: Any,
    map_size: int,
    *,
    min_z: float,
    max_z: float,
) -> None:
    rows = [
        ("prediction", (39, 118, 219), "solid"),
        ("label path", (39, 151, 92), "solid"),
        ("optimal path", (137, 76, 200), "dashed"),
        ("start viewpoint (S)", (32, 32, 32), "start"),
        ("final prediction (F)", (208, 64, 64), "final"),
        ("label goal (G)", (39, 151, 92), "label_goal"),
        ("optimal target (O)", (137, 76, 200), "optimal_target"),
        ("target set", (137, 76, 200), "marker"),
        ("prediction step index", (39, 118, 219), "step_index"),
    ]
    padding = max(8, map_size // 100)
    line_height = max(18, map_size // 45)
    width, height = topology_legend_dimensions(map_size)
    left = padding
    top = padding
    draw.rectangle(
        (left, top, left + width, top + height),
        fill=(255, 255, 255),
        outline=(220, 220, 216),
    )
    draw.text((left + padding, top + padding), title, fill=(32, 32, 32), font=font)
    y = top + padding + line_height
    draw.text((left + padding, y), "top-down axes: horizontal=pose[3] x, vertical=pose[7] y", fill=(90, 90, 84), font=small_font)
    y += line_height
    draw.text((left + padding, y), "elevation/floor level is node color: pose[11] z", fill=(90, 90, 84), font=small_font)
    y += line_height
    gradient_left = left + padding
    gradient_right = gradient_left + max(120, map_size // 9)
    gradient_top = y + max(3, line_height // 5)
    gradient_bottom = gradient_top + max(8, line_height // 2)
    for index, x in enumerate(range(gradient_left, gradient_right)):
        ratio_value = index / max(gradient_right - gradient_left - 1, 1)
        fill = elevation_color(
            min_z + (max_z - min_z) * ratio_value,
            min_z,
            max_z,
        )
        draw.line((x, gradient_top, x, gradient_bottom), fill=fill)
    draw.rectangle((gradient_left, gradient_top, gradient_right, gradient_bottom), outline=(120, 120, 116))
    z_text = f"node elevation pose[11]: {min_z:.1f}m to {max_z:.1f}m"
    draw.text((gradient_right + padding, y), z_text, fill=(50, 50, 46), font=small_font)
    y += line_height
    for label, color, style in rows:
        sample_left = left + padding
        sample_right = sample_left + max(34, map_size // 28)
        sample_y = y + line_height // 2
        if style == "dashed":
            draw_dashed_line(
                draw,
                (sample_left, sample_y),
                (sample_right, sample_y),
                fill=color,
                width=max(3, map_size // 260),
                dash=10,
                gap=6,
            )
        elif style == "marker":
            draw.ellipse(
                (sample_left, sample_y - 5, sample_left + 10, sample_y + 5),
                outline=color,
                width=2,
            )
        elif style == "start":
            draw.ellipse(
                (sample_left, sample_y - 7, sample_left + 14, sample_y + 7),
                outline=color,
                width=2,
            )
            draw.text((sample_left + 4, sample_y - 8), "S", fill=color, font=small_font)
        elif style == "final":
            draw.ellipse(
                (sample_left, sample_y - 7, sample_left + 14, sample_y + 7),
                outline=color,
                width=2,
            )
            draw.line((sample_left, sample_y - 7, sample_left + 14, sample_y + 7), fill=color, width=2)
            draw.line((sample_left, sample_y + 7, sample_left + 14, sample_y - 7), fill=color, width=2)
        elif style == "label_goal":
            draw.rectangle(
                (sample_left, sample_y - 7, sample_left + 14, sample_y + 7),
                fill=color,
                outline=(255, 255, 255),
            )
        elif style == "optimal_target":
            draw.polygon(
                [
                    (sample_left + 7, sample_y - 8),
                    (sample_left + 15, sample_y),
                    (sample_left + 7, sample_y + 8),
                    (sample_left - 1, sample_y),
                ],
                fill=color,
                outline=(255, 255, 255),
            )
        elif style == "step_index":
            draw.ellipse(
                (sample_left, sample_y - 8, sample_left + 16, sample_y + 8),
                fill=color,
                outline=(255, 255, 255),
            )
            draw.text((sample_left + 4, sample_y - 8), "0", fill=(255, 255, 255), font=small_font)
        else:
            draw.line(
                (sample_left, sample_y, sample_right, sample_y),
                fill=color,
                width=max(3, map_size // 260),
            )
        draw.text((sample_right + padding, y), label, fill=(50, 50, 46), font=small_font)
        y += line_height
    draw.text((left + padding, y), "overlapping markers mean the same viewpoint has multiple roles", fill=(90, 90, 84), font=small_font)


def topology_legend_dimensions(map_size: int) -> tuple[int, int]:
    padding = max(8, map_size // 100)
    line_height = max(18, map_size // 45)
    row_count = 9
    width = max(390, map_size // 3)
    height = padding * 5 + line_height * (row_count + 5)
    return width, height


def topology_focus_nodes(graph: TopologyGraph, paths: list[list[str]], hops: int) -> set[str]:
    selected = {
        viewpoint
        for path in paths
        for viewpoint in path
        if viewpoint in graph.nodes
    }
    if not selected:
        return set(graph.nodes)

    frontier = set(selected)
    for _ in range(hops):
        next_frontier: set[str] = set()
        for viewpoint in frontier:
            next_frontier.update(graph.adjacency.get(viewpoint, {}))
        next_frontier -= selected
        selected.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return selected


def topology_bounds(graph: TopologyGraph, visible_nodes: set[str]) -> tuple[float, float, float, float]:
    xs = [graph.nodes[viewpoint].x for viewpoint in visible_nodes]
    ys = [graph.nodes[viewpoint].y for viewpoint in visible_nodes]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    margin = max(x_span, y_span) * 0.08
    return (min_x - margin, max_x + margin, min_y - margin, max_y + margin)


def project_topology_node(
    node: TopologyNode,
    bounds: tuple[float, float, float, float],
    plot_box: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    x_span = max(max_x - min_x, EPS)
    y_span = max(max_y - min_y, EPS)
    left, top, right, bottom = plot_box
    usable_width = max(right - left, 1.0)
    usable_height = max(bottom - top, 1.0)
    scale = min(usable_width / x_span, usable_height / y_span)
    draw_width = x_span * scale
    draw_height = y_span * scale
    offset_x = left + (usable_width - draw_width) / 2.0
    offset_y = top + (usable_height - draw_height) / 2.0
    return (
        offset_x + (node.x - min_x) * scale,
        offset_y + (max_y - node.y) * scale,
    )


def elevation_color(z_value: float, min_z: float, max_z: float) -> tuple[int, int, int]:
    if max_z - min_z <= EPS:
        return (226, 226, 220)
    ratio_value = (z_value - min_z) / (max_z - min_z)
    low = (242, 190, 75)
    high = (45, 137, 188)
    return tuple(
        int(low[index] + (high[index] - low[index]) * ratio_value)
        for index in range(3)
    )


def collect_missing_topology_nodes(
    *,
    graph: TopologyGraph,
    paths: dict[str, list[str]],
) -> list[dict[str, Any]]:
    roles_by_node: dict[str, set[str]] = {}
    for role, path in paths.items():
        for viewpoint in path:
            if viewpoint not in graph.nodes:
                roles_by_node.setdefault(viewpoint, set()).add(role)
    return [
        {"viewpoint": viewpoint, "roles": sorted(roles)}
        for viewpoint, roles in sorted(roles_by_node.items())
    ]


def normalize_viewpoint_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    return [str(value) for value in values if value]


def unique_viewpoints(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for viewpoint in normalize_viewpoint_list(values):
        if viewpoint in seen:
            continue
        seen.add(viewpoint)
        result.append(viewpoint)
    return result


def topology_pose_distance_m(item_a: dict[str, Any], item_b: dict[str, Any]) -> float:
    pose_a = item_a["pose"]
    pose_b = item_b["pose"]
    return math.sqrt(
        (pose_a[3] - pose_b[3]) ** 2
        + (pose_a[7] - pose_b[7]) ** 2
        + (pose_a[11] - pose_b[11]) ** 2
    )


def reconstruct_path(previous: dict[str, str], source: str, target: str) -> list[str]:
    path = [target]
    while path[-1] != source:
        parent = previous.get(path[-1])
        if parent is None:
            return []
        path.append(parent)
    path.reverse()
    return path


def same_shortest_state(left: tuple[float, int], right: tuple[float, int]) -> bool:
    return abs(left[0] - right[0]) <= EPS and left[1] == right[1]


def is_better_shortest(candidate: tuple[float, int], previous: tuple[float, int]) -> bool:
    return candidate[0] < previous[0] - EPS or (
        abs(candidate[0] - previous[0]) <= EPS and candidate[1] < previous[1]
    )


def build_replay_frames(trajectory: list[str], trace_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace_steps:
        return [
            {
                "kind": "legacy_viewpoint",
                "viewpoint": viewpoint,
                "trajectory_index": index,
            }
            for index, viewpoint in enumerate(trajectory)
        ]

    frames: list[dict[str, Any]] = []
    for trace_step in trace_steps:
        route = list(trace_step.get("route_viewpoints") or [])
        if not route:
            current = trace_step.get("current_viewpoint")
            route = [current] if current else []
        if not route:
            continue

        decision_step = trace_step.get("step")
        frames.append(
            {
                "kind": "decision",
                "viewpoint": route[0],
                "decision_step": decision_step,
                "decision": trace_step,
            }
        )

        selected = trace_step.get("selected") or {}
        selection_kind = selected.get("selection_kind")
        should_show_route = (
            selection_kind == "global_node"
            or len(route) > 2
            or (selection_kind == "stop" and len(route) > 1)
        )
        if should_show_route:
            hop_count = max(len(route) - 1, 0)
            for hop_index, viewpoint in enumerate(route[1:], start=1):
                frames.append(
                    {
                        "kind": "route",
                        "viewpoint": viewpoint,
                        "decision_step": decision_step,
                        "hop_index": hop_index,
                        "hop_count": hop_count,
                        "selection_kind": selection_kind,
                        "selected_viewpoint": selected.get("executed_viewpoint") or selected.get("viewpoint"),
                    }
                )

    return frames or [
        {
            "kind": "legacy_viewpoint",
            "viewpoint": viewpoint,
            "trajectory_index": index,
        }
        for index, viewpoint in enumerate(trajectory)
    ]


def annotate_decision_sheet(
    *,
    sheet: Any,
    frame: dict[str, Any],
    width: int,
    height: int,
    show_topk: int | None,
    header_font_size: int,
    candidate_font_size: int,
) -> None:
    from PIL import ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel

    draw = ImageDraw.Draw(sheet)
    header_font = load_overlay_font(header_font_size)
    candidate_font = load_overlay_font(candidate_font_size)
    decision = frame["decision"]
    selected = decision.get("selected") or {}
    visible_candidates = list(decision.get("visible_candidates") or [])
    ranked_candidates = rank_visible_candidates(visible_candidates)
    allowed = {
        id(candidate)
        for candidate in (
            ranked_candidates[:show_topk]
            if show_topk is not None and show_topk >= 0
            else ranked_candidates
        )
    }

    selected_candidate = None
    for candidate in visible_candidates:
        if candidate.get("is_selected"):
            selected_candidate = candidate
            allowed.add(id(candidate))

    header = format_decision_header(decision)
    draw_text_box(draw, (8, 8), header, header_font, fill=(255, 255, 255), background=(0, 0, 0))
    draw_stop_badge(draw, sheet.size, decision, header_font)

    first_hop_point_id = first_hop_point_id_for_global(decision)
    for rank, candidate in enumerate(ranked_candidates, start=1):
        if id(candidate) not in allowed:
            continue
        point_id = candidate.get("point_id")
        if point_id is None:
            continue
        box = point_id_box(int(point_id), width, height)
        prob = candidate_display_prob(candidate)
        label = f"rank={rank}\n{decision_score_name(decision)}={format_prob(prob)}"
        draw_text_box(
            draw,
            (box[0] + 8, box[1] + 8),
            label,
            candidate_font,
            fill=(70, 150, 255),
            background=(0, 0, 0),
        )

    if selected.get("selection_kind") == "current_visible" and selected_candidate is not None:
        point_id = selected_candidate.get("point_id")
        if point_id is not None:
            draw.rectangle(point_id_box(int(point_id), width, height), outline=(255, 0, 0), width=8)
    elif selected.get("selection_kind") == "global_node" and first_hop_point_id is not None:
        draw_dashed_rectangle(
            draw,
            point_id_box(int(first_hop_point_id), width, height),
            fill=(255, 0, 0),
            width=6,
        )


def annotate_route_sheet(*, sheet: Any, frame: dict[str, Any], route_font_size: int) -> None:
    from PIL import ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel

    draw = ImageDraw.Draw(sheet)
    font = load_overlay_font(route_font_size)
    selected = short_id(frame.get("selected_viewpoint"))
    text = (
        f"route from decision {frame.get('decision_step')} "
        f"hop {frame.get('hop_index')}/{frame.get('hop_count')} "
        f"to {selected}; not a new model decision"
    )
    draw_text_box(draw, (8, 8), text, font, fill=(255, 80, 80), background=(0, 0, 0))


def format_decision_header(decision: dict[str, Any]) -> str:
    selected = decision.get("selected") or {}
    step = decision.get("step")
    kind = selected.get("selection_kind")
    fusion = decision.get("fusion")
    score_name = decision_score_name(decision)
    selected_vp = short_id(selected.get("executed_viewpoint") or selected.get("viewpoint"))
    selected_prob = format_prob(selected.get("prob"))
    fuse_weight = decision.get("fuse_weight")
    fuse_text = "--" if fuse_weight is None else f"{float(fuse_weight):.2f}"
    return (
        f"decision {step} {fusion}/{kind} score={score_name} "
        f"selected={selected_vp} p={selected_prob} fuse={fuse_text}"
    )


def decision_score_name(decision: dict[str, Any]) -> str:
    fusion = decision.get("fusion")
    if fusion == "local":
        return "local"
    if fusion == "global":
        return "global"
    return "fused"


def rank_visible_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate_display_prob(candidate) is not None,
            candidate_display_prob(candidate) or -1.0,
        ),
        reverse=True,
    )


def candidate_display_prob(candidate: dict[str, Any]) -> float | None:
    for score_name in ("fused", "global", "local"):
        score = candidate.get(score_name) or {}
        prob = score.get("prob")
        if prob is not None:
            return float(prob)
    return None


def draw_stop_badge(draw: Any, image_size: tuple[int, int], decision: dict[str, Any], font: Any) -> None:
    text = f"STOP {format_prob(decision.get('stop_prob'))}"
    padding = text_padding(font)
    bbox = text_bbox(draw, (0, 0), text, font)
    text_width = bbox[2] - bbox[0]
    left = max(8, image_size[0] - text_width - padding * 2 - 8)
    draw_text_box(
        draw,
        (left + padding, 8 + padding),
        text,
        font,
        fill=(255, 255, 255),
        background=(150, 30, 30),
        padding=padding,
    )


def first_hop_point_id_for_global(decision: dict[str, Any]) -> int | None:
    route = decision.get("route_viewpoints") or []
    if len(route) < 2:
        return None
    first_hop = route[1]
    for candidate in decision.get("visible_candidates") or []:
        if candidate.get("viewpoint") == first_hop:
            point_id = candidate.get("point_id")
            return int(point_id) if point_id is not None else None
    return None


def point_id_box(point_id: int, width: int, height: int) -> tuple[int, int, int, int]:
    row = point_id // 12
    col = point_id % 12
    return (col * width, row * height, (col + 1) * width - 1, (row + 1) * height - 1)


def draw_text_box(
    draw: Any,
    xy: tuple[int, int],
    text: str,
    font: Any,
    *,
    fill: tuple[int, int, int],
    background: tuple[int, int, int],
    padding: int | None = None,
) -> None:
    left, top = xy
    bbox = text_bbox(draw, (left, top), text, font)
    padding = text_padding(font) if padding is None else padding
    draw.rectangle(
        (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
        fill=background,
    )
    draw.multiline_text((left, top), text, font=font, fill=fill, spacing=text_line_spacing(font))


def load_overlay_font(size: int) -> Any:
    from PIL import ImageFont  # pylint: disable=import-outside-toplevel

    size = max(int(size), 8)
    for font_name in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_bbox(draw: Any, xy: tuple[int, int], text: str, font: Any) -> tuple[int, int, int, int]:
    if "\n" in text:
        try:
            return draw.multiline_textbbox(xy, text, font=font, spacing=text_line_spacing(font))
        except AttributeError:
            line_boxes = [
                text_bbox(draw, (0, 0), line, font)
                for line in text.splitlines()
            ]
            width = max((box[2] - box[0] for box in line_boxes), default=0)
            height = sum((box[3] - box[1] for box in line_boxes), 0)
            height += text_line_spacing(font) * max(len(line_boxes) - 1, 0)
            return (xy[0], xy[1], xy[0] + width, xy[1] + height)
    try:
        return draw.textbbox(xy, text, font=font)
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)
        return (xy[0], xy[1], xy[0] + text_width, xy[1] + text_height)


def text_padding(font: Any) -> int:
    size = getattr(font, "size", 12)
    return max(4, int(size * 0.22))


def text_line_spacing(font: Any) -> int:
    size = getattr(font, "size", 12)
    return max(2, int(size * 0.15))


def draw_dashed_rectangle(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: int = 24,
    gap: int = 14,
) -> None:
    left, top, right, bottom = box
    draw_dashed_line(draw, (left, top), (right, top), fill=fill, width=width, dash=dash, gap=gap)
    draw_dashed_line(draw, (right, top), (right, bottom), fill=fill, width=width, dash=dash, gap=gap)
    draw_dashed_line(draw, (right, bottom), (left, bottom), fill=fill, width=width, dash=dash, gap=gap)
    draw_dashed_line(draw, (left, bottom), (left, top), fill=fill, width=width, dash=dash, gap=gap)


def draw_dashed_line(
    draw: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        draw.line(
            (
                x1 + dx * distance,
                y1 + dy * distance,
                x1 + dx * segment_end,
                y1 + dy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        distance += dash + gap


def format_prob(prob: Any) -> str:
    if prob is None:
        return "--"
    percent = float(prob) * 100.0
    if abs(percent) < 1e-12:
        return "0%"
    if percent >= 10:
        return f"{percent:.0f}%"
    if percent >= 1:
        return f"{percent:.1f}%"
    if percent >= 0.01:
        return f"{percent:.2f}%"
    return f"{percent:.2e}%"


def short_id(value: Any) -> str:
    if value is None:
        return "None"
    text = str(value)
    return text if len(text) <= 8 else text[:8]


def init_sim(
    *,
    dataset_path: Path,
    nav_graph_path: Path,
    width: int,
    height: int,
    vfov_deg: float,
) -> Any:
    import MatterSim  # pylint: disable=import-outside-toplevel

    sim = MatterSim.Simulator()
    sim.setDatasetPath(str(dataset_path))
    sim.setNavGraphPath(str(nav_graph_path))
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov_deg))
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setDiscretizedViewingAngles(False)
    sim.initialize()
    return sim


def render_single_view(
    sim: Any,
    scan: str,
    viewpoint: str,
    heading_deg: float,
    elevation_deg: float,
) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    sim.newEpisode(
        [scan],
        [viewpoint],
        [math.radians(heading_deg)],
        [math.radians(elevation_deg)],
    )
    state = sim.getState()[0]
    rgb = np.array(state.rgb, copy=True)
    rgb = rgb[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(rgb)


def build_contact_sheet(
    *,
    sim: Any,
    scan: str,
    viewpoint: str,
    width: int,
    height: int,
) -> Any:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    elevations = [-30.0, 0.0, 30.0]
    headings = list(range(0, 360, 30))

    sheet = Image.new("RGB", (width * len(headings), height * len(elevations)))
    for row, elevation in enumerate(elevations):
        for col, heading in enumerate(headings):
            frame = render_single_view(sim, scan, viewpoint, heading, elevation)
            sheet.paste(frame, (col * width, row * height))
    return sheet


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
