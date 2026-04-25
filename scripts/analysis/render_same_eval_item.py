#!/usr/bin/env python3
"""Render one SAME eval item trajectory by internal item id."""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class EvalItemMatch:
    item: dict[str, Any]
    eval_items_path: Path
    dataset: str
    split: str


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
    if not scan_dir.is_dir():
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
