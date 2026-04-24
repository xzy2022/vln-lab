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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = item.get("annotation", {})
    prediction = item.get("prediction", {})
    identity = item.get("identity", {})
    scan = annotation["scan"]
    trajectory = prediction["trajectory"]

    sim = init_sim(
        dataset_path=dataset_path,
        nav_graph_path=nav_graph_path,
        width=width,
        height=height,
        vfov_deg=vfov_deg,
    )
    try:
        for step_idx, viewpoint in enumerate(trajectory):
            sheet = build_contact_sheet(
                sim=sim,
                scan=scan,
                viewpoint=viewpoint,
                width=width,
                height=height,
            )
            sheet.save(output_dir / f"step_{step_idx:03d}_{viewpoint}.jpg", quality=90)

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
            "note": (
                "This is a viewpoint-level replay: each step saves the 36 standard views "
                "for that viewpoint. It is not an exact SAME action-level replay because "
                "eval_items does not store per-step heading/elevation/action traces by default."
            ),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        sim.close()


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
