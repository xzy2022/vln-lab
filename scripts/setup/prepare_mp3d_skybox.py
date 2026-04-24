#!/usr/bin/env python3
"""Extract Matterport skybox archives and build *_skybox_small.jpg files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np


FACE_COUNT = 6
DEFAULT_SIZE = 512
SKYBOX_FACE_RE = re.compile(r"^(?P<pano>.+)_skybox(?P<face>[0-5])_sami\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Matterport skybox images for MatterSim. "
            "This extracts matterport_skybox_images.zip when needed and "
            "creates <PANO_ID>_skybox_small.jpg files."
        )
    )
    parser.add_argument(
        "--scans-dir",
        type=Path,
        default=None,
        help=(
            "Path to the extracted v1/scans directory. "
            "Defaults to $MP3D_DATA_DIR or $MATTERPORT_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--scan-id",
        action="append",
        default=[],
        help="Scan id to prepare. May be provided multiple times.",
    )
    parser.add_argument(
        "--scan-list-file",
        type=Path,
        default=None,
        help="Optional text file with one scan id per line.",
    )
    parser.add_argument(
        "--from-rendertest-spec",
        type=Path,
        default=None,
        help="Optional path to Matterport3DSimulator/src/test/rendertest_spec.json.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_SIZE,
        help="Downsized face width. Default: 512.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_SIZE,
        help="Downsized face height. Default: 512.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild *_skybox_small.jpg even if it already exists.",
    )
    return parser.parse_args()


def resolve_scans_dir(arg_value: Path | None) -> Path:
    if arg_value is not None:
        return arg_value.expanduser().resolve()

    for env_name in ("MP3D_DATA_DIR", "MATTERPORT_DATA_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value).expanduser().resolve()

    raise SystemExit(
        "Missing --scans-dir. Set it explicitly or export MP3D_DATA_DIR/MATTERPORT_DATA_DIR."
    )


def load_scan_ids(args: argparse.Namespace, scans_dir: Path) -> list[str]:
    ordered_scan_ids: list[str] = []

    for scan_id in args.scan_id:
        if scan_id and scan_id not in ordered_scan_ids:
            ordered_scan_ids.append(scan_id)

    if args.scan_list_file is not None:
        for line in args.scan_list_file.read_text(encoding="utf-8").splitlines():
            scan_id = line.strip()
            if scan_id and scan_id not in ordered_scan_ids:
                ordered_scan_ids.append(scan_id)

    if args.from_rendertest_spec is not None:
        root = json.loads(args.from_rendertest_spec.read_text(encoding="utf-8"))
        for batch in root:
            for item in batch:
                scan_id = item["scanId"]
                if scan_id not in ordered_scan_ids:
                    ordered_scan_ids.append(scan_id)

    if ordered_scan_ids:
        return ordered_scan_ids

    return sorted(path.name for path in scans_dir.iterdir() if path.is_dir())


def ensure_extracted(scans_dir: Path, scan_id: str) -> Path:
    scan_dir = scans_dir / scan_id
    if not scan_dir.exists():
        raise FileNotFoundError(f"Missing scan directory: {scan_dir}")

    skybox_dir = scan_dir / "matterport_skybox_images"
    if skybox_dir.is_dir():
        return skybox_dir

    archive_path = scan_dir / "matterport_skybox_images.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing both extracted skyboxes and archive for scan {scan_id}: {archive_path}"
        )

    print(f"[extract] {scan_id}: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(scans_dir)

    if not skybox_dir.is_dir():
        raise FileNotFoundError(f"Extraction finished but {skybox_dir} was not created")

    return skybox_dir


def find_pano_ids(skybox_dir: Path) -> list[str]:
    pano_ids: dict[str, set[int]] = {}
    for path in skybox_dir.iterdir():
        if not path.is_file():
            continue
        match = SKYBOX_FACE_RE.match(path.name)
        if match is None:
            continue
        pano_ids.setdefault(match.group("pano"), set()).add(int(match.group("face")))

    ready = [pano_id for pano_id, faces in pano_ids.items() if len(faces) == FACE_COUNT]
    return sorted(ready)


def build_small_skyboxes(
    scan_id: str,
    skybox_dir: Path,
    width: int,
    height: int,
    overwrite: bool,
) -> tuple[int, int]:
    pano_ids = find_pano_ids(skybox_dir)
    created = 0
    skipped = 0

    if not pano_ids:
        raise RuntimeError(f"No *_skybox*_sami.jpg faces found in {skybox_dir}")

    for pano_id in pano_ids:
        output_path = skybox_dir / f"{pano_id}_skybox_small.jpg"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        faces = []
        for face_idx in range(FACE_COUNT):
            face_path = skybox_dir / f"{pano_id}_skybox{face_idx}_sami.jpg"
            image = cv2.imread(str(face_path))
            if image is None:
                raise RuntimeError(f"Failed to read {face_path}")
            faces.append(
                cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            )

        merged = np.concatenate(faces, axis=1)
        if not cv2.imwrite(str(output_path), merged):
            raise RuntimeError(f"Failed to write {output_path}")
        created += 1

    print(
        f"[prepare] {scan_id}: panoramas={len(pano_ids)}, created={created}, skipped={skipped}"
    )
    return created, skipped


def main() -> int:
    args = parse_args()
    scans_dir = resolve_scans_dir(args.scans_dir)
    scan_ids = load_scan_ids(args, scans_dir)

    if not scans_dir.is_dir():
        raise SystemExit(f"scans dir does not exist: {scans_dir}")
    if not scan_ids:
        raise SystemExit(f"No scan directories found in {scans_dir}")

    total_created = 0
    total_skipped = 0
    failures: list[str] = []

    for scan_id in scan_ids:
        try:
            skybox_dir = ensure_extracted(scans_dir, scan_id)
            created, skipped = build_small_skyboxes(
                scan_id=scan_id,
                skybox_dir=skybox_dir,
                width=args.width,
                height=args.height,
                overwrite=args.overwrite,
            )
            total_created += created
            total_skipped += skipped
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{scan_id}: {exc}")
            print(f"[error] {scan_id}: {exc}", file=sys.stderr)

    print(
        f"[summary] scans={len(scan_ids)}, created={total_created}, skipped={total_skipped}, "
        f"failures={len(failures)}"
    )
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
