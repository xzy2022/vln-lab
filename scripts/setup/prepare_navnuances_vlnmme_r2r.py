#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "navnuances" / "same" / "R2R"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "navnuances" / "vlnmme" / "R2R"
DEFAULT_SPLITS = ("DC", "LR", "RR", "VM", "NU")
STANDARD_SPLIT = "val_unseen"
REQUIRED_KEYS = ("scan", "path_id", "path", "heading", "instructions", "instr_encodings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install NavNuances R2R encoded splits into a VLN-MME-readable "
            "data root under data/navnuances/vlnmme."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing R2R_DC_enc.json, R2R_LR_enc.json, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="VLN-MME R2R annotation output directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="NavNuances split suffixes to install. Default: DC LR RR VM NU.",
    )
    parser.add_argument(
        "--standard",
        choices=("include", "skip"),
        default="skip",
        help="Whether to also install R2R_val_unseen_enc.json. Default skips it.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Install files by copying or by absolute symlink. Default: copy.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if a target file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned file operations without writing anything.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def normalize_split(split: str) -> str:
    split = Path(split.strip()).name
    if split.lower().endswith(".json"):
        split = split[:-5]
    if split.lower().startswith("r2r_"):
        split = split[4:]
    if split.lower().endswith("_enc"):
        split = split[:-4]
    if split.lower() in {"standard", STANDARD_SPLIT}:
        return STANDARD_SPLIT
    return split.upper()


def encoded_filename(split: str) -> str:
    return f"R2R_{normalize_split(split)}_enc.json"


def resolve_splits(splits: list[str], standard: str) -> list[str]:
    resolved = [normalize_split(split) for split in splits]
    if standard == "include":
        resolved.append(STANDARD_SPLIT)

    unique: list[str] = []
    seen: set[str] = set()
    for split in resolved:
        if split not in seen:
            unique.append(split)
            seen.add(split)
    return unique


def validate_vlnmme_r2r_file(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{repo_rel(path)} must contain a JSON list")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{repo_rel(path)} item {index} must be an object")
        missing = [key for key in REQUIRED_KEYS if key not in item]
        if missing:
            raise ValueError(f"{repo_rel(path)} item {index} is missing keys: {missing}")
        if not isinstance(item["instructions"], list):
            raise ValueError(f"{repo_rel(path)} item {index} instructions must be a list")
        if not isinstance(item["instr_encodings"], list):
            raise ValueError(f"{repo_rel(path)} item {index} instr_encodings must be a list")

    return sum(len(item["instructions"]) for item in data)


def install_file(
    source_path: Path,
    target_path: Path,
    *,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source annotation: {repo_rel(source_path)}")

    count = validate_vlnmme_r2r_file(source_path)

    if target_path.exists() or target_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {repo_rel(target_path)}")
        if not dry_run:
            target_path.unlink()

    if dry_run:
        return f"would install ({count} instructions)"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source_path, target_path)
    elif mode == "symlink":
        os.symlink(source_path.resolve(), target_path)
    else:
        raise ValueError(f"Unsupported install mode: {mode}")
    return f"{mode} ({count} instructions)"


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    splits = resolve_splits(args.splits, args.standard)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        filename = encoded_filename(split)
        source_path = source_dir / filename
        target_path = output_dir / filename
        status = install_file(
            source_path,
            target_path,
            mode=args.mode,
            overwrite=not args.no_overwrite,
            dry_run=args.dry_run,
        )
        print(
            f"{status}: {split} {repo_rel(source_path)} -> {repo_rel(target_path)}",
            flush=True,
        )

    print(f"Done: {len(splits)} VLN-MME NavNuances annotation files", flush=True)


if __name__ == "__main__":
    main()
