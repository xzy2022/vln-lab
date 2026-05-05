#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "navnuances" / "same" / "R2R"
DEFAULT_NAVGPT2_DATA_ROOT = Path(
    os.environ.get("NAVGPT2_DATASETS", REPO_ROOT / "data" / "navgpt2" / "datasets")
)
DEFAULT_SPLITS = ("DC", "LR", "RR", "VM", "NU")
STANDARD_SPLIT = "val_unseen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or symlink NavNuances R2R encoded splits into the NavGPT-2 "
            "R2R annotation directory."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing R2R_DC_enc.json, R2R_LR_enc.json, ...",
    )
    parser.add_argument(
        "--navgpt2-data-root",
        type=Path,
        default=DEFAULT_NAVGPT2_DATA_ROOT,
        help="NavGPT-2 datasets root. Defaults to $NAVGPT2_DATASETS or data/navgpt2/datasets.",
    )
    parser.add_argument(
        "--target-annotations-dir",
        type=Path,
        help=(
            "Exact NavGPT-2 R2R annotations directory. Default: "
            "<navgpt2-data-root>/R2R/annotations."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="NavNuances split suffixes to install. Default: DC LR RR VM NU.",
    )
    parser.add_argument(
        "--standard",
        choices=("auto", "include", "skip"),
        default="skip",
        help=(
            "How to handle R2R_val_unseen_enc.json from the NavNuances copy. "
            "Default skips it so official NavGPT-2 R2R annotations are not overwritten."
        ),
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


def annotation_filename(split: str) -> str:
    normalized = normalize_split(split)
    return f"R2R_{normalized}_enc.json"


def resolve_splits(
    source_dir: Path,
    splits: list[str] | tuple[str, ...],
    standard: str,
) -> list[str]:
    resolved = [normalize_split(split) for split in splits]
    if standard == "include":
        resolved.append(STANDARD_SPLIT)
    elif standard == "auto" and (source_dir / annotation_filename(STANDARD_SPLIT)).exists():
        resolved.append(STANDARD_SPLIT)

    unique: list[str] = []
    seen: set[str] = set()
    for split in resolved:
        if split not in seen:
            unique.append(split)
            seen.add(split)
    return unique


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

    if target_path.exists() or target_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {repo_rel(target_path)}")
        if dry_run:
            return "would overwrite"
        target_path.unlink()

    if dry_run:
        return "would install"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source_path, target_path)
    elif mode == "symlink":
        os.symlink(source_path.resolve(), target_path)
    else:
        raise ValueError(f"Unsupported install mode: {mode}")
    return mode


def install_navnuances_annotations(
    *,
    source_dir: Path,
    target_annotations_dir: Path,
    splits: list[str] | tuple[str, ...] = DEFAULT_SPLITS,
    standard: str = "skip",
    mode: str = "copy",
    overwrite: bool = True,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    source_dir = source_dir.resolve()
    target_annotations_dir = target_annotations_dir.resolve()
    resolved_splits = resolve_splits(source_dir, splits, standard)

    if not dry_run:
        target_annotations_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for split in resolved_splits:
        filename = annotation_filename(split)
        source_path = source_dir / filename
        target_path = target_annotations_dir / filename
        status = install_file(
            source_path,
            target_path,
            mode=mode,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        manifest.append(
            {
                "split": split,
                "source": repo_rel(source_path),
                "target": repo_rel(target_path),
                "status": status,
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    target_annotations_dir = (
        args.target_annotations_dir
        or args.navgpt2_data_root / "R2R" / "annotations"
    )
    manifest = install_navnuances_annotations(
        source_dir=args.source_dir,
        target_annotations_dir=target_annotations_dir,
        splits=args.splits,
        standard=args.standard,
        mode=args.mode,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
    )

    for entry in manifest:
        print(
            f"{entry['status']}: {entry['split']} "
            f"{entry['source']} -> {entry['target']}",
            flush=True,
        )
    print(f"Done: {len(manifest)} NavNuances annotation files", flush=True)


if __name__ == "__main__":
    main()
