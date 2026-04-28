#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT = REPO_ROOT / "data" / "same" / "R2R" / "R2R_val_unseen_enc.json"
DEFAULT_RIGHT = REPO_ROOT / "data" / "navnuances" / "annotations" / "NavNuances" / "R2R_val_unseen.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two R2R_val_unseen annotation files.")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT, help="First R2R_val_unseen JSON file.")
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT, help="Second R2R_val_unseen JSON file.")
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of differing path_ids to print for each category.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_items(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {repo_rel(path)}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{repo_rel(path)} must contain a JSON list")

    items = {}
    duplicates = []
    for item in data:
        path_id = str(item["path_id"])
        if path_id in items:
            duplicates.append(path_id)
        items[path_id] = item
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{repo_rel(path)} has duplicate path_id values, e.g. {preview}")
    return items


def normalized_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "scan": item.get("scan"),
        "path": item.get("path"),
        "heading": item.get("heading"),
        "instructions": item.get("instructions"),
    }


def print_examples(title: str, values: list[str], limit: int) -> None:
    print(f"{title}: {len(values)}")
    for value in values[:limit]:
        print(f"  {value}")


def main() -> None:
    args = parse_args()
    left = load_items(args.left.resolve())
    right = load_items(args.right.resolve())

    left_ids = set(left)
    right_ids = set(right)
    only_left = sorted(left_ids - right_ids)
    only_right = sorted(right_ids - left_ids)
    shared = sorted(left_ids & right_ids)

    field_diffs = []
    for path_id in shared:
        if normalized_item(left[path_id]) != normalized_item(right[path_id]):
            field_diffs.append(path_id)

    left_instruction_total = sum(len(item.get("instructions", [])) for item in left.values())
    right_instruction_total = sum(len(item.get("instructions", [])) for item in right.values())

    print(f"Left:  {repo_rel(args.left.resolve())}")
    print(f"Right: {repo_rel(args.right.resolve())}")
    print(f"Left paths: {len(left)}")
    print(f"Right paths: {len(right)}")
    print(f"Left instructions: {left_instruction_total}")
    print(f"Right instructions: {right_instruction_total}")
    print(f"Shared path_ids: {len(shared)}")
    print_examples("Only left", only_left, args.show_examples)
    print_examples("Only right", only_right, args.show_examples)
    print_examples("Shared path_ids with field differences", field_diffs, args.show_examples)

    if not only_left and not only_right and not field_diffs:
        print("Result: equivalent after ignoring instr_encodings and extra metadata.")
    else:
        print("Result: different.")


if __name__ == "__main__":
    main()
