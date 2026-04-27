#!/usr/bin/env python3
"""Extract SAME dataset instructions into one JSONL per dataset split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "same"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "artifacts" / "extract_same_instructions"
SCHEMA_VERSION = "same_instructions.v1"


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    split: str
    relpath: Path


SPLIT_SPECS = [
    SplitSpec("R2R", "train", Path("R2R") / "R2R_train_mergesim_enc.json"),
    SplitSpec("R2R", "val_train_seen", Path("R2R") / "R2R_val_train_seen_enc.json"),
    SplitSpec("R2R", "val_seen", Path("R2R") / "R2R_val_seen_enc.json"),
    SplitSpec("R2R", "val_unseen", Path("R2R") / "R2R_val_unseen_enc.json"),
    SplitSpec("R2R", "test", Path("R2R") / "R2R_test_enc.json"),
    SplitSpec("REVERIE", "train", Path("REVERIE") / "REVERIE_train_enc.json"),
    SplitSpec("REVERIE", "val_train_seen", Path("REVERIE") / "REVERIE_val_train_seen_enc.json"),
    SplitSpec("REVERIE", "val_seen", Path("REVERIE") / "REVERIE_val_seen_enc.json"),
    SplitSpec("REVERIE", "val_unseen", Path("REVERIE") / "REVERIE_val_unseen_enc.json"),
    SplitSpec("REVERIE", "test", Path("REVERIE") / "REVERIE_test_enc.json"),
    SplitSpec("CVDN", "train", Path("CVDN") / "train.json"),
    SplitSpec("CVDN", "val_seen", Path("CVDN") / "val_seen.json"),
    SplitSpec("CVDN", "val_unseen", Path("CVDN") / "val_unseen.json"),
    SplitSpec("CVDN", "test", Path("CVDN") / "test_cleaned.json"),
    SplitSpec("SOON", "train", Path("SOON") / "train_enc_pseudo_obj_ade30k_label.jsonl"),
    SplitSpec("SOON", "val_seen", Path("SOON") / "val_unseen_instrs_enc_pseudo_obj_ade30k_label.jsonl"),
    SplitSpec("SOON", "val_unseen", Path("SOON") / "val_unseen_house_enc_pseudo_obj_ade30k_label.jsonl"),
    SplitSpec("SOON", "test", Path("SOON") / "test_v2_enc.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract instructions from the four SAME datasets "
            "(R2R, REVERIE, CVDN, SOON) into split-level JSONL files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"SAME data directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--max-per-jsonl",
        type=non_negative_int,
        default=None,
        help=(
            "Maximum number of instruction records to write per output JSONL. "
            "Records beyond this limit are discarded. Default: extract all."
        ),
    )
    return parser.parse_args()


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield item


def make_record(
    *,
    dataset: str,
    split: str,
    source_file: Path,
    instruction_id: str,
    instruction: str,
    raw_idx: int,
    sample_idx: int,
    instr_idx: int | None,
    source_ids: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "instruction_id": instruction_id,
        "raw_idx": raw_idx,
        "sample_idx": sample_idx,
        "instr_idx": instr_idx,
        "instruction": instruction,
        "source_file": source_file.as_posix(),
        "source_ids": compact_dict(source_ids),
    }


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def extract_r2r(spec: SplitSpec, source_path: Path) -> Iterator[dict[str, Any]]:
    sample_idx = 0
    for raw_idx, item in enumerate(read_json_list(source_path)):
        path_id = item.get("path_id")
        for instr_idx, instruction in enumerate(item.get("instructions", [])):
            yield make_record(
                dataset=spec.dataset,
                split=spec.split,
                source_file=display_path(source_path),
                instruction_id=f"r2r_{path_id}_{instr_idx}",
                instruction=instruction,
                raw_idx=raw_idx,
                sample_idx=sample_idx,
                instr_idx=instr_idx,
                source_ids={
                    "path_id": path_id,
                    "scan": item.get("scan"),
                },
            )
            sample_idx += 1


def extract_reverie(spec: SplitSpec, source_path: Path) -> Iterator[dict[str, Any]]:
    sample_idx = 0
    for raw_idx, item in enumerate(read_json_list(source_path)):
        path_id = item.get("path_id")
        obj_id = item.get("objId")
        fallback_id = item.get("id")
        for instr_idx, instruction in enumerate(item.get("instructions", [])):
            if "objId" in item:
                instruction_id = f"reverie_{path_id}_{obj_id}_{instr_idx}"
            else:
                path_id = fallback_id
                instruction_id = f"reverie_{fallback_id}_{instr_idx}"
            yield make_record(
                dataset=spec.dataset,
                split=spec.split,
                source_file=display_path(source_path),
                instruction_id=instruction_id,
                instruction=instruction,
                raw_idx=raw_idx,
                sample_idx=sample_idx,
                instr_idx=instr_idx,
                source_ids={
                    "path_id": path_id,
                    "id": fallback_id,
                    "obj_id": obj_id,
                    "scan": item.get("scan"),
                },
            )
            sample_idx += 1


def extract_cvdn(spec: SplitSpec, source_path: Path) -> Iterator[dict[str, Any]]:
    for sample_idx, item in enumerate(read_json_list(source_path)):
        path_id = item.get("inst_idx")
        yield make_record(
            dataset=spec.dataset,
            split=spec.split,
            source_file=display_path(source_path),
            instruction_id=f"cvdn_{sample_idx}_{path_id}",
            instruction=build_cvdn_instruction(item),
            raw_idx=sample_idx,
            sample_idx=sample_idx,
            instr_idx=None,
            source_ids={
                "path_id": path_id,
                "inst_idx": item.get("inst_idx"),
                "game_idx": item.get("game_idx"),
                "scan": item.get("scan"),
                "target": item.get("target"),
                "dialog_turn_count": len(item.get("dialog_history", [])),
            },
        )


def build_cvdn_instruction(item: dict[str, Any]) -> str:
    instruction = f"The goal room contains a {item.get('target')}.\n"
    sentences = []
    for turn in item.get("dialog_history", []):
        message = str(turn.get("message", ""))
        if not message.endswith(("?", ".")):
            message = f"{message}."

        role = turn.get("role")
        if role == "navigator":
            sentences.append(f"Question: {message}\n")
        elif role == "oracle":
            sentences.append(f"Answer: {message}\n")
        else:
            raise ValueError(f"Unsupported CVDN dialog role: {role!r}")

    instruction += "".join(sentences)
    return instruction[:-1] if instruction.endswith("\n") else instruction


def extract_soon(spec: SplitSpec, source_path: Path) -> Iterator[dict[str, Any]]:
    sample_idx = 0
    for raw_idx, item in enumerate(read_jsonl(source_path)):
        path_id = item.get("path_id")
        for instr_idx, instruction in enumerate(item.get("instructions", [])):
            yield make_record(
                dataset=spec.dataset,
                split=spec.split,
                source_file=display_path(source_path),
                instruction_id=f"soon_{raw_idx}_{path_id}_{instr_idx}",
                instruction=instruction.get("full"),
                raw_idx=raw_idx,
                sample_idx=sample_idx,
                instr_idx=instr_idx,
                source_ids={
                    "path_id": path_id,
                    "scan": item.get("scan"),
                    "obj_name": item.get("obj_name"),
                },
            )
            sample_idx += 1


def extract_records(spec: SplitSpec, source_path: Path) -> Iterator[dict[str, Any]]:
    if spec.dataset == "R2R":
        yield from extract_r2r(spec, source_path)
    elif spec.dataset == "REVERIE":
        yield from extract_reverie(spec, source_path)
    elif spec.dataset == "CVDN":
        yield from extract_cvdn(spec, source_path)
    elif spec.dataset == "SOON":
        yield from extract_soon(spec, source_path)
    else:
        raise ValueError(f"Unsupported dataset: {spec.dataset}")


def limit_label(max_per_jsonl: int | None) -> str:
    if max_per_jsonl is None:
        return "max_all"
    return f"max_{max_per_jsonl}_per_jsonl"


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path, max_per_jsonl: int | None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            if max_per_jsonl is not None and count >= max_per_jsonl:
                break
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_root.resolve() / limit_label(args.max_per_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for spec in SPLIT_SPECS:
        source_path = data_dir / spec.relpath
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source annotation file: {source_path}")

        output_path = output_dir / spec.dataset / f"{spec.split}.jsonl"
        count = write_jsonl(
            extract_records(spec, source_path),
            output_path,
            args.max_per_jsonl,
        )
        total += count
        rel_output = display_path(output_path)
        print(f"{spec.dataset}/{spec.split}: wrote {count} records -> {rel_output}")

    print(f"Done. Wrote {total} records under {display_path(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
