#!/usr/bin/env python3
"""Sample completion rule labels into a human audit CSV template."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULE_DIR = (
    REPO_ROOT
    / "reports"
    / "artifacts"
    / "completion_labels"
    / "0013_same_val_all_r2r_reverie_cvdn_soon_same_s0_v4"
    / "rule_v0"
)
DEFAULT_LABELS_JSONL = DEFAULT_RULE_DIR / "completion_rule_labels_v0.jsonl"
DEFAULT_AUDIT_DIR = DEFAULT_RULE_DIR / "audits" / "audit_round_001"
DEFAULT_SPLITS = ("val_seen", "val_train_seen")
DEFAULT_SEED = 20260427

AUDIT_FIELDNAMES = [
    "audit_round",
    "audit_stratum",
    "dataset",
    "split",
    "evidence_scope_primary",
    "target_type",
    "object_reference_role",
    "local_visual_judgability",
    "internal_item_id",
    "saved_instr_id",
    "instruction",
    "matched_phrases_json",
    "evidence_scope_secondary_json",
    "has_object_goal",
    "has_room_region_goal",
    "has_spatial_relation",
    "has_attribute_description",
    "has_temporal_route_order",
    "has_level_floor_constraint",
    "has_dialog_history",
    "human_verdict",
    "corrected_target_type",
    "corrected_evidence_scope_primary",
    "corrected_local_visual_judgability",
    "corrected_has_object_goal",
    "corrected_has_room_region_goal",
    "corrected_has_spatial_relation",
    "corrected_has_attribute_description",
    "corrected_has_temporal_route_order",
    "corrected_has_level_floor_constraint",
    "corrected_has_dialog_history",
    "corrected_object_reference_role",
    "error_type",
    "action",
    "note",
    "reviewer",
    "reviewed_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic random audit sample from completion rule labels."
    )
    parser.add_argument(
        "--labels-jsonl",
        type=Path,
        default=DEFAULT_LABELS_JSONL,
        help=f"Combined rule-label JSONL. Default: {DEFAULT_LABELS_JSONL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help=f"Audit output directory. Default: {DEFAULT_AUDIT_DIR}",
    )
    parser.add_argument(
        "--audit-round",
        default="rule_v0_round_001",
        help="Audit round identifier written into the CSV.",
    )
    parser.add_argument(
        "--per-bucket",
        type=positive_int,
        default=20,
        help="Maximum examples per dataset x evidence_scope_primary bucket. Default: 20.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=(
            "Splits to sample. Use 'all' to include every split in the label file. "
            f"Default: {' '.join(DEFAULT_SPLITS)}"
        ),
    )
    parser.add_argument(
        "--high-complexity-count",
        type=non_negative_int,
        default=0,
        help="Extra high-complexity examples to sample after evidence buckets. Default: 0.",
    )
    parser.add_argument(
        "--random-count",
        type=non_negative_int,
        default=0,
        help="Extra random baseline examples to sample after evidence buckets. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed. Default: {DEFAULT_SEED}.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing audit files.")
    return parser.parse_args()


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def main() -> None:
    args = parse_args()
    sample_audit(
        labels_jsonl=args.labels_jsonl.resolve(),
        output_dir=args.output_dir.resolve(),
        audit_round=args.audit_round,
        per_bucket=args.per_bucket,
        high_complexity_count=args.high_complexity_count,
        random_count=args.random_count,
        splits=args.splits,
        seed=args.seed,
        overwrite=args.overwrite,
    )


def sample_audit(
    *,
    labels_jsonl: Path,
    output_dir: Path,
    audit_round: str,
    per_bucket: int,
    high_complexity_count: int,
    random_count: int,
    splits: list[str],
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not labels_jsonl.exists():
        raise FileNotFoundError(f"Rule-label JSONL not found: {labels_jsonl}")
    ensure_output_paths(output_dir, overwrite)

    selected_splits = None if splits == ["all"] else set(splits)
    rows = [
        row
        for row in read_jsonl(labels_jsonl)
        if selected_splits is None or row.get("split") in selected_splits
    ]
    if not rows:
        raise ValueError(f"No rows matched splits={splits!r} in {labels_jsonl}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    high_complexity_pool: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("dataset", ""), row.get("completion_evidence", {}).get("evidence_scope_primary", ""))
        grouped[key].append(row)
        if row.get("instruction_semantics", {}).get("language_complexity") == "high":
            high_complexity_pool.append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    bucket_counts: Counter[tuple[str, str]] = Counter()
    stratum_tags: dict[int, str] = {}

    # Stratum 1: per evidence-bucket sampling
    for key in sorted(grouped):
        bucket_rows = grouped[key][:]
        rng.shuffle(bucket_rows)
        picked = bucket_rows[:per_bucket]
        for row in picked:
            stratum_tags[id(row)] = "evidence_bucket"
        selected.extend(picked)
        bucket_counts[key] = len(picked)

    if high_complexity_count:
        already = {id(row) for row in selected}
        high_available = [row for row in high_complexity_pool if id(row) not in already]
        rng.shuffle(high_available)
        high_picked = high_available[:min(high_complexity_count, len(high_available))]
        for row in high_picked:
            stratum_tags[id(row)] = "high_complexity"
        selected.extend(high_picked)

    if random_count:
        already = {id(row) for row in selected}
        random_pool = [row for row in rows if id(row) not in already]
        rng.shuffle(random_pool)
        random_picked = random_pool[:min(random_count, len(random_pool))]
        for row in random_picked:
            stratum_tags[id(row)] = "random"
        selected.extend(random_picked)

    # Tag each row with its stratum
    for row in selected:
        row["audit_stratum"] = stratum_tags.get(id(row), "evidence_bucket")

    selected.sort(
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("completion_evidence", {}).get("evidence_scope_primary", "")),
            str(row.get("split", "")),
            str(row.get("identity", {}).get("internal_item_id", "")),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "audit_round_001_to_fill.csv"
    jsonl_path = output_dir / "audit_round_001_sample.jsonl"
    summary_path = output_dir / "audit_round_001_summary.md"
    instructions_path = output_dir / "audit_round_001_instructions.md"
    manifest_path = output_dir / "manifest.json"

    write_audit_csv(csv_path, selected, audit_round)
    write_jsonl(jsonl_path, selected)
    write_summary(
        summary_path,
        labels_jsonl,
        rows,
        selected,
        bucket_counts,
        per_bucket,
        high_complexity_count,
        random_count,
        seed,
        splits,
    )
    write_instructions(instructions_path)

    manifest = {
        "audit_round": audit_round,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "labels_jsonl": path_to_string(labels_jsonl),
        "output_dir": path_to_string(output_dir),
        "splits": "all" if selected_splits is None else sorted(selected_splits),
        "seed": seed,
        "per_bucket": per_bucket,
        "counts": {
            "pool_rows": len(rows),
            "sample_rows": len(selected),
            "buckets": len(grouped),
            "high_complexity_pool_rows": len(high_complexity_pool),
        },
        "sampling": {
            "per_bucket": per_bucket,
            "high_complexity_count": high_complexity_count,
            "random_count": random_count,
        },
        "files": {
            "to_fill_csv": path_to_string(csv_path),
            "sample_jsonl": path_to_string(jsonl_path),
            "summary_md": path_to_string(summary_path),
            "instructions_md": path_to_string(instructions_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def write_audit_csv(path: Path, rows: list[dict[str, Any]], audit_round: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            identity = row.get("identity", {})
            semantics = row.get("instruction_semantics", {})
            evidence = row.get("completion_evidence", {})
            writer.writerow(
                {
                    "audit_round": audit_round,
                    "audit_stratum": row.get("audit_stratum", ""),
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "evidence_scope_primary": evidence.get("evidence_scope_primary"),
                    "target_type": semantics.get("target_type"),
                    "object_reference_role": semantics.get("object_reference_role"),
                    "local_visual_judgability": evidence.get("local_visual_judgability"),
                    "internal_item_id": identity.get("internal_item_id"),
                    "saved_instr_id": identity.get("saved_instr_id"),
                    "instruction": row.get("annotation", {}).get("instruction"),
                    "matched_phrases_json": json.dumps(row.get("matched_phrases", {}), ensure_ascii=False, sort_keys=True),
                    "evidence_scope_secondary_json": json.dumps(
                        evidence.get("evidence_scope_secondary", []), ensure_ascii=False
                    ),
                    "has_object_goal": semantics.get("has_object_goal"),
                    "has_room_region_goal": semantics.get("has_room_region_goal"),
                    "has_spatial_relation": semantics.get("has_spatial_relation"),
                    "has_attribute_description": semantics.get("has_attribute_description"),
                    "has_temporal_route_order": semantics.get("has_temporal_route_order"),
                    "has_level_floor_constraint": semantics.get("has_level_floor_constraint"),
                    "has_dialog_history": semantics.get("has_dialog_history"),
                    "human_verdict": "",
                    "corrected_target_type": "",
                    "corrected_evidence_scope_primary": "",
                    "corrected_local_visual_judgability": "",
                    "corrected_has_object_goal": "",
                    "corrected_has_room_region_goal": "",
                    "corrected_has_spatial_relation": "",
                    "corrected_has_attribute_description": "",
                    "corrected_has_temporal_route_order": "",
                    "corrected_has_level_floor_constraint": "",
                    "corrected_has_dialog_history": "",
                    "corrected_object_reference_role": "",
                    "error_type": "",
                    "action": "",
                    "note": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )


def write_summary(
    path: Path,
    labels_jsonl: Path,
    pool_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    bucket_counts: Counter[tuple[str, str]],
    per_bucket: int,
    high_complexity_count: int,
    random_count: int,
    seed: int,
    splits: list[str],
) -> None:
    pool_counts: Counter[tuple[str, str]] = Counter()
    for row in pool_rows:
        pool_counts[(row["dataset"], row["completion_evidence"]["evidence_scope_primary"])] += 1

    stratum_counts: Counter[str] = Counter()
    for row in selected:
        stratum_counts[row.get("audit_stratum", "evidence_bucket")] += 1

    lines = [
        "# Rule v0 Audit Round 001 Summary",
        "",
        f"- Source labels: `{path_to_string(labels_jsonl)}`",
        f"- Splits: `{' '.join(splits)}`",
        f"- Seed: `{seed}`",
        f"- Target per evidence bucket: `{per_bucket}`",
        f"- Extra high-complexity target: `{high_complexity_count}`",
        f"- Extra random target: `{random_count}`",
        f"- Pool rows: `{len(pool_rows)}`",
        f"- Sample rows: `{len(selected)}`",
        "",
        "## Stratum Counts",
        "",
        "| stratum | count |",
        "|---|---:|",
    ]
    for stratum in ("evidence_bucket", "high_complexity", "random"):
        count = stratum_counts.get(stratum, 0)
        lines.append(f"| {stratum} | {count} |")
    lines.extend(
        [
            "",
            "## Evidence Bucket Counts",
            "",
            "| dataset | evidence_scope_primary | pool | sampled | note |",
            "|---|---|---:|---:|---|",
        ]
    )
    for key in sorted(pool_counts):
        pool_count = pool_counts[key]
        sample_count = bucket_counts.get(key, 0)
        note = "short bucket" if sample_count < min(per_bucket, pool_count) else ""
        lines.append(f"| {key[0]} | `{key[1]}` | {pool_count} | {sample_count} | {note} |")
    lines.extend(
        [
            "",
            "## Fill Status",
            "",
            "- Fill `audit_round_001_to_fill.csv` columns from `human_verdict` onward.",
            "- After filling, use the summary counts to decide which rule/schema changes are needed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instructions(path: Path) -> None:
    lines = [
        "# Rule v0 Human Audit Instructions",
        "",
        "只需要填写 CSV 中 `human_verdict` 及其右侧列。左侧列是规则输出和抽样上下文，不要改。",
        "",
        "## Required Columns",
        "",
        "- `human_verdict`: `accept`, `reject`, or `uncertain`.",
        "- `corrected_target_type`: 只有 reject/uncertain 且你能判断时填写；可选值 `object`, `room`, `region`, `route_endpoint`, `dialog_goal`, `mixed`, `unknown`.",
        "- `corrected_evidence_scope_primary`: 只有 reject/uncertain 且你能判断时填写；可选值 `L0_object_visible`, `L1_room_region`, `L2_local_relation`, `H1_route_progress`, `H2_dialog_dependent`, `G_search_disambiguation`, `M_mixed_local_route`, `unknown`.",
        "- `corrected_local_visual_judgability`: 可选值 `0`, `1`, `2`；不知道就留空。",
        "- `corrected_has_*`: 只有对应布尔字段错了才填 `true` 或 `false`；`has_object_goal` 在 v0 中表示目标/完成语义里是否有 object cue，不表示最终视野必须对准物体。",
        "- `corrected_object_reference_role`: 可选值 `none`, `target_object`, `goal_room_cue`, `route_landmark_or_endpoint`, `mentioned_object`。",
        "- `error_type`: 推荐值 `wrong_primary_scope`, `missed_object`, `missed_room_region`, `missed_spatial_relation`, `missed_attribute`, `missed_dialog`, `missed_route_history`, `overtriggered_keyword`, `schema_gap`, `duplicate_or_unclear`, `other`.",
        "- `action`: 推荐值 `keep_rule`, `fix_rule`, `fix_schema`, `add_to_llm_fewshot`, `ignore_for_v0`.",
        "- `note`: 简短说明为什么错，或为什么 uncertain。",
        "- `reviewer`: 你的名字或代号。",
        "- `reviewed_at`: 建议使用 `YYYY-MM-DD`。",
        "",
        "## Audit Goal",
        "",
        "这一轮不是正式人工标签集。目标是估计规则高精度桶是否可信，并收集下一轮规则/schema/LLM few-shot 的修改依据。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_output_paths(output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    paths = [
        output_dir / "audit_round_001_to_fill.csv",
        output_dir / "audit_round_001_sample.jsonl",
        output_dir / "audit_round_001_summary.md",
        output_dir / "audit_round_001_instructions.md",
        output_dir / "manifest.json",
    ]
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists; pass --overwrite to replace:\n{joined}")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_to_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


if __name__ == "__main__":
    main()
