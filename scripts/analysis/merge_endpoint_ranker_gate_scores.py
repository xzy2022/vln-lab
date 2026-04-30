#!/usr/bin/env python3
"""Merge candidate scores from a ranker CSV with gate scores from a gate CSV."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_endpoint_reranker as reranker_eval  # noqa: E402
import train_endpoint_ranker_baseline as baseline  # noqa: E402


SCHEMA_VERSION = "endpoint_merged_ranker_gate_scores.v1"
DEFAULT_OUTPUT_NAME = "merged_ranker_gate_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace ranker score CSV gate_score with scores from another gate CSV.",
    )
    parser.add_argument("--ranker-score-csv", required=True)
    parser.add_argument("--gate-score-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--candidate-score-column", default=None)
    parser.add_argument("--gate-score-column", default=None)
    parser.add_argument("--default-gate-score", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else Path(args.ranker_score_csv).resolve().parent / DEFAULT_OUTPUT_NAME
    )
    manifest = merge_ranker_gate_scores(
        ranker_score_csv=Path(args.ranker_score_csv).resolve(),
        gate_score_csv=Path(args.gate_score_csv).resolve(),
        output_csv=output_csv,
        candidate_score_column=args.candidate_score_column,
        gate_score_column=args.gate_score_column,
        default_gate_score=args.default_gate_score,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def merge_ranker_gate_scores(
    ranker_score_csv: Path,
    gate_score_csv: Path,
    output_csv: Path,
    candidate_score_column: str | None = None,
    gate_score_column: str | None = None,
    default_gate_score: float | None = None,
) -> dict[str, Any]:
    ranker = pd.read_csv(ranker_score_csv, low_memory=False)
    gate = pd.read_csv(gate_score_csv, low_memory=False)
    if "candidate_id" not in ranker.columns:
        raise ValueError(f"Ranker score CSV must contain candidate_id: {ranker_score_csv}")
    if "episode_id" not in ranker.columns:
        raise ValueError(f"Ranker score CSV must contain episode_id: {ranker_score_csv}")
    if "episode_id" not in gate.columns:
        raise ValueError(f"Gate score CSV must contain episode_id: {gate_score_csv}")

    resolved_candidate_score = resolve_column(
        ranker.columns.tolist(),
        explicit=candidate_score_column,
        aliases=reranker_eval.CANDIDATE_SCORE_ALIASES,
        label="candidate score",
    )
    resolved_gate_score = resolve_column(
        gate.columns.tolist(),
        explicit=gate_score_column,
        aliases=reranker_eval.GATE_SCORE_ALIASES,
        label="gate score",
    )

    gate_map = build_gate_map(gate, resolved_gate_score)
    output = ranker.copy()
    if resolved_candidate_score != "candidate_score":
        output["candidate_score"] = pd.to_numeric(output[resolved_candidate_score], errors="coerce")
    else:
        output["candidate_score"] = pd.to_numeric(output["candidate_score"], errors="coerce")
    output["gate_score"] = output["episode_id"].astype(str).map(gate_map)
    if default_gate_score is not None:
        output["gate_score"] = output["gate_score"].fillna(default_gate_score)
    if output["gate_score"].isna().any():
        missing = output.loc[output["gate_score"].isna(), "episode_id"].iloc[0]
        raise ValueError(f"Missing replacement gate score for episode_id={missing}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ranker_score_csv": baseline.path_to_string(ranker_score_csv),
        "gate_score_csv": baseline.path_to_string(gate_score_csv),
        "candidate_score_column": resolved_candidate_score,
        "gate_score_column": resolved_gate_score,
        "default_gate_score": default_gate_score,
        "files": {
            "merged_score_csv": baseline.path_to_string(output_csv),
        },
        "counts": {
            "rows": int(len(output)),
            "episodes": int(output["episode_id"].nunique()),
        },
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    baseline.write_json(manifest_path, manifest)
    manifest["files"]["manifest_json"] = baseline.path_to_string(manifest_path)
    baseline.write_json(manifest_path, manifest)
    return manifest


def build_gate_map(gate: pd.DataFrame, gate_score_column: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for episode_id, rows in gate.groupby("episode_id", sort=False):
        scores = pd.to_numeric(rows[gate_score_column], errors="coerce").dropna().unique()
        if len(scores) == 0:
            continue
        if len(scores) > 1 and float(scores.max() - scores.min()) > 1e-12:
            raise ValueError(f"Conflicting gate scores for episode_id={episode_id}")
        values[str(episode_id)] = float(scores[0])
    return values


def resolve_column(columns: list[str], explicit: str | None, aliases: tuple[str, ...], label: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Missing explicit {label} column {explicit!r}")
        return explicit
    for alias in aliases:
        if alias in columns:
            return alias
    raise ValueError(f"Could not resolve {label} column from {columns}")


if __name__ == "__main__":
    main()
