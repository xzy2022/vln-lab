from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "build_same_fine_metrics.py"
SPEC = importlib.util.spec_from_file_location("build_same_fine_metrics", MODULE_PATH)
fine_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["build_same_fine_metrics"] = fine_metrics
SPEC.loader.exec_module(fine_metrics)


class BuildSameFineMetricsTests(unittest.TestCase):
    def test_builds_common_goal_region_and_csv_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_dir = tmp / "experiment_outputs" / "0001_same_demo"
            eval_items_dir = experiment_dir / "eval_items"
            connectivity_dir = tmp / "connectivity"
            output_dir = tmp / "fine_metrics"
            eval_items_dir.mkdir(parents=True)
            connectivity_dir.mkdir()

            write_connectivity(connectivity_dir / "scan1_connectivity.json")
            write_context(eval_items_dir, "CVDN", "val_unseen", "CVDN_val_unseen_eval_items.jsonl")
            write_jsonl(
                eval_items_dir / "CVDN_val_unseen_eval_items.jsonl",
                [
                    minimal_eval_item(
                        dataset="CVDN",
                        internal_item_id="cvdn_1",
                        raw_action_steps=None,
                        trajectory=["A", "B", "C"],
                        pred_path_segments=[["A"], ["B"], ["C"]],
                        region_extras={"cvdn": {"end_panos": ["B"]}},
                    )
                ],
            )

            manifest = fine_metrics.build_fine_metrics(experiment_dir, connectivity_dir, output_dir)

            self.assertEqual(manifest["counts"]["items"], 1)
            rows = read_jsonl(output_dir / "jsonl" / "CVDN_val_unseen_fine_metrics.jsonl")
            self.assertEqual(len(rows), 1)
            row = rows[0]

            self.assertEqual(row["common"]["action_step_count"], 2)
            self.assertEqual(row["common"]["move_step_count"], 2)
            self.assertEqual(row["common"]["path_edge_count"], 2)
            self.assertEqual(row["common"]["instruction_token_count"], 7)
            self.assertAlmostEqual(row["common"]["path_length_m"], 2.0)

            goal = row["eval_end_goal"]
            self.assertTrue(goal["final_success"])
            self.assertTrue(goal["oracle_success"])
            self.assertAlmostEqual(goal["final_distance_to_goal_m"], 1.0)
            self.assertEqual(goal["final_distance_to_goal_edges"], 1)
            self.assertAlmostEqual(goal["path_length_ratio"], 2.0 / 3.0)
            self.assertAlmostEqual(goal["oracle_path_length_m"], 1.0)
            self.assertEqual(goal["oracle_path_edge_count"], 1)
            self.assertAlmostEqual(goal["oracle_path_length_ratio"], 1.0 / 3.0)
            self.assertAlmostEqual(goal["shortest_path_length_m"], 3.0)
            self.assertEqual(goal["shortest_path_edge_count"], 3)

            region = row["eval_end_region"]
            self.assertFalse(region["final_success"])
            self.assertTrue(region["oracle_success"])
            self.assertAlmostEqual(region["final_distance_to_goal_m"], 1.0)
            self.assertEqual(region["final_distance_to_goal_edges"], 1)
            self.assertAlmostEqual(region["shortest_path_length_m"], 1.0)
            self.assertEqual(region["shortest_path_edge_count"], 1)
            self.assertAlmostEqual(region["oracle_path_length_m"], 1.0)
            self.assertEqual(region["oracle_path_edge_count"], 1)

            wide_rows = read_csv(output_dir / "tables" / "fine_metrics_wide.csv")
            self.assertEqual(wide_rows[0]["common.action_step_count"], "2")
            self.assertEqual(wide_rows[0]["eval_end_goal.final_success"], "true")
            self.assertEqual(wide_rows[0]["eval_end_region.final_success"], "false")

            long_rows = read_csv(output_dir / "tables" / "fine_metrics_long.csv")
            self.assertIn(
                {
                    "experiment_id": "0001_same_demo",
                    "dataset": "CVDN",
                    "split": "val_unseen",
                    "internal_item_id": "cvdn_1",
                    "metric_group": "common",
                    "metric_name": "action_step_count",
                    "value_num": "2",
                    "value_bool": "",
                    "value_type": "num",
                },
                long_rows,
            )
            self.assertIn(
                {
                    "experiment_id": "0001_same_demo",
                    "dataset": "CVDN",
                    "split": "val_unseen",
                    "internal_item_id": "cvdn_1",
                    "metric_group": "eval_end_region",
                    "metric_name": "final_success",
                    "value_num": "",
                    "value_bool": "false",
                    "value_type": "bool",
                },
                long_rows,
            )

    def test_uses_raw_action_steps_when_present_and_skips_r2r_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_dir = tmp / "experiment_outputs" / "0002_same_demo"
            eval_items_dir = experiment_dir / "eval_items"
            connectivity_dir = tmp / "connectivity"
            output_dir = tmp / "fine_metrics"
            eval_items_dir.mkdir(parents=True)
            connectivity_dir.mkdir()

            write_connectivity(connectivity_dir / "scan1_connectivity.json")
            write_context(eval_items_dir, "R2R", "val_unseen", "R2R_val_unseen_eval_items.jsonl")
            write_jsonl(
                eval_items_dir / "R2R_val_unseen_eval_items.jsonl",
                [
                    minimal_eval_item(
                        dataset="R2R",
                        internal_item_id="r2r_1",
                        raw_action_steps=8,
                        trajectory=["A", "A", "B"],
                        pred_path_segments=[["A"], ["A"], ["B"]],
                        region_extras={},
                    )
                ],
            )

            fine_metrics.build_fine_metrics(experiment_dir, connectivity_dir, output_dir)

            row = read_jsonl(output_dir / "jsonl" / "R2R_val_unseen_fine_metrics.jsonl")[0]
            self.assertEqual(row["common"]["action_step_count"], 8)
            self.assertEqual(row["common"]["move_step_count"], 1)
            self.assertIsNone(row["eval_end_region"])

            wide_row = read_csv(output_dir / "tables" / "fine_metrics_wide.csv")[0]
            self.assertEqual(wide_row["eval_end_region.final_success"], "")

    def test_real_0011_smoke_if_available(self) -> None:
        experiment_dir = (
            REPO_ROOT
            / "experiment_outputs"
            / "0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1"
        )
        connectivity_dir = REPO_ROOT / "data" / "same" / "simulator" / "connectivity"
        if not experiment_dir.exists() or not connectivity_dir.exists():
            self.skipTest("0011 experiment output or connectivity data is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "fine_metrics"
            manifest = fine_metrics.build_fine_metrics(experiment_dir, connectivity_dir, output_dir)
            expected_items = sum(
                count_lines(path)
                for path in (experiment_dir / "eval_items").glob("*_eval_items.jsonl")
            )
            self.assertEqual(manifest["counts"]["items"], expected_items)

            r2r_rows = read_jsonl(output_dir / "jsonl" / "R2R_val_unseen_fine_metrics.jsonl")
            self.assertGreater(len(r2r_rows), 0)
            self.assertIsNone(r2r_rows[0]["eval_end_region"])
            self.assertAlmostEqual(
                r2r_rows[0]["eval_end_goal"]["final_distance_to_goal_m"],
                5.163127264153355,
            )


def write_context(eval_items_dir: Path, dataset: str, split: str, items_name: str) -> None:
    payload = {
        "schema_version": "eval_context.v1",
        "items_schema_version": "eval_items.v2",
        "run_context": {
            "framework": "SAME",
            "experiment_id": "demo",
            "dataset": dataset,
            "split": split,
            "simulation_env": "mattersim",
            "success_threshold_m": 3.0,
        },
        "files": {
            "official_results": f"{dataset}_{split}_results.json",
            "eval_items": items_name,
        },
    }
    (eval_items_dir / f"{dataset}_{split}_eval_context.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def minimal_eval_item(
    dataset: str,
    internal_item_id: str,
    raw_action_steps: int | None,
    trajectory: list[str],
    pred_path_segments: list[list[str]],
    region_extras: dict[str, object],
) -> dict[str, object]:
    raw_same = {
        "trajectory_lengths": 2.0,
        "success": 1.0,
    }
    if raw_action_steps is not None:
        raw_same["action_steps"] = raw_action_steps
    return {
        "schema_version": "eval_items.v2",
        "identity": {
            "internal_item_id": internal_item_id,
            "saved_instr_id": internal_item_id,
            "source_ids": {"path_id": 1},
        },
        "annotation": {
            "scan": "scan1",
            "instruction": "go",
            "instruction_meta": {"encoding_len": 7},
            "start_viewpoint": "A",
            "nav_goal_viewpoint": "D",
            "success_target_viewpoints": ["B"],
            "gt_path": ["A", "B", "C", "D"],
        },
        "prediction": {
            "pred_path_segments": pred_path_segments,
            "trajectory": trajectory,
        },
        "primitives": {
            "final_viewpoint": trajectory[-1],
            "trajectory_cumulative_lengths_m": [0.0, 1.0, 2.0],
            "distance_to_nav_goal_by_step_m": [3.0, 2.0, 1.0],
            "shortest_start_to_nav_goal_distance_m": 3.0,
            "trajectory_edge_lengths_m": [1.0, 1.0],
        },
        "official_item_scores": {
            "raw_same": raw_same,
            "canonical": {
                "actual_length_m": 2.0,
                "shortest_path_length_m": 3.0,
            },
        },
        "dataset_extras": region_extras,
    }


def write_connectivity(path: Path) -> None:
    nodes = {
        "A": (0.0, 0.0, 0.0),
        "B": (1.0, 0.0, 0.0),
        "C": (2.0, 0.0, 0.0),
        "D": (3.0, 0.0, 0.0),
    }
    order = list(nodes)
    edge_set = {tuple(sorted(edge)) for edge in [("A", "B"), ("B", "C"), ("C", "D")]}
    data = []
    for source in order:
        x, y, z = nodes[source]
        unobstructed = []
        for target in order:
            unobstructed.append(tuple(sorted((source, target))) in edge_set)
        pose = [0.0] * 16
        pose[3], pose[7], pose[11] = x, y, z
        data.append(
            {
                "image_id": source,
                "pose": pose,
                "included": True,
                "unobstructed": unobstructed,
            }
        )
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


if __name__ == "__main__":
    unittest.main()
