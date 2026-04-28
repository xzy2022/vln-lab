from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "build_oracle_gap_report.py"
SPEC = importlib.util.spec_from_file_location("build_oracle_gap_report", MODULE_PATH)
oracle_gap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = oracle_gap
SPEC.loader.exec_module(oracle_gap)


class BuildOracleGapReportTests(unittest.TestCase):
    def test_item_row_splits_first_success_and_best_distance_steps(self) -> None:
        source = oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_unseen",
            context_path=Path("/tmp/R2R_val_unseen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_unseen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = minimal_item(distances=[5.0, 2.5, 1.0, 4.0], cumulative=[0.0, 1.0, 2.0, 3.0])
        fine_row = {"official.shortest_path_length_m": "2.0"}
        target_scope = oracle_gap.build_target_scope(item, "R2R", "official")

        row = oracle_gap.build_item_row(item, source, fine_row, target_scope)

        self.assertEqual(row["first_success_step"], 1)
        self.assertEqual(row["best_distance_step"], 2)
        self.assertFalse(row["final_success"])
        self.assertTrue(row["oracle_success"])
        self.assertTrue(row["nearest_endpoint_success"])
        self.assertTrue(row["overshoot"])
        self.assertFalse(row["stop_too_early_proxy"])
        self.assertAlmostEqual(row["first_success_oracle_spl"], 1.0)
        self.assertAlmostEqual(row["nearest_endpoint_spl"], 1.0)

    def test_stop_too_early_proxy_requires_final_step_to_be_best_distance(self) -> None:
        source = oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_unseen",
            context_path=Path("/tmp/R2R_val_unseen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_unseen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = minimal_item(distances=[7.0, 5.0, 4.0], cumulative=[0.0, 1.0, 2.0])
        fine_row = {"official.shortest_path_length_m": "6.0"}
        target_scope = oracle_gap.build_target_scope(item, "R2R", "official")

        row = oracle_gap.build_item_row(item, source, fine_row, target_scope)

        self.assertIsNone(row["first_success_step"])
        self.assertEqual(row["best_distance_step"], 2)
        self.assertFalse(row["final_success"])
        self.assertFalse(row["oracle_success"])
        self.assertFalse(row["nearest_endpoint_success"])
        self.assertFalse(row["overshoot"])
        self.assertTrue(row["stop_too_early_proxy"])
        self.assertFalse(row["never_reached"])

    def test_summary_rates_are_fractions_not_percent_points(self) -> None:
        rows = [
            {
                "experiment_id": "exp",
                "dataset": "R2R",
                "split": "val_unseen",
                "target_scope": "official",
                "final_success": False,
                "oracle_success": True,
                "nearest_endpoint_success": True,
                "recovered_by_nearest_endpoint": True,
                "overshoot": True,
                "stop_too_early_proxy": False,
                "never_reached": False,
                "final_spl": 0.0,
                "first_success_oracle_spl": 1.0,
                "nearest_endpoint_spl": 1.0,
            },
            {
                "experiment_id": "exp",
                "dataset": "R2R",
                "split": "val_unseen",
                "target_scope": "official",
                "final_success": False,
                "oracle_success": False,
                "nearest_endpoint_success": False,
                "recovered_by_nearest_endpoint": False,
                "overshoot": False,
                "stop_too_early_proxy": True,
                "never_reached": False,
                "final_spl": 0.0,
                "first_success_oracle_spl": 0.0,
                "nearest_endpoint_spl": 0.0,
            },
        ]

        summary = oracle_gap.build_summary_rows(rows)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertAlmostEqual(row["final_success_rate"], 0.0)
        self.assertAlmostEqual(row["oracle_success_rate"], 0.5)
        self.assertAlmostEqual(row["oracle_gap_rate"], 0.5)
        self.assertAlmostEqual(row["nearest_endpoint_success_rate"], 0.5)
        self.assertAlmostEqual(row["overshoot_rate"], 0.5)
        self.assertAlmostEqual(row["stop_too_early_proxy_rate"], 0.5)
        self.assertAlmostEqual(row["nearest_endpoint_spl"], 0.5)


def minimal_item(distances: list[float], cumulative: list[float]) -> dict:
    trajectory = [f"vp_{index}" for index in range(len(distances))]
    return {
        "identity": {
            "internal_item_id": "r2r_1_0",
            "saved_instr_id": "1_0",
        },
        "annotation": {
            "success_target_viewpoints": ["vp_1"],
        },
        "prediction": {
            "trajectory": trajectory,
        },
        "primitives": {
            "distance_to_nav_goal_by_step_m": distances,
            "trajectory_cumulative_lengths_m": cumulative,
        },
    }


if __name__ == "__main__":
    unittest.main()
