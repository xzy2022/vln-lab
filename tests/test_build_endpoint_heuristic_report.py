from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "build_endpoint_heuristic_report.py"
SPEC = importlib.util.spec_from_file_location("build_endpoint_heuristic_report", MODULE_PATH)
endpoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = endpoint
SPEC.loader.exec_module(endpoint)


class BuildEndpointHeuristicReportTests(unittest.TestCase):
    def test_last_k_max_stop_uses_trace_and_selects_highest_recent_stop(self) -> None:
        episode = endpoint.build_sanitized_episode(
            minimal_item(
                stop_probs=[0.01, 0.7, 0.2, 0.4, 0.3],
                distances=[7.0, 2.0, 5.0, 6.0, 8.0],
                cumulative=[0.0, 1.0, 2.0, 3.0, 4.0],
            )
        )

        selection = endpoint.select_last_k_max_stop(episode, k_value=3)

        self.assertEqual(selection.step, 3)
        self.assertEqual(selection.reason, "last_3_max_stop_prob")

    def test_item_row_reports_recovery_when_heuristic_selects_successful_endpoint(self) -> None:
        source = endpoint.oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_unseen",
            context_path=Path("/tmp/R2R_val_unseen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_unseen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = minimal_item(
            stop_probs=[0.01, 0.7, 0.2, 0.4],
            distances=[7.0, 2.0, 5.0, 8.0],
            cumulative=[0.0, 1.0, 2.0, 3.0],
        )
        target_scope = endpoint.oracle_gap.build_target_scope(item, "R2R", "official")
        fine_row = {"official.shortest_path_length_m": "1.0"}
        episode = endpoint.build_sanitized_episode(item)
        eval_context = endpoint.build_eval_context(item, source, fine_row, target_scope)
        heuristic = endpoint.HeuristicSpec("max_stop_prob", "", endpoint.select_max_stop_prob)

        row = endpoint.build_item_row(
            item=item,
            source=source,
            target_scope=target_scope,
            eval_context=eval_context,
            episode=episode,
            heuristic=heuristic,
            selection=endpoint.Selection(1, "max_stop_prob"),
        )

        self.assertFalse(row["final_success"])
        self.assertTrue(row["heuristic_success"])
        self.assertTrue(row["oracle_success"])
        self.assertEqual(row["final_failure_bucket"], "overshoot")
        self.assertTrue(row["recovered_by_heuristic"])
        self.assertFalse(row["harmed_by_heuristic"])
        self.assertAlmostEqual(row["heuristic_spl"], 1.0)

    def test_summary_computes_gap_capture_recovery_and_harm_rates(self) -> None:
        rows = [
            summary_item(final=False, heur=True, nearest=True, bucket="overshoot"),
            summary_item(final=True, heur=False, nearest=True, bucket="final_success"),
            summary_item(final=True, heur=True, nearest=True, bucket="final_success"),
            summary_item(final=False, heur=False, nearest=False, bucket="never_reached"),
        ]

        summary = endpoint.build_summary_rows(rows)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertAlmostEqual(row["final_success_rate"], 0.5)
        self.assertAlmostEqual(row["heuristic_success_rate"], 0.5)
        self.assertAlmostEqual(row["nearest_endpoint_success_rate"], 0.75)
        self.assertAlmostEqual(row["gap_capture_rate"], 0.0)
        self.assertAlmostEqual(row["recovery_rate"], 0.25)
        self.assertAlmostEqual(row["harm_rate"], 0.25)
        self.assertAlmostEqual(row["overshoot_recovery_rate"], 1.0)
        self.assertEqual(row["never_reached_items"], 1)


def minimal_item(stop_probs: list[float], distances: list[float], cumulative: list[float]) -> dict:
    trajectory = [f"vp_{index}" for index in range(len(stop_probs))]
    steps = [
        {
            "step": index,
            "current_viewpoint": trajectory[index],
            "fusion": "dynamic",
            "stop_prob": stop_prob,
            "selected": {
                "prob": max(0.0, 1.0 - stop_prob),
                "selection_kind": "stop" if index == len(stop_probs) - 1 else "current_visible",
            },
            "gmap_candidates": [
                {
                    "index": 0,
                    "viewpoint": None,
                    "valid": True,
                    "actionable": True,
                    "fused": {"logit": stop_prob * 10.0, "prob": stop_prob},
                },
                {
                    "index": 1,
                    "viewpoint": f"next_{index}",
                    "valid": True,
                    "actionable": True,
                    "fused": {"logit": 1.0, "prob": max(0.0, 1.0 - stop_prob)},
                },
            ],
            "route_viewpoints": [trajectory[index]],
            "moe": {"router_entropy": 1.0 + index},
        }
        for index, stop_prob in enumerate(stop_probs)
    ]
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
            "decision_trace": {"steps": steps},
        },
        "primitives": {
            "distance_to_nav_goal_by_step_m": distances,
            "trajectory_cumulative_lengths_m": cumulative,
        },
    }


def summary_item(final: bool, heur: bool, nearest: bool, bucket: str) -> dict:
    return {
        "experiment_id": "exp",
        "dataset": "R2R",
        "split": "val_unseen",
        "target_scope": "official",
        "heuristic": "h",
        "heuristic_params": "",
        "trace_available": True,
        "selected_changed": final != heur,
        "final_success": final,
        "heuristic_success": heur,
        "nearest_endpoint_success": nearest,
        "recovered_by_heuristic": (not final) and heur,
        "harmed_by_heuristic": final and (not heur),
        "final_failure_bucket": bucket,
        "final_spl": 1.0 if final else 0.0,
        "heuristic_spl": 1.0 if heur else 0.0,
        "nearest_endpoint_spl": 1.0 if nearest else 0.0,
        "selected_step_delta": -1.0 if final != heur else 0.0,
        "final_stop_prob": 0.4,
        "selected_stop_prob": 0.5,
    }


if __name__ == "__main__":
    unittest.main()
