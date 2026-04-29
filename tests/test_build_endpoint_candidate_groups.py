from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "build_endpoint_candidate_groups.py"
SPEC = importlib.util.spec_from_file_location("build_endpoint_candidate_groups", MODULE_PATH)
candidate_groups = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = candidate_groups
SPEC.loader.exec_module(candidate_groups)


class BuildEndpointCandidateGroupsTests(unittest.TestCase):
    def test_step_level_candidates_keep_duplicate_viewpoints_and_mark_rerank_labels(self) -> None:
        source = candidate_groups.oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_train_seen",
            context_path=Path("/tmp/R2R_val_train_seen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_train_seen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = minimal_item(
            trajectory=["vp_0", "vp_1", "vp_0", "vp_2"],
            stop_probs=[0.1, 0.8, 0.2, 0.3],
            distances=[5.0, 2.0, 4.0, 6.0],
            cumulative=[0.0, 1.0, 2.0, 3.0],
        )
        target_scope = candidate_groups.oracle_gap.build_target_scope(item, "R2R", "official")
        artifacts = candidate_groups.build_episode_artifacts(
            item=item,
            source=source,
            fine_row={"official.shortest_path_length_m": "1.0"},
            target_scope=target_scope,
            protocol_split="train",
            last_k=2,
            loop_window=5,
        )

        self.assertEqual(len(artifacts.candidate_rows), 4)
        self.assertEqual([row["viewpoint"] for row in artifacts.candidate_rows], ["vp_0", "vp_1", "vp_0", "vp_2"])
        self.assertTrue(artifacts.candidate_rows[2]["is_revisit"])
        self.assertTrue(artifacts.candidate_rows[0]["is_loop_region"])
        self.assertFalse(artifacts.candidate_rows[1]["is_last_k"])
        self.assertTrue(artifacts.candidate_rows[2]["is_last_k"])
        self.assertTrue(artifacts.group_row["should_rerank"])
        self.assertEqual(artifacts.group_row["final_failure_bucket"], "overshoot")
        self.assertTrue(artifacts.candidate_rows[1]["success_label"])
        self.assertTrue(artifacts.candidate_rows[1]["is_best_success_candidate"])
        self.assertEqual(len(artifacts.pair_rows), 3)
        self.assertEqual({row["pair_type"] for row in artifacts.pair_rows}, {"success_gt_fail"})

    def test_pairs_include_success_fail_better_spl_and_final_stay_classes(self) -> None:
        source = candidate_groups.oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_train_seen",
            context_path=Path("/tmp/R2R_val_train_seen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_train_seen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = minimal_item(
            trajectory=["vp_0", "vp_1", "vp_2", "vp_3"],
            stop_probs=[0.1, 0.4, 0.5, 0.9],
            distances=[5.0, 2.0, 2.0, 1.0],
            cumulative=[0.0, 4.0, 5.0, 6.0],
        )
        target_scope = candidate_groups.oracle_gap.build_target_scope(item, "R2R", "official")
        artifacts = candidate_groups.build_episode_artifacts(
            item=item,
            source=source,
            fine_row={"official.shortest_path_length_m": "4.0", "official.spl": "0.6666666667"},
            target_scope=target_scope,
            protocol_split="dev",
        )

        pair_type_counts = {
            pair_type: sum(1 for row in artifacts.pair_rows if row["pair_type"] == pair_type)
            for pair_type in candidate_groups.PAIR_TYPES
        }
        self.assertFalse(artifacts.group_row["should_rerank"])
        self.assertEqual(pair_type_counts["success_gt_fail"], 3)
        self.assertEqual(pair_type_counts["better_spl_success_gt_lower_spl_success"], 3)
        self.assertEqual(pair_type_counts["final_success_final_gt_failed_nonfinal"], 1)
        final_stay_pair = [
            row for row in artifacts.pair_rows
            if row["pair_type"] == "final_success_final_gt_failed_nonfinal"
        ][0]
        self.assertTrue(final_stay_pair["chosen_is_final"])
        self.assertFalse(final_stay_pair["rejected_success_label"])

    def test_trace_alignment_uses_route_viewpoints_instead_of_raw_index(self) -> None:
        source = candidate_groups.oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_train_seen",
            context_path=Path("/tmp/R2R_val_train_seen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_train_seen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = {
            "identity": {
                "internal_item_id": "r2r_2_0",
                "saved_instr_id": "2_0",
            },
            "annotation": {
                "success_target_viewpoints": ["vp_c"],
            },
            "prediction": {
                "trajectory": ["vp_a", "vp_b", "vp_c", "vp_d"],
                "decision_trace": {
                    "steps": [
                        trace_step("vp_a", 0.1, ["vp_a", "vp_b", "vp_c"]),
                        trace_step("vp_c", 0.8, ["vp_c", "vp_d"]),
                        trace_step("vp_d", 0.6, ["vp_d"]),
                    ],
                },
            },
            "primitives": {
                "distance_to_nav_goal_by_step_m": [5.0, 4.0, 2.0, 6.0],
                "trajectory_cumulative_lengths_m": [0.0, 1.0, 2.0, 3.0],
            },
        }
        target_scope = candidate_groups.oracle_gap.build_target_scope(item, "R2R", "official")

        artifacts = candidate_groups.build_episode_artifacts(
            item=item,
            source=source,
            fine_row={"official.shortest_path_length_m": "2.0"},
            target_scope=target_scope,
            protocol_split="train",
        )
        rows = artifacts.candidate_rows

        self.assertEqual([row["viewpoint"] for row in rows], ["vp_a", "vp_b", "vp_c", "vp_d"])
        self.assertEqual([row["decision_trace_index"] for row in rows], [0, None, 1, 2])
        self.assertEqual([row["stop_prob"] for row in rows], [0.1, None, 0.8, 0.6])
        self.assertEqual([row["has_decision_trace"] for row in rows], [True, False, True, True])
        self.assertEqual([row["is_route_intermediate"] for row in rows], [False, True, False, False])
        self.assertEqual([row["is_route_expanded_without_decision"] for row in rows], [False, True, False, False])
        self.assertEqual(artifacts.group_row["decision_trace_step_count"], 3)
        self.assertEqual(artifacts.group_row["candidates_with_decision_trace"], 3)
        self.assertEqual(artifacts.group_row["route_intermediate_candidates"], 1)
        self.assertEqual(artifacts.group_row["route_expanded_without_decision_candidates"], 1)

    def test_final_route_endpoint_without_next_decision_gets_expanded_mask(self) -> None:
        source = candidate_groups.oracle_gap.EvalItemSource(
            dataset="R2R",
            split="val_train_seen",
            context_path=Path("/tmp/R2R_val_train_seen_eval_context.json"),
            items_path=Path("/tmp/exp/eval_items/R2R_val_train_seen_eval_items.jsonl"),
            success_threshold_m=3.0,
        )
        item = {
            "identity": {
                "internal_item_id": "r2r_3_0",
                "saved_instr_id": "3_0",
            },
            "annotation": {
                "success_target_viewpoints": ["vp_c"],
            },
            "prediction": {
                "trajectory": ["vp_a", "vp_b", "vp_c"],
                "decision_trace": {
                    "steps": [
                        trace_step("vp_a", 0.2, ["vp_a", "vp_b", "vp_c"]),
                    ],
                },
            },
            "primitives": {
                "distance_to_nav_goal_by_step_m": [5.0, 4.0, 2.0],
                "trajectory_cumulative_lengths_m": [0.0, 1.0, 2.0],
            },
        }
        target_scope = candidate_groups.oracle_gap.build_target_scope(item, "R2R", "official")

        artifacts = candidate_groups.build_episode_artifacts(
            item=item,
            source=source,
            fine_row={"official.shortest_path_length_m": "2.0"},
            target_scope=target_scope,
            protocol_split="train",
        )

        self.assertEqual([row["has_decision_trace"] for row in artifacts.candidate_rows], [True, False, False])
        self.assertEqual(
            [row["is_route_intermediate"] for row in artifacts.candidate_rows],
            [False, True, False],
        )
        self.assertEqual(
            [row["is_route_expanded_without_decision"] for row in artifacts.candidate_rows],
            [False, True, True],
        )

    def test_train_eval_alias_maps_to_protocol_train(self) -> None:
        protocol_split = candidate_groups.protocol_split_for(
            dataset="R2R",
            split="train_eval",
            internal_item_id="r2r_1_0",
            dev_ratio=0.2,
            salt="test",
        )

        self.assertEqual(protocol_split, "train")


def minimal_item(
    trajectory: list[str],
    stop_probs: list[float],
    distances: list[float],
    cumulative: list[float],
) -> dict:
    steps = [
        {
            "step": index,
            "current_viewpoint": trajectory[index],
            "fusion": "dynamic",
            "fuse_weight": 0.25 + index * 0.1,
            "stop_prob": stop_prob,
            "selected": {
                "prob": max(0.0, 1.0 - stop_prob),
                "selection_kind": "stop" if index == len(trajectory) - 1 else "current_visible",
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


def trace_step(current_viewpoint: str, stop_prob: float, route_viewpoints: list[str]) -> dict:
    return {
        "current_viewpoint": current_viewpoint,
        "fusion": "dynamic",
        "fuse_weight": 0.5,
        "stop_prob": stop_prob,
        "selected": {
            "prob": max(0.0, 1.0 - stop_prob),
            "selection_kind": "stop" if len(route_viewpoints) == 1 else "global_node",
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
                "viewpoint": route_viewpoints[-1],
                "valid": True,
                "actionable": True,
                "fused": {"logit": 1.0, "prob": max(0.0, 1.0 - stop_prob)},
            },
        ],
        "route_viewpoints": route_viewpoints,
        "moe": {"router_entropy": 1.0},
    }


if __name__ == "__main__":
    unittest.main()
