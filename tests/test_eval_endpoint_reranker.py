from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "eval_endpoint_reranker.py"
SPEC = importlib.util.spec_from_file_location("eval_endpoint_reranker", MODULE_PATH)
reranker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reranker
SPEC.loader.exec_module(reranker)


class EvalEndpointRerankerTests(unittest.TestCase):
    def test_gate_and_tau_recover_or_preserve_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_csv = root / "endpoint_candidates.csv"
            episode_csv = root / "episode_groups.csv"
            score_csv = root / "scores.csv"
            output_dir = root / "out"

            write_fixture_csvs(candidate_csv, episode_csv, score_csv)

            manifest = reranker.evaluate_endpoint_reranker(
                candidate_csv=candidate_csv,
                episode_csv=episode_csv,
                score_csv=score_csv,
                output_dir=output_dir,
                split_filters=("dev",),
                gate_thresholds=(0.5,),
                taus=(0.0, 1.0),
                allow_change_final_values=(True,),
            )

            self.assertEqual(manifest["counts"]["episodes"], 2)
            summary_rows = reranker.read_csv(output_dir / "endpoint_learning_summary.csv")
            by_tau = {float(row["tau"]): row for row in summary_rows}

            self.assertAlmostEqual(float(by_tau[0.0]["recovery_rate"]), 0.5)
            self.assertAlmostEqual(float(by_tau[0.0]["harm_rate"]), 0.0)
            self.assertAlmostEqual(float(by_tau[0.0]["gate_precision"]), 1.0)
            self.assertAlmostEqual(float(by_tau[0.0]["gate_recall"]), 1.0)
            self.assertAlmostEqual(float(by_tau[0.0]["gate_auc"]), 1.0)

            self.assertAlmostEqual(float(by_tau[1.0]["recovery_rate"]), 0.0)
            self.assertAlmostEqual(float(by_tau[1.0]["harm_rate"]), 0.0)
            self.assertAlmostEqual(float(by_tau[1.0]["changed_endpoint_rate"]), 0.0)

    def test_allow_change_final_false_is_frozen_final_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_csv = root / "endpoint_candidates.csv"
            episode_csv = root / "episode_groups.csv"
            score_csv = root / "scores.csv"
            output_dir = root / "out"

            write_fixture_csvs(candidate_csv, episode_csv, score_csv)

            reranker.evaluate_endpoint_reranker(
                candidate_csv=candidate_csv,
                episode_csv=episode_csv,
                score_csv=score_csv,
                output_dir=output_dir,
                split_filters=("dev",),
                gate_thresholds=(0.5,),
                taus=(0.0,),
                allow_change_final_values=(False,),
            )

            summary = reranker.read_csv(output_dir / "endpoint_learning_summary.csv")[0]
            items = reranker.read_csv(output_dir / "endpoint_learning_items.csv")

            self.assertAlmostEqual(float(summary["changed_endpoint_rate"]), 0.0)
            self.assertAlmostEqual(float(summary["recovery_rate"]), 0.0)
            self.assertAlmostEqual(float(summary["harm_rate"]), 0.0)
            self.assertEqual({row["selection_reason"] for row in items}, {"change_disabled"})


def write_fixture_csvs(candidate_csv: Path, episode_csv: Path, score_csv: Path) -> None:
    episode_rows = [
        episode_row("episode_pos", final_success=False, oracle_success=True, final_spl=0.0),
        episode_row("episode_neg", final_success=True, oracle_success=True, final_spl=1.0),
    ]
    candidate_rows = [
        candidate_row("episode_pos", 0, is_final=False, success=True, spl=1.0, distance=1.0),
        candidate_row("episode_pos", 1, is_final=True, success=False, spl=0.0, distance=5.0),
        candidate_row("episode_neg", 0, is_final=False, success=False, spl=0.0, distance=5.0),
        candidate_row("episode_neg", 1, is_final=True, success=True, spl=1.0, distance=1.0),
    ]
    score_rows = [
        {
            "candidate_id": "episode_pos:step_000",
            "candidate_score": "0.9",
            "gate_score": "0.8",
        },
        {
            "candidate_id": "episode_pos:step_001",
            "candidate_score": "0.1",
            "gate_score": "0.8",
        },
        {
            "candidate_id": "episode_neg:step_000",
            "candidate_score": "0.9",
            "gate_score": "0.2",
        },
        {
            "candidate_id": "episode_neg:step_001",
            "candidate_score": "0.1",
            "gate_score": "0.2",
        },
    ]
    reranker.write_csv(episode_csv, episode_fieldnames(), episode_rows)
    reranker.write_csv(candidate_csv, candidate_fieldnames(), candidate_rows)
    reranker.write_csv(score_csv, ["candidate_id", "candidate_score", "gate_score"], score_rows)


def episode_row(
    episode_id: str,
    final_success: bool,
    oracle_success: bool,
    final_spl: float,
) -> dict:
    return {
        "experiment_id": "exp",
        "dataset": "R2R",
        "split": "val_train_seen",
        "protocol_split": "dev",
        "target_scope": "official",
        "episode_id": episode_id,
        "internal_item_id": episode_id,
        "saved_instr_id": episode_id,
        "final_success": final_success,
        "oracle_success": oracle_success,
        "nearest_endpoint_success": oracle_success,
        "should_rerank": (not final_success) and oracle_success,
        "final_failure_bucket": "overshoot" if not final_success and oracle_success else "final_success",
        "final_spl": final_spl,
        "nearest_endpoint_spl": 1.0,
        "final_distance_m": 5.0 if not final_success else 1.0,
        "final_path_length_m": 2.0,
    }


def candidate_row(
    episode_id: str,
    step: int,
    is_final: bool,
    success: bool,
    spl: float,
    distance: float,
) -> dict:
    return {
        "experiment_id": "exp",
        "dataset": "R2R",
        "split": "val_train_seen",
        "protocol_split": "dev",
        "target_scope": "official",
        "episode_id": episode_id,
        "candidate_id": f"{episode_id}:step_{step:03d}",
        "internal_item_id": episode_id,
        "saved_instr_id": episode_id,
        "candidate_step": step,
        "viewpoint": f"vp_{episode_id}_{step}",
        "is_final": is_final,
        "success_label": success,
        "spl_at_candidate": spl,
        "distance_to_goal_m": distance,
        "path_length_m": float(step + 1),
    }


def episode_fieldnames() -> list[str]:
    return [
        "experiment_id",
        "dataset",
        "split",
        "protocol_split",
        "target_scope",
        "episode_id",
        "internal_item_id",
        "saved_instr_id",
        "final_success",
        "oracle_success",
        "nearest_endpoint_success",
        "should_rerank",
        "final_failure_bucket",
        "final_spl",
        "nearest_endpoint_spl",
        "final_distance_m",
        "final_path_length_m",
    ]


def candidate_fieldnames() -> list[str]:
    return [
        "experiment_id",
        "dataset",
        "split",
        "protocol_split",
        "target_scope",
        "episode_id",
        "candidate_id",
        "internal_item_id",
        "saved_instr_id",
        "candidate_step",
        "viewpoint",
        "is_final",
        "success_label",
        "spl_at_candidate",
        "distance_to_goal_m",
        "path_length_m",
    ]


if __name__ == "__main__":
    unittest.main()
