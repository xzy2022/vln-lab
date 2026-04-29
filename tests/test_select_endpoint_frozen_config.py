from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "select_endpoint_frozen_config.py"
SPEC = importlib.util.spec_from_file_location("select_endpoint_frozen_config", MODULE_PATH)
selector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


class SelectEndpointFrozenConfigTests(unittest.TestCase):
    def test_weighted_selection_uses_all_datasets_before_ranking(self) -> None:
        summary = selector.normalize_summary_frame(
            pd.DataFrame(
                [
                    summary_row("R2R", gate=0.5, items=100, final_sr=0.5, sr=0.7, harm=0.0),
                    summary_row("CVDN", gate=0.5, items=300, final_sr=0.5, sr=0.4, harm=0.0),
                    summary_row("R2R", gate=0.9, items=100, final_sr=0.5, sr=0.51, harm=0.0),
                    summary_row("CVDN", gate=0.9, items=300, final_sr=0.5, sr=0.51, harm=0.0),
                ]
            )
        )

        weighted = selector.build_selection_summary(
            summary,
            selection_aggregation="weighted",
            selection_split="dev",
        )
        grid, selected = selector.choose_config(
            summary=weighted,
            min_delta_sr=0.0,
            min_delta_spl=0.0,
            max_harm_rate=0.01,
            max_dataset_harm_rate=0.01,
            allow_frozen_final_fallback=False,
        )

        self.assertEqual(set(weighted["dataset"]), {"ALL"})
        self.assertEqual(len(grid), 2)
        self.assertEqual(selected["dataset"], "ALL")
        self.assertAlmostEqual(float(selected["gate_threshold"]), 0.9)
        self.assertAlmostEqual(float(selected["delta_SR"]), 0.01)

    def test_max_dataset_harm_guard_blocks_aggregate_safe_config(self) -> None:
        summary = selector.normalize_summary_frame(
            pd.DataFrame(
                [
                    summary_row("R2R", gate=0.5, items=100, final_sr=0.5, sr=0.56, harm=0.02),
                    summary_row("CVDN", gate=0.5, items=900, final_sr=0.5, sr=0.56, harm=0.0),
                    summary_row("R2R", gate=0.9, items=100, final_sr=0.5, sr=0.51, harm=0.005),
                    summary_row("CVDN", gate=0.9, items=900, final_sr=0.5, sr=0.51, harm=0.005),
                ]
            )
        )
        weighted = selector.build_selection_summary(
            summary,
            selection_aggregation="weighted",
            selection_split="dev",
        )

        _, selected = selector.choose_config(
            summary=weighted,
            min_delta_sr=0.0,
            min_delta_spl=0.0,
            max_harm_rate=0.01,
            max_dataset_harm_rate=0.01,
            allow_frozen_final_fallback=False,
        )

        self.assertAlmostEqual(float(selected["gate_threshold"]), 0.9)
        self.assertAlmostEqual(float(selected["max_dataset_harm_rate"]), 0.005)


def summary_row(
    dataset: str,
    gate: float,
    items: int,
    final_sr: float,
    sr: float,
    harm: float,
) -> dict:
    return {
        "experiment_id": "exp",
        "dataset": dataset,
        "split": "val_train_seen",
        "protocol_split": "dev",
        "target_scope": "official",
        "gate_threshold": gate,
        "tau": 0.0,
        "allow_change_final": True,
        "items": items,
        "final_SR": final_sr,
        "SR": sr,
        "delta_SR": sr - final_sr,
        "final_SPL": final_sr,
        "SPL": sr,
        "delta_SPL": sr - final_sr,
        "recovery_rate": max(sr - final_sr, 0.0) + harm,
        "harm_rate": harm,
        "net_recovery_rate": sr - final_sr,
        "changed_endpoint_rate": 0.1,
        "gate_pass_rate": 0.1,
        "gate_precision": 0.5,
        "gate_recall": 0.5,
        "overshoot_items": 10,
        "overshoot_recovery_rate": 0.1,
        "final_success_items": 10,
        "final_success_harm_rate": harm,
    }


if __name__ == "__main__":
    unittest.main()
