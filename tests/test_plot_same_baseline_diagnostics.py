from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "plts" / "plot_same_baseline_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("plot_same_baseline_diagnostics", MODULE_PATH)
plot_same = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = plot_same
SPEC.loader.exec_module(plot_same)


class PlotSameBaselineDiagnosticsTests(unittest.TestCase):
    def test_load_experiment_metrics_pivots_long_format_and_handles_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics_long.csv"
            rows = [
                {
                    "experiment_id": "exp-1",
                    "dataset": "R2R",
                    "split": "val_unseen",
                    "metric": "SR",
                    "value": "76.29",
                    "unit": "%",
                },
                {
                    "experiment_id": "exp-1",
                    "dataset": "R2R",
                    "split": "val_unseen",
                    "metric": "oracle_success",
                    "value": "84.80",
                    "unit": "%",
                },
                {
                    "experiment_id": "exp-1",
                    "dataset": "R2R",
                    "split": "val_unseen",
                    "metric": "NE",
                    "value": "2.70",
                    "unit": "m",
                },
                {
                    "experiment_id": "exp-1",
                    "dataset": "R2R",
                    "split": "val_unseen",
                    "metric": "nav_error",
                    "value": "2.80",
                    "unit": "m",
                },
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["experiment_id", "dataset", "split", "metric", "value", "unit"],
                )
                writer.writeheader()
                writer.writerows(rows)

            metrics_by_key, warnings = plot_same.load_experiment_metrics(csv_path, "exp-1")

        self.assertAlmostEqual(metrics_by_key[("R2R", "val_unseen")]["sr"], 76.29)
        self.assertAlmostEqual(metrics_by_key[("R2R", "val_unseen")]["oracle_sr"], 84.80)
        self.assertAlmostEqual(metrics_by_key[("R2R", "val_unseen")]["nav_error"], 2.75)
        self.assertEqual(len(warnings), 1)
        self.assertIn("duplicate metric rows", warnings[0])

    def test_build_seen_unseen_gap_records_skips_incomplete_datasets(self) -> None:
        metrics_by_key = {
            ("R2R", "val_train_seen"): {"sr": 88.74, "spl": 85.01},
            ("R2R", "val_unseen"): {"sr": 76.29, "spl": 66.24},
            ("SOON", "val_unseen"): {"sr": 36.34, "spl": 25.66},
        }

        records, skipped = plot_same.build_seen_unseen_gap_records(metrics_by_key)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "R2R")
        self.assertAlmostEqual(records[0][1], 12.45)
        self.assertAlmostEqual(records[0][2], 18.77)
        self.assertEqual(skipped, ["SOON (requires both seen and unseen splits)"])

    def test_build_error_pair_records_skips_missing_oracle_error(self) -> None:
        ordered_keys = [
            ("R2R", "val_unseen"),
            ("CVDN", "val_unseen"),
        ]
        labels = {
            ("R2R", "val_unseen"): "R2R-unseen",
            ("CVDN", "val_unseen"): "CVDN-unseen",
        }
        metrics_by_key = {
            ("R2R", "val_unseen"): {"nav_error": 2.72, "oracle_error": 1.40},
            ("CVDN", "val_unseen"): {"nav_error": 12.94},
        }

        records, skipped = plot_same.build_error_pair_records(metrics_by_key, ordered_keys, labels)

        self.assertEqual(records, [("R2R-unseen", 2.72, 1.40)])
        self.assertEqual(skipped, ["CVDN-unseen (missing oracle_error)"])

    def test_build_plot_path_includes_experiment_id(self) -> None:
        output_path = plot_same.build_plot_path(Path("/tmp/plots"), "0005_same_demo_v1", "osr_sr_gap_bar")
        self.assertEqual(output_path.name, "0005_same_demo_v1_osr_sr_gap_bar.png")


if __name__ == "__main__":
    unittest.main()
