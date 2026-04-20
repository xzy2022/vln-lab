from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "experiments" / "run_same.py"
SPEC = importlib.util.spec_from_file_location("run_same", MODULE_PATH)
run_same = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_same)


class RunSameTests(unittest.TestCase):
    def test_allocate_experiment_id_increments_sequence_and_revision(self) -> None:
        experiment_id, experiment_slug = run_same.allocate_experiment_id(
            method_slug="same",
            config_stem="val-r2r-eval-only",
            checkpoint_tag="same",
            seed=0,
            tag=None,
            existing_ids={
                "legacy_experiment",
                "0007_same_val-r2r-eval-only_same_s0_v1",
                "0008_same_other_same_s0_v1",
            },
        )
        self.assertEqual(experiment_slug, "same_val-r2r-eval-only_same_s0")
        self.assertEqual(experiment_id, "0009_same_val-r2r-eval-only_same_s0_v2")

    def test_parse_metrics_from_eval_log(self) -> None:
        log_path = REPO_ROOT / "third_party" / "SAME" / "src" / "output" / "val-r2r-eval-only" / "val-r2r-eval-only.log"
        metrics = run_same.parse_metrics_from_log(log_path)
        self.assertAlmostEqual(metrics["R2R"]["val_unseen"]["sr"]["value"], 76.29)
        self.assertEqual(metrics["R2R"]["val_unseen"]["sr"]["unit"], "%")
        self.assertAlmostEqual(metrics["R2R"]["val_train_seen"]["lengths"]["value"], 11.18)
        self.assertEqual(metrics["R2R"]["val_train_seen"]["lengths"]["unit"], "m")

    def test_parse_metrics_from_multidataset_log(self) -> None:
        log_path = (
            REPO_ROOT
            / "third_party"
            / "SAME"
            / "src"
            / "output"
            / "val-r2r-reverie-cvdn-soon"
            / "val-r2r-reverie-cvdn-soon.log"
        )
        metrics = run_same.parse_metrics_from_log(log_path)
        self.assertAlmostEqual(metrics["REVERIE"]["val_unseen"]["nDTW"]["value"], 48.36)
        self.assertAlmostEqual(metrics["CVDN"]["val_unseen"]["oracle_path_success_rate"]["value"], 80.60)
        self.assertEqual(metrics["CVDN"]["val_unseen"]["oracle_path_success_rate"]["unit"], "%")
        self.assertAlmostEqual(metrics["SOON"]["val_unseen"]["det_sr"]["value"], 0.0)

    def test_cross_check_result_metrics_matches_existing_output(self) -> None:
        experiment_dir = REPO_ROOT / "third_party" / "SAME" / "src" / "output" / "val-r2r-eval-only"
        metrics = run_same.parse_metrics_from_log(experiment_dir / "val-r2r-eval-only.log")
        warnings = run_same.cross_check_result_metrics(experiment_dir, metrics)
        self.assertEqual(warnings, [])

    def test_build_data_manifest_skips_train_refs_for_eval_only(self) -> None:
        config = {
            "experiment": {
                "data_dir": "../../../data/same",
                "resume_file": "../../../data/same/ckpt/SAME.pt",
                "eval_only": True,
                "test": False,
            },
            "model": {"pretrained_ckpt": "../../../data/same/pretrain/Attnq_pretrained_ckpt.pt"},
            "simulator": {
                "connectivity_dir": {"mattersim": "../../../data/same/simulator/connectivity/"},
                "candidate_file_dir": {"mattersim": "../../../data/same/simulator/mp3d_scanvp_candidates.json"},
                "node_location_dir": {"mattersim": "../../../data/same/simulator/mp3d_connectivity_graphs.json"},
            },
            "feature": {
                "feature_database": {"mattersim": "features/img_features/demo.hdf5"},
                "object_database": {"reverie": "features/obj_features/reverie_obj_feat"},
            },
            "task": {
                "source": ["R2R"],
                "val_source": ["R2R"],
                "eval_splits": {"R2R": ["val_train_seen", "val_unseen"]},
            },
            "dataset": {
                "R2R": {
                    "DIR": "R2R",
                    "SPLIT": {
                        "train": "R2R_train_mergesim_enc.json",
                        "val_train_seen": "R2R_val_train_seen_enc.json",
                        "val_unseen": "R2R_val_unseen_enc.json",
                    },
                }
            },
        }
        manifest = run_same.build_data_manifest(config)
        self.assertIn("eval:R2R:val_unseen", manifest)
        self.assertNotIn("train:R2R:train", manifest)

    def test_csv_helpers_reject_duplicate_metrics_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_metrics = Path(tmpdir) / "metrics_long.csv"
            original_path = run_same.METRICS_LONG_CSV
            run_same.METRICS_LONG_CSV = tmp_metrics
            try:
                rows = [
                    {
                        "experiment_id": "0001_same_demo_same_s0_v1",
                        "dataset": "R2R",
                        "split": "val_unseen",
                        "metric": "sr",
                        "value": 76.29,
                        "unit": "%",
                    }
                ]
                run_same.append_metrics_rows_if_missing(rows)
                with self.assertRaises(ValueError):
                    run_same.append_metrics_rows_if_missing(rows)
            finally:
                run_same.METRICS_LONG_CSV = original_path


if __name__ == "__main__":
    unittest.main()
