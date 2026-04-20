from __future__ import annotations

import csv
import importlib.util
import io
import subprocess
import sys
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

    def test_build_metric_semantic_warnings_flags_unsupported_grounding_metrics(self) -> None:
        config = {
            "agent": {"type": "duet"},
            "feature": {"enable_og": False},
        }
        metrics = {
            "REVERIE": {
                "val_unseen": {
                    "rgs": {"value": 0.0, "unit": "%"},
                    "rgspl": {"value": 0.0, "unit": "%"},
                }
            },
            "SOON": {
                "val_unseen": {
                    "det_sr": {"value": 0.0, "unit": "%"},
                    "det_spl": {"value": 0.0, "unit": "%"},
                }
            },
        }
        warnings = run_same.build_metric_semantic_warnings(config, metrics)
        self.assertEqual(len(warnings), 1)
        self.assertIn("feature.enable_og=false", warnings[0])
        self.assertIn("REVERIE/val_unseen: rgs, rgspl", warnings[0])
        self.assertIn("SOON/val_unseen: det_spl, det_sr", warnings[0])

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

    def test_ensure_runs_csv_duration_column_migrates_legacy_header_without_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_runs = tmp_root / "reports" / "tables" / "runs.csv"

            rows = [
                {
                    "experiment_id": "0001_same_demo_same_s0_v1",
                    "date": "2026-04-20-10:00",
                    "run_type": "checkpoint_eval",
                    "method": "SAME",
                    "datasets": "R2R",
                    "splits": "R2R:val_unseen",
                    "repo_commit": "repo",
                    "child_repo_commit": "child",
                    "config": "configs/same/demo.yaml",
                    "checkpoint": "../../../data/same/ckpt/SAME.pt",
                    "seed": "0",
                    "status": "success",
                    "log_path": "experiment_outputs/0001_same_demo_same_s0_v1/stdout.log",
                    "output_dir": "experiment_outputs/0001_same_demo_same_s0_v1",
                    "patch_set": "patches/same/base/0001-eval-only-exit.patch",
                },
                {
                    "experiment_id": "0002_same_demo_same_s0_v2",
                    "date": "2026-04-20-11:00",
                    "run_type": "checkpoint_eval",
                    "method": "SAME",
                    "datasets": "R2R",
                    "splits": "R2R:val_unseen",
                    "repo_commit": "repo",
                    "child_repo_commit": "child",
                    "config": "configs/same/demo.yaml",
                    "checkpoint": "../../../data/same/ckpt/SAME.pt",
                    "seed": "0",
                    "status": "success",
                    "log_path": "experiment_outputs/0002_same_demo_same_s0_v2/stdout.log",
                    "output_dir": "experiment_outputs/0002_same_demo_same_s0_v2",
                    "patch_set": "patches/same/base/0001-eval-only-exit.patch",
                },
            ]
            tmp_runs.parent.mkdir(parents=True, exist_ok=True)
            with tmp_runs.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=run_same.RUNS_LEGACY_HEADER)
                writer.writeheader()
                writer.writerows(rows)

            original_runs_csv = run_same.RUNS_CSV
            run_same.RUNS_CSV = tmp_runs
            try:
                run_same.ensure_runs_csv_duration_column()
            finally:
                run_same.RUNS_CSV = original_runs_csv

            with tmp_runs.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, run_same.RUNS_HEADER)
                synced_rows = list(reader)

            self.assertEqual(synced_rows[0]["duration_hms"], "")
            self.assertEqual(synced_rows[1]["duration_hms"], "")

    def test_check_official_references_treats_cvdn_gp_as_dist_to_end_reduction_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_official = Path(tmpdir) / "official_results.csv"
            tmp_official.write_text(
                "source,method,dataset,split,metric,value,note\n"
                "paper,SAME,CVDN,val,GP,6.94,Table 4\n",
                encoding="utf-8",
            )

            original_official_results = run_same.OFFICIAL_RESULTS_CSV
            run_same.OFFICIAL_RESULTS_CSV = tmp_official
            try:
                warnings = run_same.check_official_references(
                    {
                        "CVDN": {
                            "val_unseen": {
                                "dist_to_end_reduction": {"value": 6.76, "unit": "m"},
                            }
                        }
                    },
                    "SAME",
                )
            finally:
                run_same.OFFICIAL_RESULTS_CSV = original_official_results

        self.assertEqual(warnings, [])

    def test_collapse_progress_lines_keeps_only_last_line_per_segment(self) -> None:
        lines = [
            "Loading data:   0%|          | 0/500 [00:00<?, ?it/s]\n",
            "Loading data:  42%|████▏     | 210/500 [00:00<00:00, 1234.00it/s]\n",
            "Loading data: 100%|██████████| 500/500 [00:00<00:00, 4321.00it/s]\n",
            "You are using a model of type bert to instantiate a model of type .\n",
            "  0%|          | 0/376 [00:00<?, ?it/s]\n",
            "100%|██████████| 376/376 [01:23<00:00,  4.52it/s]\n",
            "[runner][2026-04-20 02:24:30+0000] official_results.csv 缺少参考项\n",
        ]
        collapsed = run_same.collapse_progress_lines(lines)
        self.assertEqual(
            collapsed,
            [
                "Loading data: 100%|██████████| 500/500 [00:00<00:00, 4321.00it/s]\n",
                "You are using a model of type bert to instantiate a model of type .\n",
                "100%|██████████| 376/376 [01:23<00:00,  4.52it/s]\n",
                "[runner][2026-04-20 02:24:30+0000] official_results.csv 缺少参考项\n",
            ],
        )

    def test_stream_process_preserves_carriage_returns_for_console_and_collapses_stderr_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            stdout_path = tmp_root / "stdout.log"
            stderr_path = tmp_root / "stderr.log"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.stderr.write('Loading data:   0%|          | 0/2 [00:00<?, ?it/s]\\r'); "
                        "sys.stderr.flush(); "
                        "sys.stderr.write('Loading data: 100%|##########| 2/2 [00:00<00:00, 999.99it/s]\\r'); "
                        "sys.stderr.flush(); "
                        "sys.stderr.write('done\\n'); "
                        "sys.stderr.flush()"
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            original_stdout = run_same.sys.stdout
            original_stderr = run_same.sys.stderr
            fake_stdout = io.StringIO()
            fake_stderr = io.StringIO()
            run_same.sys.stdout = fake_stdout
            run_same.sys.stderr = fake_stderr
            try:
                exit_code = run_same.stream_process(process, stdout_path, stderr_path)
            finally:
                run_same.sys.stdout = original_stdout
                run_same.sys.stderr = original_stderr

            self.assertEqual(exit_code, 0)
            self.assertIn("\r", fake_stderr.getvalue())
            self.assertEqual(fake_stdout.getvalue(), "")
            self.assertEqual(
                stderr_path.read_text(encoding="utf-8"),
                "Loading data: 100%|##########| 2/2 [00:00<00:00, 999.99it/s]\ndone\n",
            )


if __name__ == "__main__":
    unittest.main()
