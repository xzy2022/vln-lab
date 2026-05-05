from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


prepare_navgpt2 = load_module(
    "prepare_navnuances_navgpt2_r2r",
    REPO_ROOT / "scripts" / "setup" / "prepare_navnuances_navgpt2_r2r.py",
)
run_navgpt2_eval = load_module(
    "run_navgpt2_navnuances_eval",
    REPO_ROOT / "scripts" / "eval" / "run_navgpt2_navnuances_eval.py",
)


class PrepareNavNuancesNavGPT2Tests(unittest.TestCase):
    def test_normalize_split_accepts_filenames(self) -> None:
        self.assertEqual(prepare_navgpt2.normalize_split("R2R_DC_enc.json"), "DC")
        self.assertEqual(prepare_navgpt2.normalize_split("lr"), "LR")
        self.assertEqual(prepare_navgpt2.normalize_split("standard"), "val_unseen")

    def test_install_navnuances_annotations_copies_requested_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            target_dir = tmp / "datasets" / "R2R" / "annotations"
            source_dir.mkdir()
            (source_dir / "R2R_DC_enc.json").write_text("dc\n", encoding="utf-8")
            (source_dir / "R2R_LR_enc.json").write_text("lr\n", encoding="utf-8")

            manifest = prepare_navgpt2.install_navnuances_annotations(
                source_dir=source_dir,
                target_annotations_dir=target_dir,
                splits=["R2R_DC_enc.json", "lr"],
            )

            self.assertEqual([entry["split"] for entry in manifest], ["DC", "LR"])
            self.assertEqual((target_dir / "R2R_DC_enc.json").read_text(encoding="utf-8"), "dc\n")
            self.assertEqual((target_dir / "R2R_LR_enc.json").read_text(encoding="utf-8"), "lr\n")

    def test_no_overwrite_rejects_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            target_dir = tmp / "target"
            source_dir.mkdir()
            target_dir.mkdir()
            (source_dir / "R2R_DC_enc.json").write_text("new\n", encoding="utf-8")
            (target_dir / "R2R_DC_enc.json").write_text("old\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_navgpt2.install_navnuances_annotations(
                    source_dir=source_dir,
                    target_annotations_dir=target_dir,
                    splits=["DC"],
                    overwrite=False,
                )


class RunNavGPT2NavNuancesEvalTests(unittest.TestCase):
    def test_export_submissions_validates_and_copies_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_dir = tmp / "preds"
            submission_dir = tmp / "submission"
            pred_dir.mkdir()
            payload = [{"instr_id": "scan-pair-0-path-0_0", "trajectory": [["vp1"], ["vp2"]]}]
            write_json(pred_dir / "submit_DC.json", payload)

            counts = run_navgpt2_eval.export_submissions(
                pred_dir=pred_dir,
                submission_dir=submission_dir,
                splits=["DC"],
            )

            self.assertEqual(counts, {"DC": 1})
            self.assertEqual(read_json(submission_dir / "submit_DC.json"), payload)

    def test_resolve_eval_splits_auto_adds_standard_when_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_dir = tmp / "preds"
            annotation_root = tmp / "annotations"
            pred_dir.mkdir()
            annotation_root.mkdir()
            write_json(pred_dir / "submit_val_unseen.json", [])
            write_json(annotation_root / "R2R_val_unseen.json", [])

            splits = run_navgpt2_eval.resolve_eval_splits(
                pred_dir=pred_dir,
                pred_prefix="submit",
                annotation_root=annotation_root,
                requested_splits=None,
                standard="auto",
            )

            self.assertEqual(splits, ["DC", "LR", "RR", "VM", "NU", "val_unseen"])

    def test_build_evaluator_command_skips_standard_by_default(self) -> None:
        command = run_navgpt2_eval.build_evaluator_command(
            python_bin="python",
            eval_script=Path("eval.py"),
            annotation_root=Path("annotations"),
            submission_dir=Path("submission"),
            out_dir=Path("out"),
            connectivity_dir=Path("connectivity"),
            include_standard=False,
        )

        self.assertEqual(command[:3], ["python", "eval.py", "--skip-standard"])

    def test_validate_required_navnuances_splits_rejects_partial_skill_set(self) -> None:
        with self.assertRaises(ValueError):
            run_navgpt2_eval.validate_required_navnuances_splits(["DC"])


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
