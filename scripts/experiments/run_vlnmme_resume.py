#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "vlnmme" / "r2r_internvl3_2b_tiny.yaml"
DEFAULT_VLNMME_SRC = REPO_ROOT / "third_party" / "VLN-MME" / "src"
R2R_TASK = "R2R"
R2R_DATASET_NAME = "r2r"
ENV_FALSE_VALUES = {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VLN-MME R2R evaluation with resumable inference. The runner "
            "keeps final outputs in the original VLN-MME result path and uses "
            "temporary splits under resume_work/ for unfinished samples."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("CONFIG_PATH", DEFAULT_CONFIG)),
        help="Original VLN-MME config path. Defaults to CONFIG_PATH or tiny smoke config.",
    )
    parser.add_argument(
        "--vlnmme-src",
        type=Path,
        default=Path(os.environ.get("VLNMME_SRC_DIR", DEFAULT_VLNMME_SRC)),
        help="Path to third_party/VLN-MME/src.",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable inside the VLN-MME environment.",
    )
    parser.add_argument(
        "--raw-items-per-run",
        type=int,
        default=int(os.environ.get("VLNMME_RESUME_RAW_ITEMS", "0")),
        help=(
            "Optional cap on raw R2R path items per VLN-MME subprocess. "
            "0 means run all pending items in one subprocess."
        ),
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Do not run VLN-MME valid_from_file after all splits are complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resume plan without launching VLN-MME.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{repo_rel(path)} must contain a YAML object")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    tmp_path.replace(path)


def hf_offline_enabled() -> bool:
    return os.environ.get("VLNMME_HF_OFFLINE", "1").strip().lower() not in ENV_FALSE_VALUES


def make_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if hf_offline_enabled():
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        env.pop("HF_DATASETS_OFFLINE", None)
    return env


def resolve_runtime_path(path_value: str, *, vlnmme_src: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = vlnmme_src / path
    return path.resolve()


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def result_path(output_dir: Path, agent_type: str, model: str, env_name: str) -> Path:
    return output_dir / agent_type / model / f"{env_name}.json"


def r2r_instr_ids(item: dict[str, Any]) -> list[str]:
    instructions = item.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError("R2R item has no instructions list")
    return [f"{R2R_DATASET_NAME}_{item['path_id']}_{idx}" for idx in range(len(instructions))]


def expected_instr_ids(raw_data: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in raw_data:
        ids.extend(r2r_instr_ids(item))
    return ids


def load_r2r_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{repo_rel(path)} must contain a JSON list")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{repo_rel(path)} item {index} must be an object")
        for key in ("path_id", "instructions", "instr_encodings"):
            if key not in item:
                raise ValueError(f"{repo_rel(path)} item {index} missing key: {key}")
    return data


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"[resume] warning: skip invalid JSON result {repo_rel(path)}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        print(f"[resume] warning: skip non-list result {repo_rel(path)}", file=sys.stderr)
        return []
    return [
        item
        for item in data
        if isinstance(item, dict) and "instr_id" in item and "trajectory" in item
    ]


def ordered_unique_results(
    base_results: list[dict[str, Any]],
    extra_results: list[dict[str, Any]],
    expected_ids: list[str],
) -> tuple[list[dict[str, Any]], int]:
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}

    for item in base_results:
        instr_id = str(item["instr_id"])
        if instr_id in expected_set and instr_id not in by_id:
            by_id[instr_id] = item

    before = len(by_id)
    for item in extra_results:
        instr_id = str(item["instr_id"])
        if instr_id in expected_set and instr_id not in by_id:
            by_id[instr_id] = item

    ordered = [by_id[instr_id] for instr_id in expected_ids if instr_id in by_id]
    return ordered, len(by_id) - before


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, sort_keys=True, indent=4, separators=(",", ": "))
    tmp_path.replace(path)


def build_pending_data(
    raw_data: list[dict[str, Any]],
    completed_ids: set[str],
    *,
    raw_items_per_run: int,
) -> tuple[list[dict[str, Any]], int]:
    pending: list[dict[str, Any]] = []
    pending_instr_count = 0
    for item in raw_data:
        item_ids = r2r_instr_ids(item)
        missing = [instr_id for instr_id in item_ids if instr_id not in completed_ids]
        if not missing:
            continue
        pending.append(item)
        pending_instr_count += len(missing)
        if raw_items_per_run > 0 and len(pending) >= raw_items_per_run:
            break
    return pending, pending_instr_count


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def stream_subprocess(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[resume] $ {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def split_data_path(data_dir: Path, split: str) -> Path:
    return data_dir / R2R_TASK / f"{split}_enc.json"


def resolve_source_data_path(data_dir: Path, split: str, *, original_data_dir: str) -> Path:
    primary = split_data_path(data_dir, split)
    if primary.exists():
        return primary

    candidates: list[Path] = []
    original_path = Path(original_data_dir)
    if original_path.is_absolute():
        try:
            rel_to_workspace = original_path.relative_to("/workspace/vln-lab")
        except ValueError:
            pass
        else:
            candidates.append(REPO_ROOT / rel_to_workspace / R2R_TASK / f"{split}_enc.json")
    else:
        candidates.append(REPO_ROOT / "data" / "vlnmme" / R2R_TASK / f"{split}_enc.json")

    for candidate in candidates:
        if candidate.exists():
            print(f"[resume] source data fallback: {repo_rel(candidate)}", flush=True)
            return candidate

    return primary


def make_resume_config(
    original_config: dict[str, Any],
    *,
    temp_output_dir: Path,
    temp_data_dir: Path,
    temp_split: str,
) -> dict[str, Any]:
    config = deepcopy(original_config)
    config.setdefault("experiment", {})
    config.setdefault("task", {})
    config["experiment"]["output_dir"] = str(temp_output_dir)
    config["experiment"]["data_dir"] = str(temp_data_dir)
    config["experiment"]["valid_from_file"] = False
    config["task"]["val_source"] = [R2R_TASK]
    config["task"].setdefault("eval_splits", {})
    config["task"]["eval_splits"][R2R_TASK] = [temp_split]
    return config


def make_score_config(original_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(original_config)
    config.setdefault("experiment", {})
    config["experiment"]["valid_from_file"] = True
    return config


def run_split(
    *,
    split: str,
    original_config: dict[str, Any],
    vlnmme_src: Path,
    python_bin: str,
    output_dir: Path,
    data_dir: Path,
    original_data_dir: str,
    agent_type: str,
    model: str,
    resume_root: Path,
    raw_items_per_run: int,
    dry_run: bool,
) -> bool:
    env_name = f"{R2R_TASK}.{split}"
    original_result_path = result_path(output_dir, agent_type, model, env_name)
    source_data_path = resolve_source_data_path(data_dir, split, original_data_dir=original_data_dir)
    raw_data = load_r2r_data(source_data_path)
    expected_ids = expected_instr_ids(raw_data)

    temp_split = f"{split}__resume"
    temp_data_dir = resume_root / "data" / split
    temp_output_dir = resume_root / "runs" / split
    temp_config_path = resume_root / "configs" / f"{split}__resume.yaml"
    temp_result_path = result_path(temp_output_dir, agent_type, model, f"{R2R_TASK}.{temp_split}")

    old_final_results = load_results(original_result_path)
    merged, added = ordered_unique_results(old_final_results, load_results(temp_result_path), expected_ids)
    if added or (original_result_path.exists() and len(merged) != len(old_final_results)):
        save_results(original_result_path, merged)
        print(f"[resume] merged {added} recovered results -> {repo_rel(original_result_path)}", flush=True)

    completed_ids = {str(item["instr_id"]) for item in merged}
    total = len(expected_ids)
    done = len(completed_ids)
    if done >= total:
        print(f"[resume] {env_name}: complete ({done}/{total})", flush=True)
        return True

    pending_data, pending_instr_count = build_pending_data(
        raw_data,
        completed_ids,
        raw_items_per_run=raw_items_per_run,
    )
    print(
        f"[resume] {env_name}: {done}/{total} complete, "
        f"{pending_instr_count} pending instructions in {len(pending_data)} raw items",
        flush=True,
    )

    if dry_run:
        return False

    write_json(split_data_path(temp_data_dir, temp_split), pending_data)
    write_yaml(
        temp_config_path,
        make_resume_config(
            original_config,
            temp_output_dir=temp_output_dir,
            temp_data_dir=temp_data_dir,
            temp_split=temp_split,
        ),
    )

    return_code = stream_subprocess(
        [python_bin, "main.py", "--config_dir", str(temp_config_path)],
        cwd=vlnmme_src,
        log_path=resume_root / "logs" / f"{split}.stdout.log",
        env=make_child_env(),
    )

    merged_after, added_after = ordered_unique_results(
        load_results(original_result_path),
        load_results(temp_result_path),
        expected_ids,
    )
    if added_after:
        save_results(original_result_path, merged_after)
        print(f"[resume] merged {added_after} new results -> {repo_rel(original_result_path)}", flush=True)

    if return_code != 0:
        print(f"[resume] VLN-MME exited with code {return_code}; rerun the same command to continue.", file=sys.stderr)
        raise SystemExit(return_code)

    completed_after = len({str(item["instr_id"]) for item in merged_after})
    print(f"[resume] {env_name}: {completed_after}/{total} complete after this run", flush=True)
    return completed_after >= total


def run_score(
    *,
    original_config: dict[str, Any],
    vlnmme_src: Path,
    python_bin: str,
    resume_root: Path,
) -> None:
    score_config_path = resume_root / "configs" / "score_from_file.yaml"
    write_yaml(score_config_path, make_score_config(original_config))
    return_code = stream_subprocess(
        [python_bin, "main.py", "--config_dir", str(score_config_path)],
        cwd=vlnmme_src,
        log_path=resume_root / "logs" / "score_from_file.stdout.log",
        env=make_child_env(),
    )
    if return_code != 0:
        raise SystemExit(return_code)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    vlnmme_src = args.vlnmme_src.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {repo_rel(config_path)}")
    if not (vlnmme_src / "main.py").exists():
        raise FileNotFoundError(f"Missing VLN-MME main.py under {repo_rel(vlnmme_src)}")
    if args.raw_items_per_run < 0:
        raise ValueError("--raw-items-per-run must be >= 0")
    if hf_offline_enabled():
        print("[resume] Hugging Face offline mode enabled (set VLNMME_HF_OFFLINE=0 to allow network)", flush=True)

    original_config = load_yaml(config_path)
    val_source = get_nested(original_config, "task", "val_source")
    if val_source != [R2R_TASK]:
        raise ValueError("run_vlnmme_resume.py currently supports task.val_source: ['R2R'] only")

    splits = get_nested(original_config, "task", "eval_splits", R2R_TASK)
    if isinstance(splits, str):
        splits = [splits]
    if not isinstance(splits, list) or not all(isinstance(split, str) for split in splits):
        raise ValueError("task.eval_splits.R2R must be a string or list of strings")

    output_dir = resolve_runtime_path(str(get_nested(original_config, "experiment", "output_dir")), vlnmme_src=vlnmme_src)
    original_data_dir = str(get_nested(original_config, "experiment", "data_dir"))
    data_dir = resolve_runtime_path(original_data_dir, vlnmme_src=vlnmme_src)
    agent_type = str(get_nested(original_config, "agent", "type"))
    model = str(get_nested(original_config, "experiment", "model"))
    resume_root = output_dir / "resume_work"

    all_complete = True
    for split in splits:
        complete = run_split(
            split=split,
            original_config=original_config,
            vlnmme_src=vlnmme_src,
            python_bin=args.python_bin,
            output_dir=output_dir,
            data_dir=data_dir,
            original_data_dir=original_data_dir,
            agent_type=agent_type,
            model=model,
            resume_root=resume_root,
            raw_items_per_run=args.raw_items_per_run,
            dry_run=args.dry_run,
        )
        all_complete = all_complete and complete

    if args.dry_run:
        return

    if all_complete and not args.no_score:
        print("[resume] all splits complete; running valid_from_file scoring", flush=True)
        run_score(
            original_config=original_config,
            vlnmme_src=vlnmme_src,
            python_bin=args.python_bin,
            resume_root=resume_root,
        )
    elif not all_complete:
        print("[resume] not complete yet; rerun the same command to continue", flush=True)


if __name__ == "__main__":
    main()
