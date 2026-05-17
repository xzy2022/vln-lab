#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "vlnmme" / "r2r_internvl3_2b_tiny.yaml"
DEFAULT_VLNMME_SRC = REPO_ROOT / "third_party" / "VLN-MME" / "src"
R2R_TASK = "R2R"
R2R_DATASET_NAME = "r2r"
ENV_FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_CONTAINER_HF_CACHE = Path("/workspace/vln-lab/external/hf-cache")


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
        "--gpus",
        default=os.environ.get("VLNMME_GPUS", ""),
        help=(
            "Comma-separated physical GPU IDs to use for parallel resume, e.g. '0,2,5'. "
            "When set, one VLN-MME subprocess is launched per listed GPU with "
            "CUDA_VISIBLE_DEVICES pinned to that single GPU. Omit for the original serial runner."
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


@dataclass
class RunningProcess:
    label: str
    log_path: Path
    process: subprocess.Popen[str]
    log_handle: TextIO


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


def parse_gpu_list(value: str) -> list[str]:
    if not value.strip():
        return []
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id when provided")
    duplicates = sorted({gpu for gpu in gpus if gpus.count(gpu) > 1})
    if duplicates:
        raise ValueError(f"--gpus contains duplicate GPU ids: {', '.join(duplicates)}")
    return gpus


def make_child_env(*, cuda_visible_devices: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if DEFAULT_CONTAINER_HF_CACHE.exists():
        env.setdefault("HF_HOME", str(DEFAULT_CONTAINER_HF_CACHE))
        env.setdefault("HF_HUB_CACHE", str(DEFAULT_CONTAINER_HF_CACHE))
    if cuda_visible_devices is not None:
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
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


def load_results_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        results.extend(load_results(path))
    return results


def collect_resume_result_paths(
    resume_root: Path,
    *,
    split: str,
    agent_type: str,
    model: str,
) -> list[Path]:
    runs_root = resume_root / "runs" / split
    if not runs_root.exists():
        return []
    pattern = f"**/{agent_type}/{model}/{R2R_TASK}.{split}__resume*.json"
    return sorted(path for path in runs_root.glob(pattern) if path.is_file())


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


def build_parallel_pending_data(
    raw_data: list[dict[str, Any]],
    completed_ids: set[str],
    *,
    raw_items_per_worker: int,
    worker_count: int,
) -> tuple[list[list[dict[str, Any]]], list[int], int]:
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    instr_counts = [0 for _ in range(worker_count)]
    total_pending_instr_count = 0

    for item in raw_data:
        item_ids = r2r_instr_ids(item)
        missing = [instr_id for instr_id in item_ids if instr_id not in completed_ids]
        if not missing:
            continue

        available_workers = [
            index
            for index, chunk in enumerate(chunks)
            if raw_items_per_worker == 0 or len(chunk) < raw_items_per_worker
        ]
        if not available_workers:
            break

        worker_index = min(
            available_workers,
            key=lambda index: (instr_counts[index], len(chunks[index]), index),
        )
        chunks[worker_index].append(item)
        instr_counts[worker_index] += len(missing)
        total_pending_instr_count += len(missing)

    return chunks, instr_counts, total_pending_instr_count


def safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def start_subprocess_to_log(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    label: str,
) -> RunningProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[resume] start {label}: $ {' '.join(cmd)}", flush=True)
    print(f"[resume] {label} log: {repo_rel(log_path)}", flush=True)
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        log_handle.write(f"\n$ {' '.join(cmd)}\n")
        log_handle.flush()
        process: subprocess.Popen[str] = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        log_handle.close()
        raise
    return RunningProcess(label=label, log_path=log_path, process=process, log_handle=log_handle)


def terminate_processes(processes: list[RunningProcess]) -> None:
    for item in processes:
        if item.process.poll() is None:
            item.process.terminate()
    deadline = time.monotonic() + 20
    for item in processes:
        if item.process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            item.process.kill()
    for item in processes:
        item.log_handle.close()


def wait_logged_processes(processes: list[RunningProcess]) -> dict[str, int]:
    return_codes: dict[str, int] = {}
    remaining = list(processes)
    try:
        while remaining:
            for item in list(remaining):
                return_code = item.process.poll()
                if return_code is None:
                    continue
                item.log_handle.close()
                return_codes[item.label] = return_code
                remaining.remove(item)
                print(f"[resume] finish {item.label}: exit {return_code}", flush=True)
            if remaining:
                time.sleep(5)
    except KeyboardInterrupt:
        terminate_processes(remaining)
        raise
    return return_codes


def print_log_tail(path: Path, *, max_lines: int = 40) -> None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            tail = deque(handle, maxlen=max_lines)
    except OSError as exc:
        print(f"[resume] failed to read log tail {repo_rel(path)}: {exc}", file=sys.stderr)
        return
    print(f"[resume] last {len(tail)} log lines from {repo_rel(path)}:", file=sys.stderr)
    for line in tail:
        print(line.rstrip("\n"), file=sys.stderr)


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
        candidates.append(REPO_ROOT / "external" / "datasets" / "vlnmme" / R2R_TASK / f"{split}_enc.json")
        candidates.append(REPO_ROOT / "external" / "datasets" / R2R_TASK / f"{split}_enc.json")

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
    temp_result_paths = collect_resume_result_paths(
        resume_root,
        split=split,
        agent_type=agent_type,
        model=model,
    )
    if temp_result_path not in temp_result_paths and temp_result_path.exists():
        temp_result_paths.append(temp_result_path)
    merged, added = ordered_unique_results(
        old_final_results,
        load_results_from_paths(temp_result_paths),
        expected_ids,
    )
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
        load_results_from_paths(
            collect_resume_result_paths(
                resume_root,
                split=split,
                agent_type=agent_type,
                model=model,
            )
        ),
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


def run_parallel_split(
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
    gpus: list[str],
    dry_run: bool,
) -> bool:
    env_name = f"{R2R_TASK}.{split}"
    original_result_path = result_path(output_dir, agent_type, model, env_name)
    source_data_path = resolve_source_data_path(data_dir, split, original_data_dir=original_data_dir)
    raw_data = load_r2r_data(source_data_path)
    expected_ids = expected_instr_ids(raw_data)

    temp_result_paths = collect_resume_result_paths(
        resume_root,
        split=split,
        agent_type=agent_type,
        model=model,
    )
    old_final_results = load_results(original_result_path)
    merged, added = ordered_unique_results(
        old_final_results,
        load_results_from_paths(temp_result_paths),
        expected_ids,
    )
    if added or (original_result_path.exists() and len(merged) != len(old_final_results)):
        save_results(original_result_path, merged)
        print(f"[resume] merged {added} recovered results -> {repo_rel(original_result_path)}", flush=True)

    completed_ids = {str(item["instr_id"]) for item in merged}
    total = len(expected_ids)
    done = len(completed_ids)
    if done >= total:
        print(f"[resume] {env_name}: complete ({done}/{total})", flush=True)
        return True

    worker_chunks, worker_instr_counts, pending_instr_count = build_parallel_pending_data(
        raw_data,
        completed_ids,
        raw_items_per_worker=raw_items_per_run,
        worker_count=len(gpus),
    )
    active_worker_indexes = [index for index, chunk in enumerate(worker_chunks) if chunk]
    gpu_summary = ",".join(gpus)
    print(
        f"[resume] {env_name}: {done}/{total} complete, "
        f"{pending_instr_count} pending instructions in "
        f"{sum(len(worker_chunks[index]) for index in active_worker_indexes)} raw items "
        f"across {len(active_worker_indexes)}/{len(gpus)} workers (gpus: {gpu_summary})",
        flush=True,
    )

    for index in active_worker_indexes:
        print(
            f"[resume] worker {index:02d} gpu {gpus[index]}: "
            f"{worker_instr_counts[index]} pending instructions in {len(worker_chunks[index])} raw items",
            flush=True,
        )

    if dry_run:
        return False

    if not active_worker_indexes:
        return False

    processes: list[RunningProcess] = []
    for index in active_worker_indexes:
        gpu = gpus[index]
        temp_split = f"{split}__resume_w{index:02d}"
        worker_name = f"worker_{index:02d}"
        temp_data_dir = resume_root / "data" / split / worker_name
        temp_output_dir = resume_root / "runs" / split / worker_name
        temp_config_path = resume_root / "configs" / split / f"{worker_name}.yaml"

        write_json(split_data_path(temp_data_dir, temp_split), worker_chunks[index])
        write_yaml(
            temp_config_path,
            make_resume_config(
                original_config,
                temp_output_dir=temp_output_dir,
                temp_data_dir=temp_data_dir,
                temp_split=temp_split,
            ),
        )

        label = f"{worker_name}/gpu_{gpu}"
        log_path = resume_root / "logs" / split / f"{worker_name}_gpu_{safe_label(gpu)}.stdout.log"
        processes.append(
            start_subprocess_to_log(
                [python_bin, "main.py", "--config_dir", str(temp_config_path)],
                cwd=vlnmme_src,
                log_path=log_path,
                env=make_child_env(cuda_visible_devices=gpu),
                label=label,
            )
        )

    return_codes = wait_logged_processes(processes)

    merged_after, added_after = ordered_unique_results(
        load_results(original_result_path),
        load_results_from_paths(
            collect_resume_result_paths(
                resume_root,
                split=split,
                agent_type=agent_type,
                model=model,
            )
        ),
        expected_ids,
    )
    if added_after:
        save_results(original_result_path, merged_after)
        print(f"[resume] merged {added_after} new results -> {repo_rel(original_result_path)}", flush=True)

    failed = [
        process
        for process in processes
        if return_codes.get(process.label, 1) != 0
    ]
    if failed:
        for process in failed:
            print_log_tail(process.log_path)
        first_failure = failed[0]
        return_code = return_codes.get(first_failure.label, 1)
        print(
            f"[resume] VLN-MME parallel worker failed ({first_failure.label}, code {return_code}); "
            "rerun the same command to continue.",
            file=sys.stderr,
        )
        raise SystemExit(return_code)

    completed_after = len({str(item["instr_id"]) for item in merged_after})
    print(f"[resume] {env_name}: {completed_after}/{total} complete after this parallel run", flush=True)
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
    try:
        gpus = parse_gpu_list(args.gpus)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if hf_offline_enabled():
        print("[resume] Hugging Face offline mode enabled (set VLNMME_HF_OFFLINE=0 to allow network)", flush=True)
    if gpus:
        print(f"[resume] Parallel GPU mode enabled: {','.join(gpus)}", flush=True)

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
        if gpus:
            complete = run_parallel_split(
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
                gpus=gpus,
                dry_run=args.dry_run,
            )
        else:
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
