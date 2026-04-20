#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import unicodedata
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SAME_ROOT = REPO_ROOT / "third_party" / "SAME"
SAME_SRC_DIR = SAME_ROOT / "src"
EXPERIMENT_OUTPUTS_DIR = REPO_ROOT / "experiment_outputs"
LEGACY_EXPERIMENTS_DIR = REPO_ROOT / "experiments"
REPORT_TABLES_DIR = REPO_ROOT / "reports" / "tables"
PATCH_DIR = REPO_ROOT / "patches" / "same" / "base"
PATCH_SCRIPT = REPO_ROOT / "scripts" / "setup" / "apply_patches.sh"
DEFAULT_CONFIG_PATH = SAME_SRC_DIR / "configs" / "default.yaml"
RUNS_CSV = REPORT_TABLES_DIR / "runs.csv"
METRICS_LONG_CSV = REPORT_TABLES_DIR / "metrics_long.csv"
OFFICIAL_RESULTS_CSV = REPORT_TABLES_DIR / "official_results.csv"

RUNS_HEADER = [
    "experiment_id",
    "date",
    "run_type",
    "method",
    "datasets",
    "splits",
    "repo_commit",
    "child_repo_commit",
    "config",
    "checkpoint",
    "seed",
    "status",
    "log_path",
    "output_dir",
    "patch_set",
]
METRICS_LONG_HEADER = ["experiment_id", "dataset", "split", "metric", "value", "unit"]
OFFICIAL_RESULTS_HEADER = ["source", "method", "dataset", "split", "metric", "value", "note"]

EXPERIMENT_ID_RE = re.compile(r"^(?P<seq>\d+)_(?P<slug>.+)_v(?P<rev>\d+)$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
DATASET_LINE_RE = re.compile(
    r"^Dataset:\s*(?P<dataset>[^,]+),\s*Split:\s*(?P<split>[^,]+)(?:,\s*(?P<metrics>.*))?$"
)
METRIC_PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_ ][A-Za-z0-9_ \-]*?):\s*(?P<value>-?\d+(?:\.\d+)?)")
PROGRESS_LINE_RE = re.compile(r"^\s*(?:.*?:\s+)?\d{1,3}%\|")

PERCENT_METRICS = {
    "sr",
    "oracle_sr",
    "spl",
    "nDTW",
    "SDTW",
    "CLS",
    "rgs",
    "rgspl",
    "det_sr",
    "det_spl",
    "oracle_path_success_rate",
}
METER_METRICS = {
    "lengths",
    "nav_error",
    "oracle_error",
    "dist_to_end_reduction",
}
RESULT_CROSS_CHECK_METRICS = {
    "action_steps": ("action_steps", 1.0),
    "trajectory_steps": ("steps", 1.0),
    "trajectory_lengths": ("lengths", 1.0),
    "nav_error": ("nav_error", 1.0),
    "oracle_error": ("oracle_error", 1.0),
    "success": ("sr", 100.0),
    "oracle_success": ("oracle_sr", 100.0),
    "spl": ("spl", 100.0),
    "nDTW": ("nDTW", 100.0),
    "SDTW": ("SDTW", 100.0),
    "CLS": ("CLS", 100.0),
    "rgs": ("rgs", 100.0),
    "rgspl": ("rgspl", 100.0),
}
OUTPUT_DIR_OVERRIDE = "../../../experiment_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAME with experiment archiving and report updates.")
    parser.add_argument("--config", required=True, help="Path to the SAME config file.")
    parser.add_argument("--seed", type=int, help="Override experiment.seed.")
    parser.add_argument("--checkpoint", help="Override experiment.resume_file.")
    parser.add_argument("--tag", help="Optional extra slug segment added to experiment_id.")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional SAME OmegaConf override, repeatable.",
    )
    return parser.parse_args()


def lazy_import_omegaconf():
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "run_same.py 需要在 SAME 运行环境中执行。请先进入 docker 容器并激活 `test-v1`。"
        ) from exc
    return OmegaConf


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = normalized.strip("-._")
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized or "run"


def parse_override_value(raw_value: str) -> Any:
    if raw_value in {"True", "False"}:
        return raw_value == "True"
    if re.fullmatch(r"-?\d+", raw_value):
        return int(raw_value)
    if re.fullmatch(r"-?\d+\.\d+", raw_value):
        return float(raw_value)
    return raw_value


def normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_metric_name(name: str) -> str:
    normalized = name.strip().replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def metric_unit(metric_name: str) -> str:
    if metric_name in PERCENT_METRICS:
        return "%"
    if metric_name in METER_METRICS:
        return "m"
    return ""


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_override_key(option: str) -> str:
    if "=" not in option:
        raise ValueError(f"无效的 --option: {option!r}，需要 key=value 格式。")
    key, _ = option.split("=", 1)
    return key.strip()


def validate_user_options(seed: int | None, checkpoint: str | None, options: list[str]) -> None:
    reserved = {"experiment.id", "experiment.output_dir"}
    seen = {parse_override_key(option) for option in options}
    conflicts = sorted(reserved & seen)
    if conflicts:
        raise ValueError(f"--option 不允许覆盖 {', '.join(conflicts)}，这些字段由运行器管理。")
    if seed is not None and "experiment.seed" in seen:
        raise ValueError("请不要同时使用 --seed 和 --option experiment.seed=...")
    if checkpoint is not None and "experiment.resume_file" in seen:
        raise ValueError("请不要同时使用 --checkpoint 和 --option experiment.resume_file=...")


def build_user_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.option)
    if args.seed is not None:
        overrides.append(f"experiment.seed={args.seed}")
    if args.checkpoint is not None:
        overrides.append(f"experiment.resume_file={args.checkpoint}")
    return overrides


def load_same_config(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    OmegaConf = lazy_import_omegaconf()

    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到 SAME 默认配置: {DEFAULT_CONFIG_PATH}")
    if not config_path.exists():
        raise FileNotFoundError(f"未找到实验配置: {config_path}")

    config = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG_PATH), OmegaConf.load(config_path))
    for option in overrides:
        key, raw_value = option.split("=", 1)
        OmegaConf.update(config, key, parse_override_value(raw_value))
    return OmegaConf.to_container(config, resolve=True)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    OmegaConf = lazy_import_omegaconf()
    ensure_parent(path)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True), encoding="utf-8")


def get_in(payload: dict[str, Any], *parts: str, default: Any = None) -> Any:
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def read_csv_rows(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return None, []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def ensure_csv_header(path: Path, header: list[str]) -> None:
    ensure_parent(path)
    fieldnames, _rows = read_csv_rows(path)
    if fieldnames is None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
        return
    if fieldnames != header:
        raise ValueError(f"{repo_rel(path)} 表头不匹配，期望 {header}，实际 {fieldnames}")


def validate_readonly_csv_header(path: Path, header: list[str]) -> str | None:
    fieldnames, _rows = read_csv_rows(path)
    if fieldnames is None:
        return f"{repo_rel(path)} 为空，无法做 official_results 校验。"
    if fieldnames != header:
        return f"{repo_rel(path)} 表头不匹配，期望 {header}，实际 {fieldnames}"
    return None


def append_csv_row(path: Path, header: list[str], row: dict[str, Any]) -> None:
    ensure_csv_header(path, header)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writerow(row)


def collect_existing_experiment_ids(experiments_dirs: Iterable[Path], runs_csv: Path) -> set[str]:
    experiment_ids: set[str] = set()
    for experiments_dir in experiments_dirs:
        if experiments_dir.exists():
            experiment_ids.update(path.name for path in experiments_dir.iterdir() if path.is_dir())
    fieldnames, rows = read_csv_rows(runs_csv)
    if fieldnames and "experiment_id" in fieldnames:
        for row in rows:
            experiment_id = row.get("experiment_id", "").strip()
            if experiment_id:
                experiment_ids.add(experiment_id)
    return experiment_ids


def allocate_experiment_id(
    method_slug: str,
    config_stem: str,
    checkpoint_tag: str,
    seed: int,
    tag: str | None,
    existing_ids: Iterable[str],
) -> tuple[str, str]:
    slug_parts = [method_slug, config_stem, checkpoint_tag, f"s{seed}"]
    if tag:
        slug_parts.append(slugify(tag))
    experiment_slug = "_".join(part for part in slug_parts if part)

    max_sequence = 0
    max_revision = 0
    for existing_id in existing_ids:
        match = EXPERIMENT_ID_RE.match(existing_id)
        if not match:
            continue
        max_sequence = max(max_sequence, int(match.group("seq")))
        if match.group("slug") == experiment_slug:
            max_revision = max(max_revision, int(match.group("rev")))

    next_sequence = max_sequence + 1
    next_revision = max_revision + 1
    width = max(4, len(str(next_sequence)))
    experiment_id = f"{next_sequence:0{width}d}_{experiment_slug}_v{next_revision}"
    return experiment_id, experiment_slug


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def is_progress_line(line: str) -> bool:
    candidate = strip_ansi(line).strip("\r\n")
    return bool(PROGRESS_LINE_RE.match(candidate))


def collapse_progress_lines(lines: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    pending_progress: str | None = None
    for line in lines:
        if is_progress_line(line):
            pending_progress = line if line.endswith("\n") else f"{line}\n"
            continue
        if pending_progress is not None:
            collapsed.append(pending_progress)
            pending_progress = None
        collapsed.append(line)
    if pending_progress is not None:
        collapsed.append(pending_progress)
    return collapsed


def parse_metric_pairs(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in METRIC_PAIR_RE.finditer(text):
        metrics[normalize_metric_name(match.group("key"))] = float(match.group("value"))
    return metrics


def parse_metrics_from_log(log_path: Path) -> dict[str, dict[str, dict[str, dict[str, float | str]]]]:
    metrics: dict[str, dict[str, dict[str, dict[str, float | str]]]] = {}
    if not log_path.exists():
        return metrics

    pending_key: tuple[str, str] | None = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_ansi(raw_line).strip()
        if not line:
            continue

        dataset_match = DATASET_LINE_RE.match(line)
        if dataset_match:
            dataset = dataset_match.group("dataset").strip()
            split = dataset_match.group("split").strip()
            dataset_metrics = parse_metric_pairs(dataset_match.group("metrics") or "")
            split_metrics = metrics.setdefault(dataset, {}).setdefault(split, {})
            for metric_name, value in dataset_metrics.items():
                split_metrics[metric_name] = {"value": value, "unit": metric_unit(metric_name)}
            pending_key = (dataset, split)
            continue

        if pending_key is None:
            continue

        extra_metrics = parse_metric_pairs(line)
        if not extra_metrics:
            pending_key = None
            continue

        dataset, split = pending_key
        split_metrics = metrics.setdefault(dataset, {}).setdefault(split, {})
        for metric_name, value in extra_metrics.items():
            split_metrics[metric_name] = {"value": value, "unit": metric_unit(metric_name)}
        pending_key = None

    return metrics


def compute_result_summary(results_path: Path) -> dict[str, float]:
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload:
        return {}

    summary: dict[str, float] = {}
    for item_key, (metric_name, factor) in RESULT_CROSS_CHECK_METRICS.items():
        values = [row[item_key] for row in payload if item_key in row]
        if values:
            summary[metric_name] = statistics.fmean(values) * factor
    return summary


def cross_check_result_metrics(
    experiment_dir: Path,
    metrics: dict[str, dict[str, dict[str, dict[str, float | str]]]],
    tolerance: float = 0.05,
) -> list[str]:
    warnings: list[str] = []
    for dataset in ("R2R", "REVERIE"):
        for split, split_metrics in metrics.get(dataset, {}).items():
            results_path = experiment_dir / "results" / f"{dataset}_{split}_results.json"
            result_summary = compute_result_summary(results_path)
            if not result_summary:
                continue
            for metric_name, expected in result_summary.items():
                actual_wrapper = split_metrics.get(metric_name)
                if not actual_wrapper:
                    continue
                actual = float(actual_wrapper["value"])
                if abs(actual - expected) > tolerance:
                    warnings.append(
                        f"{dataset}/{split}/{metric_name} 与 results 文件均值不一致: "
                        f"log={actual:.4f}, results={expected:.4f}"
                    )
    return warnings


def canonical_reference_key(method: str, dataset: str, split: str, metric: str) -> tuple[str, str, str, str]:
    return (
        normalize_for_match(method),
        normalize_for_match(dataset),
        normalize_for_match(split),
        normalize_for_match(metric),
    )


def check_official_references(
    metrics: dict[str, dict[str, dict[str, dict[str, float | str]]]],
    method_name: str,
) -> list[str]:
    warnings: list[str] = []
    header_warning = validate_readonly_csv_header(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULTS_HEADER)
    if header_warning:
        return [header_warning]

    _fieldnames, rows = read_csv_rows(OFFICIAL_RESULTS_CSV)
    reference_keys = {
        canonical_reference_key(
            row.get("method", ""),
            row.get("dataset", ""),
            row.get("split", ""),
            row.get("metric", ""),
        )
        for row in rows
    }
    for dataset, split_map in metrics.items():
        for split, metric_map in split_map.items():
            for metric_name in metric_map:
                key = canonical_reference_key(method_name, dataset, split, metric_name)
                if key not in reference_keys:
                    warnings.append(
                        f"official_results.csv 缺少参考项: method={method_name}, dataset={dataset}, "
                        f"split={split}, metric={metric_name}"
                    )
    return warnings


def flatten_metrics_for_csv(
    experiment_id: str,
    metrics: dict[str, dict[str, dict[str, dict[str, float | str]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, split_map in metrics.items():
        for split, metric_map in split_map.items():
            for metric_name, payload in metric_map.items():
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": dataset,
                        "split": split,
                        "metric": metric_name,
                        "value": payload["value"],
                        "unit": payload["unit"],
                    }
                )
    return rows


def resolve_same_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (SAME_SRC_DIR / path).resolve()


def resolve_data_path(data_dir: Path, *parts: str) -> Path:
    return (data_dir / Path(*parts)).resolve()


def collect_dataset_split_refs(config: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    task_cfg = get_in(config, "task", default={}) or {}
    dataset_cfg = get_in(config, "dataset", default={}) or {}
    exp_cfg = get_in(config, "experiment", default={}) or {}

    if exp_cfg.get("test"):
        dataset_names = ensure_list(task_cfg.get("test_source"))
        split_provider = lambda dataset_name: ["test"]
        phase = "test"
    else:
        dataset_names = ensure_list(task_cfg.get("val_source")) or ensure_list(task_cfg.get("source"))
        eval_splits = task_cfg.get("eval_splits", {})
        split_provider = lambda dataset_name: ensure_list(eval_splits.get(dataset_name))
        phase = "eval"

    data_root = resolve_same_path(str(exp_cfg.get("data_dir", "")))
    if data_root is None:
        return refs

    for dataset_name in ensure_list(task_cfg.get("source")):
        if exp_cfg.get("test") or exp_cfg.get("eval_only"):
            break
        dataset_info = dataset_cfg.get(dataset_name, {})
        split_name = "train"
        split_file = get_in(dataset_info, "SPLIT", split_name)
        if split_file:
            refs.append(
                {
                    "phase": "train",
                    "dataset": dataset_name,
                    "split": split_name,
                    "raw": str(Path(dataset_info["DIR"]) / split_file),
                    "resolved": str(resolve_data_path(data_root, dataset_info["DIR"], split_file)),
                }
            )

    for dataset_name in dataset_names:
        dataset_info = dataset_cfg.get(dataset_name, {})
        for split_name in split_provider(dataset_name):
            split_file = get_in(dataset_info, "SPLIT", split_name)
            if not split_file:
                continue
            refs.append(
                {
                    "phase": phase,
                    "dataset": dataset_name,
                    "split": split_name,
                    "raw": str(Path(dataset_info["DIR"]) / split_file),
                    "resolved": str(resolve_data_path(data_root, dataset_info["DIR"], split_file)),
                }
            )
        bbox_file = dataset_info.get("bbox_file")
        if bbox_file:
            refs.append(
                {
                    "phase": "aux",
                    "dataset": dataset_name,
                    "split": "bbox",
                    "raw": str(Path(dataset_info["DIR"]) / bbox_file),
                    "resolved": str(resolve_data_path(data_root, dataset_info["DIR"], bbox_file)),
                }
            )
    return refs


def format_manifest_path(raw_path: str, resolved_path: Path | None) -> str:
    if raw_path in {"", "None"}:
        return "<unset>"
    if resolved_path is None:
        return f"{raw_path} -> <unresolved>"
    suffix = "exists" if resolved_path.exists() else "missing"
    return f"{raw_path} -> {resolved_path} [{suffix}]"


def build_data_manifest(config: dict[str, Any]) -> str:
    lines: list[str] = []
    exp_cfg = get_in(config, "experiment", default={}) or {}
    model_cfg = get_in(config, "model", default={}) or {}
    simulator_cfg = get_in(config, "simulator", default={}) or {}
    feature_cfg = get_in(config, "feature", default={}) or {}

    data_dir_raw = str(exp_cfg.get("data_dir", ""))
    data_dir_resolved = resolve_same_path(data_dir_raw)
    lines.append("[experiment]")
    lines.append(f"data_dir = {format_manifest_path(data_dir_raw, data_dir_resolved)}")

    resume_file = exp_cfg.get("resume_file")
    lines.append(f"resume_file = {format_manifest_path(str(resume_file), resolve_same_path(str(resume_file)))}")
    pretrained_ckpt = model_cfg.get("pretrained_ckpt")
    lines.append(
        f"pretrained_ckpt = {format_manifest_path(str(pretrained_ckpt), resolve_same_path(str(pretrained_ckpt)))}"
    )

    lines.append("")
    lines.append("[simulator]")
    for section_name in ("connectivity_dir", "candidate_file_dir", "node_location_dir"):
        section = simulator_cfg.get(section_name, {}) or {}
        for key in sorted(section):
            raw_path = str(section[key])
            lines.append(f"{section_name}.{key} = {format_manifest_path(raw_path, resolve_same_path(raw_path))}")

    lines.append("")
    lines.append("[feature_database]")
    for key, raw_path in sorted((feature_cfg.get("feature_database", {}) or {}).items()):
        resolved = resolve_data_path(data_dir_resolved, raw_path) if data_dir_resolved else None
        lines.append(f"{key} = {format_manifest_path(str(raw_path), resolved)}")

    lines.append("")
    lines.append("[object_database]")
    for key, raw_path in sorted((feature_cfg.get("object_database", {}) or {}).items()):
        resolved = resolve_data_path(data_dir_resolved, raw_path) if data_dir_resolved else None
        lines.append(f"{key} = {format_manifest_path(str(raw_path), resolved)}")

    lines.append("")
    lines.append("[dataset_files]")
    for ref in collect_dataset_split_refs(config):
        resolved = Path(ref["resolved"])
        lines.append(
            f"{ref['phase']}:{ref['dataset']}:{ref['split']} = "
            f"{format_manifest_path(ref['raw'], resolved)}"
        )
    return "\n".join(lines) + "\n"


def infer_run_type(config: dict[str, Any]) -> str:
    exp_cfg = get_in(config, "experiment", default={}) or {}
    has_checkpoint = bool(exp_cfg.get("resume_file"))
    if exp_cfg.get("test"):
        return "checkpoint_test" if has_checkpoint else "test"
    if exp_cfg.get("eval_only"):
        return "checkpoint_eval" if has_checkpoint else "eval_only"
    if has_checkpoint:
        return "resume_train"
    return "train"


def summarize_datasets_and_splits(config: dict[str, Any]) -> tuple[str, str]:
    refs = collect_dataset_split_refs(config)
    datasets = sorted({ref["dataset"] for ref in refs})
    splits = sorted({f"{ref['dataset']}:{ref['split']}" for ref in refs if ref["phase"] != "aux"})
    return ";".join(datasets), ";".join(splits)


def build_patch_set() -> list[str]:
    if not PATCH_DIR.exists():
        return []
    return [repo_rel(path) for path in sorted(PATCH_DIR.glob("*.patch"))]


def run_shell_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False, env=env)


def run_git_command(repo_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="run-same-git-") as temp_home:
        env = os.environ.copy()
        env["HOME"] = temp_home
        for safe_dir in (REPO_ROOT.resolve(), SAME_ROOT.resolve()):
            setup = run_shell_command(
                ["git", "config", "--global", "--add", "safe.directory", str(safe_dir)],
                repo_dir,
                env=env,
            )
            if setup.returncode != 0:
                return setup
        return run_shell_command(["git"] + args, repo_dir, env=env)


def read_git_commit(repo_dir: Path) -> str:
    result = run_git_command(repo_dir, ["rev-parse", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def read_git_status(repo_dir: Path) -> str:
    result = run_git_command(repo_dir, ["status", "--short"])
    if result.returncode != 0:
        return "<unavailable>"
    return result.stdout.strip() or "<clean>"


def build_git_info_text() -> str:
    root_commit = read_git_commit(REPO_ROOT)
    child_commit = read_git_commit(SAME_ROOT)
    root_status = read_git_status(REPO_ROOT)
    child_status = read_git_status(SAME_ROOT)
    lines = [
        "[repo]",
        f"commit = {root_commit}",
        "status =",
        root_status,
        "",
        "[third_party/SAME]",
        f"commit = {child_commit}",
        "status =",
        child_status,
        "",
    ]
    return "\n".join(lines)


def git_diff_output(repo_dir: Path, include_cached: bool, exclude_reports: bool) -> str:
    command = ["diff", "--submodule=diff"]
    if include_cached:
        command.append("--cached")
    if exclude_reports:
        command.extend(
            [
                "--",
                ".",
                ":(exclude)experiments",
                ":(exclude)experiment_outputs",
                ":(exclude)reports",
            ]
        )
    result = run_git_command(repo_dir, command)
    if result.returncode != 0:
        return "<unavailable>\n"
    return result.stdout or "<clean>\n"


def build_patch_diff_text() -> str:
    parts = [
        "# repo diff (unstaged)\n",
        git_diff_output(REPO_ROOT, include_cached=False, exclude_reports=True),
        "\n# repo diff (staged)\n",
        git_diff_output(REPO_ROOT, include_cached=True, exclude_reports=True),
        "\n# third_party/SAME diff (unstaged)\n",
        git_diff_output(SAME_ROOT, include_cached=False, exclude_reports=False),
        "\n# third_party/SAME diff (staged)\n",
        git_diff_output(SAME_ROOT, include_cached=True, exclude_reports=False),
    ]
    return "".join(parts)


def python_cuda_info(python_executable: str) -> str:
    command = [
        python_executable,
        "-c",
        (
            "import importlib.util, json; "
            "spec=importlib.util.find_spec('torch'); "
            "info={'torch_installed': bool(spec)}; "
            "print(json.dumps(info)) if not spec else exec(\""
            "import json, torch; "
            "info={"
            "'torch_version': torch.__version__, "
            "'cuda_version': torch.version.cuda, "
            "'cuda_available': torch.cuda.is_available(), "
            "'device_count': torch.cuda.device_count()"
            "}; "
            "info['device_name_0'] = torch.cuda.get_device_name(0) if info['cuda_available'] and info['device_count'] else None; "
            "print(json.dumps(info, ensure_ascii=False))"
            "\")"
        ),
    ]
    result = run_shell_command(command, REPO_ROOT)
    if result.returncode != 0:
        return f"<python torch probe failed>\n{result.stderr}"
    return result.stdout.strip()


def build_gpu_info_text(python_executable: str) -> str:
    nvidia_smi = run_shell_command(["nvidia-smi"], REPO_ROOT)
    lines = ["[nvidia-smi]"]
    if nvidia_smi.returncode == 0:
        lines.append(nvidia_smi.stdout.strip())
    else:
        lines.append(nvidia_smi.stderr.strip() or "<nvidia-smi unavailable>")
    lines.append("")
    lines.append("[python]")
    lines.append(python_cuda_info(python_executable))
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def append_stderr_message(stderr_path: Path, message: str) -> None:
    timestamp = now_local().strftime("%Y-%m-%d %H:%M:%S%z")
    with stderr_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[runner][{timestamp}] {message}\n")
    print(f"[runner] {message}", file=sys.stderr)


def stream_process(process: subprocess.Popen[str], stdout_path: Path, stderr_path: Path) -> int:
    ensure_parent(stdout_path)
    ensure_parent(stderr_path)
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")

    def forward(stream, file_handle, console, collapse_progress: bool) -> None:
        pending_progress: str | None = None
        try:
            for line in iter(stream.readline, ""):
                console.write(line)
                console.flush()
                if collapse_progress and is_progress_line(line):
                    pending_progress = line if line.endswith("\n") else f"{line}\n"
                    continue
                if pending_progress is not None:
                    file_handle.write(pending_progress)
                    file_handle.flush()
                    pending_progress = None
                file_handle.write(line)
                file_handle.flush()
            if pending_progress is not None:
                file_handle.write(pending_progress)
                file_handle.flush()
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=forward,
        args=(process.stdout, stdout_handle, sys.stdout, False),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=forward,
        args=(process.stderr, stderr_handle, sys.stderr, True),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    stdout_handle.close()
    stderr_handle.close()
    return return_code


def build_run_json(
    *,
    experiment_id: str,
    experiment_slug: str,
    config_path: Path,
    config: dict[str, Any],
    command: list[str],
    status: str,
    exit_code: int | None,
    started_at: dt.datetime,
    finished_at: dt.datetime | None,
    warnings: list[str],
    patch_set: list[str],
    run_type: str,
) -> dict[str, Any]:
    finished = finished_at or started_at
    duration_seconds = max((finished - started_at).total_seconds(), 0.0)
    return {
        "experiment_id": experiment_id,
        "experiment_slug": experiment_slug,
        "method": "SAME",
        "config_path": repo_rel(config_path),
        "command": shlex.join(command),
        "status": status,
        "exit_code": exit_code,
        "run_type": run_type,
        "started_at": started_at.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": duration_seconds,
        "seed": get_in(config, "experiment", "seed"),
        "checkpoint": get_in(config, "experiment", "resume_file"),
        "patch_set": patch_set,
        "repo_commit": read_git_commit(REPO_ROOT),
        "child_repo_commit": read_git_commit(SAME_ROOT),
        "workdir": repo_rel(SAME_SRC_DIR),
        "warnings": warnings,
    }


def write_run_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_runs_row_if_missing(row: dict[str, Any]) -> None:
    ensure_csv_header(RUNS_CSV, RUNS_HEADER)
    _fieldnames, rows = read_csv_rows(RUNS_CSV)
    existing = {current["experiment_id"] for current in rows}
    if row["experiment_id"] in existing:
        raise ValueError(f"runs.csv 已存在 experiment_id={row['experiment_id']}")
    append_csv_row(RUNS_CSV, RUNS_HEADER, row)


def append_metrics_rows_if_missing(rows_to_add: list[dict[str, Any]]) -> None:
    ensure_csv_header(METRICS_LONG_CSV, METRICS_LONG_HEADER)
    _fieldnames, rows = read_csv_rows(METRICS_LONG_CSV)
    existing = {
        (row["experiment_id"], row["dataset"], row["split"], row["metric"])
        for row in rows
    }
    incoming = set()
    for row in rows_to_add:
        key = (row["experiment_id"], row["dataset"], row["split"], row["metric"])
        if key in existing:
            raise ValueError(f"metrics_long.csv 已存在 {key}")
        if key in incoming:
            raise ValueError(f"本次运行产生了重复指标 {key}")
        incoming.add(key)
    for row in rows_to_add:
        append_csv_row(METRICS_LONG_CSV, METRICS_LONG_HEADER, row)


def checkpoint_tag_from_config(config: dict[str, Any]) -> str:
    checkpoint = get_in(config, "experiment", "resume_file")
    if not checkpoint:
        return "no-ckpt"
    return slugify(Path(str(checkpoint)).stem)


def apply_same_patches(stderr_path: Path) -> None:
    command = ["bash", str(PATCH_SCRIPT), "--method", "same"]
    result = run_shell_command(command, REPO_ROOT)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "apply_patches.sh failed"
        append_stderr_message(stderr_path, f"应用 SAME patches 失败: {message}")
        raise RuntimeError("无法应用 SAME base patches")


def build_same_command(config_path: Path, user_overrides: list[str], experiment_id: str) -> list[str]:
    command = [
        sys.executable,
        "run.py",
        "--config_dir",
        str(config_path),
    ]
    final_overrides = list(user_overrides)
    final_overrides.extend(
        [
            f"experiment.id={experiment_id}",
            f"experiment.output_dir={OUTPUT_DIR_OVERRIDE}",
        ]
    )
    if final_overrides:
        command.append("--options")
        command.extend(final_overrides)
    return command


def main() -> int:
    args = parse_args()
    validate_user_options(args.seed, args.checkpoint, args.option)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    user_overrides = build_user_overrides(args)
    config = load_same_config(config_path, user_overrides)

    existing_ids = collect_existing_experiment_ids(
        [EXPERIMENT_OUTPUTS_DIR, LEGACY_EXPERIMENTS_DIR],
        RUNS_CSV,
    )
    experiment_id, experiment_slug = allocate_experiment_id(
        method_slug="same",
        config_stem=slugify(config_path.stem),
        checkpoint_tag=checkpoint_tag_from_config(config),
        seed=int(get_in(config, "experiment", "seed", default=0)),
        tag=args.tag,
        existing_ids=existing_ids,
    )
    experiment_dir = EXPERIMENT_OUTPUTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = experiment_dir / "stdout.log"
    stderr_path = experiment_dir / "stderr.log"
    same_log_path = experiment_dir / f"{experiment_id}.log"
    metrics_path = experiment_dir / "metrics.json"
    run_json_path = experiment_dir / "run.json"
    config_resolved_path = experiment_dir / "config_resolved.yaml"
    git_info_path = experiment_dir / "git_info.txt"
    patch_diff_path = experiment_dir / "patch.diff"
    gpu_info_path = experiment_dir / "gpu_info.txt"
    data_manifest_path = experiment_dir / "data_manifest.txt"

    warnings: list[str] = []
    patch_set = build_patch_set()
    started_at = now_local()
    run_type = infer_run_type(config)

    dump_yaml(config_resolved_path, config)
    write_text(git_info_path, build_git_info_text())
    write_text(patch_diff_path, build_patch_diff_text())
    write_text(gpu_info_path, build_gpu_info_text(sys.executable))
    write_text(data_manifest_path, build_data_manifest(config))

    write_run_json(
        run_json_path,
        build_run_json(
            experiment_id=experiment_id,
            experiment_slug=experiment_slug,
            config_path=config_path,
            config=config,
            command=["<pending>"],
            status="running",
            exit_code=None,
            started_at=started_at,
            finished_at=None,
            warnings=warnings,
            patch_set=patch_set,
            run_type=run_type,
        ),
    )

    exit_code = 1
    status = "failed"

    try:
        apply_same_patches(stderr_path)
        command = build_same_command(config_path, user_overrides, experiment_id)
        write_run_json(
            run_json_path,
            build_run_json(
                experiment_id=experiment_id,
                experiment_slug=experiment_slug,
                config_path=config_path,
                config=config,
                command=command,
                status="running",
                exit_code=None,
                started_at=started_at,
                finished_at=None,
                warnings=warnings,
                patch_set=patch_set,
                run_type=run_type,
            ),
        )

        process = subprocess.Popen(
            command,
            cwd=SAME_SRC_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        exit_code = stream_process(process, stdout_path, stderr_path)
        status = "success" if exit_code == 0 else "failed"
    except Exception as exc:
        warnings.append(str(exc))
        append_stderr_message(stderr_path, str(exc))
    finally:
        metrics = parse_metrics_from_log(same_log_path)
        warnings.extend(cross_check_result_metrics(experiment_dir, metrics))
        if metrics:
            warnings.extend(check_official_references(metrics, "SAME"))
        deduped_warnings = list(dict.fromkeys(warnings))
        for warning in deduped_warnings:
            append_stderr_message(stderr_path, warning)

        metrics_payload = {
            "experiment_id": experiment_id,
            "status": status,
            "metrics": metrics,
        }
        write_text(metrics_path, json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

        finished_at = now_local()
        datasets_summary, splits_summary = summarize_datasets_and_splits(config)
        write_run_json(
            run_json_path,
            build_run_json(
                experiment_id=experiment_id,
                experiment_slug=experiment_slug,
                config_path=config_path,
                config=config,
                command=build_same_command(config_path, user_overrides, experiment_id),
                status=status,
                exit_code=exit_code,
                started_at=started_at,
                finished_at=finished_at,
                warnings=deduped_warnings,
                patch_set=patch_set,
                run_type=run_type,
            ),
        )

        ensure_csv_header(RUNS_CSV, RUNS_HEADER)
        ensure_csv_header(METRICS_LONG_CSV, METRICS_LONG_HEADER)
        runs_row = {
            "experiment_id": experiment_id,
            "date": started_at.strftime("%Y-%m-%d-%H:%M"),
            "run_type": run_type,
            "method": "SAME",
            "datasets": datasets_summary,
            "splits": splits_summary,
            "repo_commit": read_git_commit(REPO_ROOT),
            "child_repo_commit": read_git_commit(SAME_ROOT),
            "config": repo_rel(config_path),
            "checkpoint": get_in(config, "experiment", "resume_file") or "",
            "seed": get_in(config, "experiment", "seed"),
            "status": status,
            "log_path": repo_rel(stdout_path),
            "output_dir": repo_rel(experiment_dir),
            "patch_set": ";".join(patch_set),
        }
        append_runs_row_if_missing(runs_row)

        if status == "success" and metrics:
            append_metrics_rows_if_missing(flatten_metrics_for_csv(experiment_id, metrics))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
