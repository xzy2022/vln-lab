#!/usr/bin/env python3
"""Generate reusable SAME baseline diagnostic plots from reports/tables/metrics_long.csv.

Recommended usage:

    conda run -n plots python scripts/plts/plot_same_baseline_diagnostics.py \
      --experiment-id 0005_same_val_r2r_reverie_cvdn_soon_same_s0_v1 \
      --metrics-csv reports/tables/metrics_long.csv \
      --output-dir reports/artifacts/plots

The script is designed around the repository's current reporting tables:

- reads long-format metric rows and pivots them into dataset/split records
- normalizes a few metric/split aliases to tolerate light schema drift
- generates four baseline diagnostic figures for a chosen experiment
- writes a short markdown summary with the key values and skipped items
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_CSV = REPO_ROOT / "reports" / "tables" / "metrics_long.csv"
DEFAULT_RUNS_CSV = REPO_ROOT / "reports" / "tables" / "runs.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "artifacts" / "plots"

EXPECTED_METRICS_COLUMNS = {"experiment_id", "dataset", "split", "metric", "value"}
EXPECTED_RUNS_COLUMNS = {"experiment_id", "splits"}

PREFERRED_DATASET_ORDER = [
    "R2R",
    "REVERIE",
    "CVDN",
    "SOON",
    "RxR-EN",
    "R2R-CE",
    "OBJECTNAV-MP3D",
]

METRIC_ALIASES = {
    "sr": "sr",
    "success": "sr",
    "spl": "spl",
    "oracle_sr": "oracle_sr",
    "oracle_success": "oracle_sr",
    "oracle_success_rate": "oracle_sr",
    "osr": "oracle_sr",
    "nav_error": "nav_error",
    "navigation_error": "nav_error",
    "ne": "nav_error",
    "oracle_error": "oracle_error",
    "oracle_navigation_error": "oracle_error",
    "oracle_ne": "oracle_error",
    "oe": "oracle_error",
}

SPLIT_ALIASES = {
    "val_train_seen": "val_train_seen",
    "train_seen": "train_seen",
    "val_seen": "val_seen",
    "seen": "seen",
    "val_unseen": "val_unseen",
    "unseen": "unseen",
    "test_seen": "test_seen",
    "test_unseen": "test_unseen",
    "val": "val",
    "test": "test",
}

SEEN_SPLIT_PRIORITY = ["val_train_seen", "val_seen", "train_seen", "seen", "test_seen"]
UNSEEN_SPLIT_PRIORITY = ["val_unseen", "unseen", "test_unseen"]

SERIES_COLORS = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
}


@dataclass(frozen=True)
class PlotArtifact:
    name: str
    title: str
    output_path: Path
    used_labels: list[str]
    skipped_items: list[str]
    key_messages: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAME baseline diagnostics for one experiment.")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier in reports/tables.")
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="Path to the long-format metrics table.",
    )
    parser.add_argument(
        "--runs-csv",
        default=str(DEFAULT_RUNS_CSV),
        help="Optional path to runs.csv for stable dataset/split ordering.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and the markdown summary will be written.",
    )
    parser.add_argument(
        "--style",
        default="default",
        help="Matplotlib style name. Falls back to default if unavailable.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output PNG DPI.")
    return parser.parse_args()


def normalize_token(value: str) -> str:
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in lowered)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def canonical_metric_name(name: str) -> str:
    token = normalize_token(name)
    return METRIC_ALIASES.get(token, token)


def canonical_split_name(split: str) -> str:
    token = normalize_token(split)
    return SPLIT_ALIASES.get(token, token)


def split_family(split: str) -> str | None:
    canonical = canonical_split_name(split)
    if "unseen" in canonical:
        return "unseen"
    if canonical in {"val_train_seen", "val_seen", "train_seen", "seen", "test_seen"}:
        return "seen"
    return None


def split_short_label(split: str) -> str:
    canonical = canonical_split_name(split)
    if canonical in {"val_train_seen", "val_seen", "train_seen", "seen"}:
        return "seen"
    if canonical in {"val_unseen", "unseen"}:
        return "unseen"
    return canonical.replace("_", "-")


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def load_run_order(runs_csv: Path, experiment_id: str) -> tuple[list[tuple[str, str]], list[str]]:
    warnings: list[str] = []
    if not runs_csv.exists():
        return [], [f"runs.csv not found, using metric-derived ordering: {repo_rel(runs_csv)}"]

    fieldnames, rows = read_csv_rows(runs_csv)
    missing_columns = sorted(EXPECTED_RUNS_COLUMNS - set(fieldnames))
    if missing_columns:
        warnings.append(
            f"{repo_rel(runs_csv)} is missing columns {missing_columns}, using metric-derived ordering."
        )
        return [], warnings

    for row in rows:
        if row.get("experiment_id", "").strip() != experiment_id:
            continue
        splits_field = row.get("splits", "").strip()
        order: list[tuple[str, str]] = []
        for chunk in splits_field.split(";"):
            if ":" not in chunk:
                continue
            dataset, raw_split = chunk.split(":", 1)
            dataset = dataset.strip()
            split = canonical_split_name(raw_split.strip())
            if dataset and split:
                order.append((dataset, split))
        if order:
            return order, warnings
        warnings.append(
            f"{repo_rel(runs_csv)} has experiment {experiment_id} but its splits column is empty."
        )
        return [], warnings

    warnings.append(f"{repo_rel(runs_csv)} does not contain experiment {experiment_id}.")
    return [], warnings


def load_experiment_metrics(
    metrics_csv: Path, experiment_id: str
) -> tuple[dict[tuple[str, str], dict[str, float]], list[str]]:
    fieldnames, rows = read_csv_rows(metrics_csv)
    missing_columns = sorted(EXPECTED_METRICS_COLUMNS - set(fieldnames))
    if missing_columns:
        raise ValueError(
            f"{repo_rel(metrics_csv)} is missing required columns {missing_columns}; "
            f"found columns: {fieldnames}"
        )

    grouped_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    warnings: list[str] = []
    matched_rows = 0

    for row in rows:
        if row.get("experiment_id", "").strip() != experiment_id:
            continue
        matched_rows += 1
        dataset = row.get("dataset", "").strip()
        split = canonical_split_name(row.get("split", ""))
        metric = canonical_metric_name(row.get("metric", ""))
        raw_value = row.get("value", "").strip()
        if not dataset or not split or not metric:
            warnings.append(f"Skipped malformed row with empty dataset/split/metric: {row}")
            continue
        if raw_value == "":
            warnings.append(f"Skipped empty metric value for {dataset}/{split}/{metric}.")
            continue
        try:
            value = float(raw_value)
        except ValueError:
            warnings.append(
                f"Skipped non-numeric metric value for {dataset}/{split}/{metric}: {raw_value!r}"
            )
            continue
        grouped_values[(dataset, split, metric)].append(value)

    if matched_rows == 0:
        raise ValueError(f"No rows found for experiment_id={experiment_id} in {repo_rel(metrics_csv)}")

    wide_metrics: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (dataset, split, metric), values in sorted(grouped_values.items()):
        if len(values) > 1:
            warnings.append(
                f"Found duplicate metric rows for {dataset}/{split}/{metric}; using their mean "
                f"({statistics.fmean(values):.4f}) from {len(values)} rows."
            )
        wide_metrics[(dataset, split)][metric] = statistics.fmean(values)

    return dict(wide_metrics), warnings


def sort_dataset_split_keys(
    metrics_by_key: dict[tuple[str, str], dict[str, float]], run_order: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    existing_keys = set(metrics_by_key)
    dataset_rank = {dataset: index for index, dataset in enumerate(PREFERRED_DATASET_ORDER)}
    run_index = {key: index for index, key in enumerate(run_order)}
    split_rank = {
        "val_train_seen": 0,
        "val_seen": 1,
        "train_seen": 2,
        "seen": 3,
        "val_unseen": 4,
        "unseen": 5,
        "test_seen": 6,
        "test_unseen": 7,
    }
    ordered_keys = sorted(
        existing_keys,
        key=lambda item: (
            dataset_rank.get(item[0], len(PREFERRED_DATASET_ORDER)),
            run_index.get(item, len(existing_keys)),
            split_rank.get(item[1], len(split_rank)),
            item[0],
            item[1],
        )
    )
    return ordered_keys


def build_display_labels(keys: Iterable[tuple[str, str]]) -> dict[tuple[str, str], str]:
    key_list = list(keys)
    short_counts = Counter((dataset, split_short_label(split)) for dataset, split in key_list)
    labels: dict[tuple[str, str], str] = {}
    for dataset, split in key_list:
        short_name = split_short_label(split)
        if short_counts[(dataset, short_name)] == 1:
            labels[(dataset, split)] = f"{dataset}-{short_name}"
        else:
            labels[(dataset, split)] = f"{dataset}-{canonical_split_name(split).replace('_', '-')}"
    return labels


def build_gap_records(
    metrics_by_key: dict[tuple[str, str], dict[str, float]],
    ordered_keys: Iterable[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    left_metric: str,
    right_metric: str,
) -> tuple[list[tuple[str, float]], list[str]]:
    records: list[tuple[str, float]] = []
    skipped: list[str] = []
    for key in ordered_keys:
        metric_map = metrics_by_key.get(key, {})
        if left_metric not in metric_map or right_metric not in metric_map:
            missing = [name for name in (left_metric, right_metric) if name not in metric_map]
            skipped.append(f"{labels[key]} (missing {', '.join(missing)})")
            continue
        records.append((labels[key], metric_map[left_metric] - metric_map[right_metric]))
    return records, skipped


def pick_dataset_family_key(
    metrics_by_key: dict[tuple[str, str], dict[str, float]],
    dataset: str,
    family: str,
) -> tuple[str, dict[str, float]] | None:
    if family == "seen":
        priority = SEEN_SPLIT_PRIORITY
    elif family == "unseen":
        priority = UNSEEN_SPLIT_PRIORITY
    else:
        return None

    available = {
        split: metric_map
        for (key_dataset, split), metric_map in metrics_by_key.items()
        if key_dataset == dataset and split_family(split) == family
    }
    if not available:
        return None
    for candidate in priority:
        if candidate in available:
            return candidate, available[candidate]
    split = sorted(available)[0]
    return split, available[split]


def build_seen_unseen_gap_records(
    metrics_by_key: dict[tuple[str, str], dict[str, float]]
) -> tuple[list[tuple[str, float, float]], list[str]]:
    dataset_rank = {dataset: index for index, dataset in enumerate(PREFERRED_DATASET_ORDER)}
    datasets = sorted(
        {dataset for dataset, _split in metrics_by_key},
        key=lambda dataset: (dataset_rank.get(dataset, len(PREFERRED_DATASET_ORDER)), dataset),
    )
    records: list[tuple[str, float, float]] = []
    skipped: list[str] = []
    for dataset in datasets:
        seen_record = pick_dataset_family_key(metrics_by_key, dataset, "seen")
        unseen_record = pick_dataset_family_key(metrics_by_key, dataset, "unseen")
        if seen_record is None or unseen_record is None:
            skipped.append(f"{dataset} (requires both seen and unseen splits)")
            continue

        _seen_split, seen_metrics = seen_record
        _unseen_split, unseen_metrics = unseen_record
        required = ("sr", "spl")
        missing = [
            name
            for name in required
            if name not in seen_metrics or name not in unseen_metrics
        ]
        if missing:
            skipped.append(f"{dataset} (missing {', '.join(sorted(set(missing)))})")
            continue

        sr_gap = seen_metrics["sr"] - unseen_metrics["sr"]
        spl_gap = seen_metrics["spl"] - unseen_metrics["spl"]
        records.append((dataset, sr_gap, spl_gap))
    return records, skipped


def build_error_pair_records(
    metrics_by_key: dict[tuple[str, str], dict[str, float]],
    ordered_keys: Iterable[tuple[str, str]],
    labels: dict[tuple[str, str], str],
) -> tuple[list[tuple[str, float, float]], list[str]]:
    records: list[tuple[str, float, float]] = []
    skipped: list[str] = []
    for key in ordered_keys:
        metric_map = metrics_by_key.get(key, {})
        missing = [name for name in ("nav_error", "oracle_error") if name not in metric_map]
        if missing:
            skipped.append(f"{labels[key]} (missing {', '.join(missing)})")
            continue
        records.append((labels[key], metric_map["nav_error"], metric_map["oracle_error"]))
    return records, skipped


def lazy_import_matplotlib(style_name: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_warning = ""
    try:
        plt.style.use(style_name)
    except OSError:
        plt.style.use("default")
        style_warning = f"Matplotlib style {style_name!r} is unavailable; fell back to 'default'."

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    return plt, style_warning


def annotate_bars(ax, bars) -> None:
    ymin, ymax = ax.get_ylim()
    offset = max((ymax - ymin) * 0.015, 0.05)
    for bar in bars:
        height = bar.get_height()
        xpos = bar.get_x() + bar.get_width() / 2
        if height >= 0:
            ypos = height + offset
            valign = "bottom"
        else:
            ypos = height - offset
            valign = "top"
        ax.text(xpos, ypos, f"{height:.2f}", ha="center", va=valign, fontsize=9)


def style_axes(ax, labels: list[str], title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_bar_chart(
    plt,
    records: list[tuple[str, float]],
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
) -> None:
    labels = [label for label, _value in records]
    values = [value for _label, value in records]
    width = max(8.0, len(labels) * 1.25)
    fig, ax = plt.subplots(figsize=(width, 4.8))
    bars = ax.bar(list(range(len(labels))), values, color=color, width=0.62)
    style_axes(ax, labels, title, ylabel)
    annotate_bars(ax, bars)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grouped_bar_chart(
    plt,
    records: list[tuple[str, float, float]],
    title: str,
    ylabel: str,
    legend_labels: tuple[str, str],
    colors: tuple[str, str],
    output_path: Path,
) -> None:
    labels = [label for label, _left, _right in records]
    left_values = [left for _label, left, _right in records]
    right_values = [right for _label, _left, right in records]
    width = max(8.0, len(labels) * 1.35)
    fig, ax = plt.subplots(figsize=(width, 4.8))
    positions = list(range(len(labels)))
    bar_width = 0.36
    left_positions = [position - bar_width / 2 for position in positions]
    right_positions = [position + bar_width / 2 for position in positions]
    left_bars = ax.bar(left_positions, left_values, width=bar_width, label=legend_labels[0], color=colors[0])
    right_bars = ax.bar(
        right_positions,
        right_values,
        width=bar_width,
        label=legend_labels[1],
        color=colors[1],
    )
    style_axes(ax, labels, title, ylabel)
    annotate_bars(ax, left_bars)
    annotate_bars(ax, right_bars)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_plot_path(output_dir: Path, experiment_id: str, suffix: str) -> Path:
    return output_dir / f"{experiment_id}_{suffix}.png"


def summarize_gap_records(records: list[tuple[str, float]], gap_name: str) -> list[str]:
    if not records:
        return [f"{gap_name}: no plottable records."]
    largest_label, largest_value = max(records, key=lambda item: item[1])
    return [
        f"{gap_name}: max = {largest_label} ({largest_value:.2f}).",
        f"{gap_name}: used {', '.join(label for label, _value in records)}.",
    ]


def summarize_grouped_gap_records(records: list[tuple[str, float, float]]) -> list[str]:
    if not records:
        return ["Seen-unseen gap: no plottable datasets."]
    largest_sr = max(records, key=lambda item: item[1])
    largest_spl = max(records, key=lambda item: item[2])
    return [
        f"Seen-unseen gap: largest SR gap = {largest_sr[0]} ({largest_sr[1]:.2f}).",
        f"Seen-unseen gap: largest SPL gap = {largest_spl[0]} ({largest_spl[2]:.2f}).",
        f"Seen-unseen gap: used {', '.join(dataset for dataset, _sr, _spl in records)}.",
    ]


def summarize_error_pair_records(records: list[tuple[str, float, float]]) -> list[str]:
    if not records:
        return ["NE vs oracle_error: no plottable dataset-splits."]
    largest_gap = max(records, key=lambda item: item[1] - item[2])
    return [
        "NE vs oracle_error: largest degradation gap = "
        f"{largest_gap[0]} ({largest_gap[1] - largest_gap[2]:.2f} m).",
        f"NE vs oracle_error: used {', '.join(label for label, _nav, _oracle in records)}.",
    ]


def generate_plots(
    experiment_id: str,
    metrics_by_key: dict[tuple[str, str], dict[str, float]],
    ordered_keys: list[tuple[str, str]],
    output_dir: Path,
    style_name: str,
    dpi: int,
) -> tuple[list[PlotArtifact], list[str]]:
    labels = build_display_labels(ordered_keys)
    plt, style_warning = lazy_import_matplotlib(style_name)
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    warnings = [style_warning] if style_warning else []
    artifacts: list[PlotArtifact] = []

    osr_records, osr_skipped = build_gap_records(metrics_by_key, ordered_keys, labels, "oracle_sr", "sr")
    if osr_records:
        path = build_plot_path(output_dir, experiment_id, "osr_sr_gap_bar")
        plot_single_bar_chart(
            plt,
            osr_records,
            title="SAME Diagnostics: OSR-SR Gap",
            ylabel="oracle_sr - sr (pct points)",
            color=SERIES_COLORS["blue"],
            output_path=path,
        )
        artifacts.append(
            PlotArtifact(
                name="osr_sr_gap_bar",
                title="OSR-SR gap bar",
                output_path=path,
                used_labels=[label for label, _value in osr_records],
                skipped_items=osr_skipped,
                key_messages=summarize_gap_records(osr_records, "OSR-SR gap"),
            )
        )

    sr_spl_records, sr_spl_skipped = build_gap_records(metrics_by_key, ordered_keys, labels, "sr", "spl")
    if sr_spl_records:
        path = build_plot_path(output_dir, experiment_id, "sr_spl_gap_bar")
        plot_single_bar_chart(
            plt,
            sr_spl_records,
            title="SAME Diagnostics: SR-SPL Gap",
            ylabel="sr - spl (pct points)",
            color=SERIES_COLORS["orange"],
            output_path=path,
        )
        artifacts.append(
            PlotArtifact(
                name="sr_spl_gap_bar",
                title="SR-SPL gap bar",
                output_path=path,
                used_labels=[label for label, _value in sr_spl_records],
                skipped_items=sr_spl_skipped,
                key_messages=summarize_gap_records(sr_spl_records, "SR-SPL gap"),
            )
        )

    seen_unseen_records, seen_unseen_skipped = build_seen_unseen_gap_records(metrics_by_key)
    if seen_unseen_records:
        path = build_plot_path(output_dir, experiment_id, "seen_unseen_gap_bar")
        plot_grouped_bar_chart(
            plt,
            seen_unseen_records,
            title="SAME Diagnostics: Seen-Unseen Gap",
            ylabel="seen - unseen (pct points)",
            legend_labels=("SR gap", "SPL gap"),
            colors=(SERIES_COLORS["green"], SERIES_COLORS["red"]),
            output_path=path,
        )
        artifacts.append(
            PlotArtifact(
                name="seen_unseen_gap_bar",
                title="Seen-unseen gap bar",
                output_path=path,
                used_labels=[dataset for dataset, _sr, _spl in seen_unseen_records],
                skipped_items=seen_unseen_skipped,
                key_messages=summarize_grouped_gap_records(seen_unseen_records),
            )
        )

    error_pair_records, error_pair_skipped = build_error_pair_records(metrics_by_key, ordered_keys, labels)
    if error_pair_records:
        path = build_plot_path(output_dir, experiment_id, "ne_oracle_error_bar")
        plot_grouped_bar_chart(
            plt,
            error_pair_records,
            title="SAME Diagnostics: NE vs Oracle Error",
            ylabel="meters",
            legend_labels=("nav_error", "oracle_error"),
            colors=(SERIES_COLORS["blue"], SERIES_COLORS["orange"]),
            output_path=path,
        )
        artifacts.append(
            PlotArtifact(
                name="ne_oracle_error_bar",
                title="NE vs oracle_error bar",
                output_path=path,
                used_labels=[label for label, _nav, _oracle in error_pair_records],
                skipped_items=error_pair_skipped,
                key_messages=summarize_error_pair_records(error_pair_records),
            )
        )

    return artifacts, warnings


def write_summary(
    experiment_id: str,
    metrics_csv: Path,
    runs_csv: Path,
    artifacts: list[PlotArtifact],
    warnings: list[str],
    output_dir: Path,
) -> Path:
    summary_path = output_dir / f"{experiment_id}_baseline_diagnostics_summary.md"
    lines = [
        "# SAME Baseline Diagnostics Summary",
        "",
        f"- experiment_id: {experiment_id}",
        f"- generated_at: {dt.datetime.now().astimezone().isoformat()}",
        f"- metrics_csv: {repo_rel(metrics_csv)}",
        f"- runs_csv: {repo_rel(runs_csv)}",
        f"- generated_plots: {len(artifacts)}",
        "",
        "## Figures",
    ]
    for artifact in artifacts:
        lines.append(f"- {artifact.output_path.name}")
        lines.append(f"  used: {', '.join(artifact.used_labels) if artifact.used_labels else 'none'}")
        if artifact.skipped_items:
            lines.append(f"  skipped: {'; '.join(artifact.skipped_items)}")
        else:
            lines.append("  skipped: none")
    lines.append("")
    lines.append("## Key Signals")
    for artifact in artifacts:
        lines.append(f"### {artifact.title}")
        for message in artifact.key_messages:
            lines.append(f"- {message}")
        if artifact.skipped_items:
            lines.append(f"- Skipped: {'; '.join(artifact.skipped_items)}")
        else:
            lines.append("- Skipped: none")
        lines.append("")

    lines.append("## Warnings")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def print_run_report(artifacts: list[PlotArtifact], warnings: list[str], summary_path: Path) -> None:
    print(f"Generated {len(artifacts)} diagnostic plot(s).")
    for artifact in artifacts:
        print(f"- {artifact.output_path}")
        print(f"  used: {', '.join(artifact.used_labels) if artifact.used_labels else 'none'}")
        print(f"  skipped: {'; '.join(artifact.skipped_items) if artifact.skipped_items else 'none'}")
    print(f"Summary: {summary_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id.strip()
    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    runs_csv = Path(args.runs_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_key, metric_warnings = load_experiment_metrics(metrics_csv, experiment_id)
    run_order, run_warnings = load_run_order(runs_csv, experiment_id)
    ordered_keys = sort_dataset_split_keys(metrics_by_key, run_order)

    artifacts, plot_warnings = generate_plots(
        experiment_id=experiment_id,
        metrics_by_key=metrics_by_key,
        ordered_keys=ordered_keys,
        output_dir=output_dir,
        style_name=args.style,
        dpi=args.dpi,
    )
    if not artifacts:
        raise SystemExit("No plots were generated because none of the requested metric combinations were available.")

    warnings = metric_warnings + run_warnings + plot_warnings
    summary_path = write_summary(
        experiment_id=experiment_id,
        metrics_csv=metrics_csv,
        runs_csv=runs_csv,
        artifacts=artifacts,
        warnings=warnings,
        output_dir=output_dir,
    )
    print_run_report(artifacts, warnings, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
