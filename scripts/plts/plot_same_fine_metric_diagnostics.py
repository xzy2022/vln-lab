#!/usr/bin/env python3
"""Plot SAME fine-metric diagnostics from fine_metrics/tables/fine_metrics_wide.csv.

Recommended usage:

    conda run -n plots python scripts/plts/plot_same_fine_metric_diagnostics.py \
      --fine-metrics-dir experiment_outputs/0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1/fine_metrics \
      --output-dir reports/artifacts/plots

The script generates:

- SR/OSR curves against instruction token length
- SR/OSR curves against action steps
- SR/OSR curves against move steps
- SR/OSR curves against move length in meters
- grouped bars for SR, OSR, SPL, and Oracle-SPL
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINE_METRICS_DIR = (
    REPO_ROOT
    / "experiment_outputs"
    / "0011_same_val_unseen_r2r_reverie_cvdn_soon_same_s0_v1"
    / "fine_metrics"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "artifacts" / "plots"

DATASET_ORDER = ["R2R", "REVERIE", "CVDN", "SOON"]
DATASET_COLORS = {
    "R2R": "#4C72B0",
    "REVERIE": "#55A868",
    "CVDN": "#C44E52",
    "SOON": "#DD8452",
}
METRIC_STYLES = {
    "SR": {"linestyle": "-", "marker": "o"},
    "OSR": {"linestyle": "--", "marker": "s"},
}
BAR_COLORS = {
    "SR": "#4C72B0",
    "OSR": "#DD8452",
    "SPL": "#55A868",
    "Oracle-SPL": "#C44E52",
}

CURVE_SPECS = [
    (
        "instruction_token_count",
        "common.instruction_token_count",
        "Instruction Token Length",
        "instruction token length",
    ),
    (
        "action_step_count",
        "common.action_step_count",
        "Action Steps",
        "action step count",
    ),
    (
        "move_step_count",
        "common.move_step_count",
        "Move Steps",
        "move step count",
    ),
    (
        "path_length_m",
        "common.path_length_m",
        "Move Length (m)",
        "move length (m)",
    ),
]

REQUIRED_COLUMNS = {
    "experiment_id",
    "dataset",
    "split",
    "common.instruction_token_count",
    "common.action_step_count",
    "common.move_step_count",
    "common.path_length_m",
    "official.final_success",
    "official.oracle_success",
    "official.spl",
    "official.oracle_path_length_m",
    "official.shortest_path_length_m",
}

EPS = 1e-12


@dataclass(frozen=True)
class FineMetricRow:
    experiment_id: str
    dataset: str
    split: str
    values: dict[str, str]


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    center: float
    label: str


@dataclass(frozen=True)
class CurvePoint:
    x_name: str
    dataset: str
    metric: str
    bin_label: str
    x_center: float
    n: int
    value_pct: float


@dataclass(frozen=True)
class SummaryRecord:
    dataset: str
    items: int
    sr_pct: float
    osr_pct: float
    spl_pct: float
    oracle_spl_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAME fine-metric diagnostics.")
    parser.add_argument(
        "--fine-metrics-dir",
        default=str(DEFAULT_FINE_METRICS_DIR),
        help="Path to an experiment fine_metrics directory.",
    )
    parser.add_argument(
        "--wide-csv",
        default=None,
        help="Optional direct path to fine_metrics_wide.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and summary files will be written.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=14,
        help="Number of global bins for each curve x-axis.",
    )
    parser.add_argument(
        "--min-bin-items",
        type=int,
        default=15,
        help="Minimum items per dataset/bin required to draw a curve point.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output PNG DPI.",
    )
    parser.add_argument(
        "--style",
        default="seaborn-v0_8-whitegrid",
        help="Matplotlib style name. Falls back to default if unavailable.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_rows(wide_csv: Path) -> list[FineMetricRow]:
    if not wide_csv.exists():
        raise FileNotFoundError(f"wide CSV not found: {wide_csv}")

    with wide_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(
                f"{repo_rel(wide_csv)} is missing required columns {missing}; "
                f"found columns: {sorted(fieldnames)}"
            )
        rows = [
            FineMetricRow(
                experiment_id=row["experiment_id"].strip(),
                dataset=row["dataset"].strip(),
                split=row["split"].strip(),
                values=row,
            )
            for row in reader
            if row.get("dataset", "").strip()
        ]

    if not rows:
        raise ValueError(f"No data rows found in {repo_rel(wide_csv)}")
    return rows


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    token = value.strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no"}:
        return False
    if token == "":
        return None
    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    result = float(value)
    if math.isfinite(result):
        return result
    return None


def ordered_datasets(rows: Iterable[FineMetricRow]) -> list[str]:
    available = {row.dataset for row in rows}
    ordered = [dataset for dataset in DATASET_ORDER if dataset in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def experiment_id_from_rows(rows: list[FineMetricRow]) -> str:
    ids = sorted({row.experiment_id for row in rows if row.experiment_id})
    if not ids:
        return "same_fine_metrics"
    if len(ids) == 1:
        return ids[0]
    return "mixed_experiments"


def success_value(row: FineMetricRow, metric: str) -> float | None:
    if metric == "SR":
        value = parse_bool(row.values.get("official.final_success"))
    elif metric == "OSR":
        value = parse_bool(row.values.get("official.oracle_success"))
    else:
        raise ValueError(f"Unsupported curve metric: {metric}")
    if value is None:
        return None
    return 1.0 if value else 0.0


def official_spl(row: FineMetricRow) -> float | None:
    return parse_float(row.values.get("official.spl"))


def oracle_spl(row: FineMetricRow) -> float | None:
    oracle_success = parse_bool(row.values.get("official.oracle_success"))
    if oracle_success is None:
        return None
    if not oracle_success:
        return 0.0

    shortest = parse_float(row.values.get("official.shortest_path_length_m"))
    oracle_path = parse_float(row.values.get("official.oracle_path_length_m"))
    if shortest is None or oracle_path is None:
        return None
    denominator = max(shortest, oracle_path)
    if denominator <= EPS:
        return 1.0
    return shortest / denominator


def build_bins(rows: list[FineMetricRow], x_column: str, bin_count: int) -> list[Bin]:
    values = sorted(
        value
        for row in rows
        if (value := parse_float(row.values.get(x_column))) is not None
    )
    if not values:
        raise ValueError(f"No numeric values available for {x_column}")

    lower = values[0]
    upper = values[-1]
    if abs(upper - lower) <= EPS:
        label = format_bin_label(lower)
        return [Bin(lower=lower, upper=upper, center=lower, label=label)]

    bin_count = max(1, bin_count)
    is_integer_axis = x_column != "common.path_length_m"
    if is_integer_axis:
        lower_int = math.floor(lower)
        upper_int = math.ceil(upper)
        width = max(1, math.ceil((upper_int - lower_int + 1) / bin_count))
        bins = []
        start = lower_int
        while start <= upper_int:
            stop = min(start + width, upper_int + 1)
            inclusive_stop = stop - 1
            if start == inclusive_stop:
                label = str(start)
            else:
                label = f"{start}-{inclusive_stop}"
            bins.append(
                Bin(
                    lower=float(start),
                    upper=float(stop),
                    center=(start + inclusive_stop) / 2,
                    label=label,
                )
            )
            start = stop
        return bins

    width = (upper - lower) / bin_count
    bins = []
    for index in range(bin_count):
        start = lower + index * width
        stop = upper if index == bin_count - 1 else lower + (index + 1) * width
        bins.append(
            Bin(
                lower=start,
                upper=stop,
                center=(start + stop) / 2,
                label=f"{format_bin_label(start)}-{format_bin_label(stop)}",
            )
        )
    return bins


def format_bin_label(value: float) -> str:
    if abs(value - round(value)) <= EPS:
        return str(int(round(value)))
    return f"{value:.1f}"


def find_bin(value: float, bins: list[Bin]) -> Bin | None:
    for index, item in enumerate(bins):
        if index == len(bins) - 1:
            if item.lower <= value <= item.upper:
                return item
        elif item.lower <= value < item.upper:
            return item
    return None


def aggregate_curve_points(
    rows: list[FineMetricRow],
    x_name: str,
    x_column: str,
    bin_count: int,
    min_bin_items: int,
) -> list[CurvePoint]:
    bins = build_bins(rows, x_column, bin_count)
    points: list[CurvePoint] = []

    for dataset in ordered_datasets(rows):
        dataset_rows = [row for row in rows if row.dataset == dataset]
        for metric in ("SR", "OSR"):
            grouped: dict[str, list[float]] = {item.label: [] for item in bins}
            centers = {item.label: item.center for item in bins}
            for row in dataset_rows:
                x_value = parse_float(row.values.get(x_column))
                y_value = success_value(row, metric)
                if x_value is None or y_value is None:
                    continue
                bin_item = find_bin(x_value, bins)
                if bin_item is None:
                    continue
                grouped[bin_item.label].append(y_value)

            for bin_item in bins:
                values = grouped[bin_item.label]
                if len(values) < min_bin_items:
                    continue
                points.append(
                    CurvePoint(
                        x_name=x_name,
                        dataset=dataset,
                        metric=metric,
                        bin_label=bin_item.label,
                        x_center=centers[bin_item.label],
                        n=len(values),
                        value_pct=100.0 * sum(values) / len(values),
                    )
                )

    return points


def build_summary_records(rows: list[FineMetricRow]) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for dataset in ordered_datasets(rows):
        dataset_rows = [row for row in rows if row.dataset == dataset]
        sr_values = [
            value
            for row in dataset_rows
            if (value := success_value(row, "SR")) is not None
        ]
        osr_values = [
            value
            for row in dataset_rows
            if (value := success_value(row, "OSR")) is not None
        ]
        spl_values = [
            value
            for row in dataset_rows
            if (value := official_spl(row)) is not None
        ]
        oracle_spl_values = [
            value
            for row in dataset_rows
            if (value := oracle_spl(row)) is not None
        ]
        records.append(
            SummaryRecord(
                dataset=dataset,
                items=len(dataset_rows),
                sr_pct=percent_mean(sr_values),
                osr_pct=percent_mean(osr_values),
                spl_pct=percent_mean(spl_values),
                oracle_spl_pct=percent_mean(oracle_spl_values),
            )
        )
    return records


def percent_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(values) / len(values)


def lazy_import_matplotlib(style_name: str, dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style_warning = ""
    try:
        plt.style.use(style_name)
    except OSError:
        plt.style.use("default")
        style_warning = f"Matplotlib style {style_name!r} is unavailable; fell back to default."

    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt, Line2D, style_warning


def plot_curve(
    plt,
    Line2D,
    points: list[CurvePoint],
    x_name: str,
    x_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for dataset in DATASET_ORDER:
        for metric in ("SR", "OSR"):
            series = [
                point
                for point in points
                if point.x_name == x_name and point.dataset == dataset and point.metric == metric
            ]
            if not series:
                continue
            series = sorted(series, key=lambda point: point.x_center)
            style = METRIC_STYLES[metric]
            ax.plot(
                [point.x_center for point in series],
                [point.value_pct for point in series],
                color=DATASET_COLORS[dataset],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.2,
                linewidth=1.8,
                alpha=0.94,
            )

    ax.set_title(f"SAME Fine Metrics: SR/OSR vs {x_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)

    dataset_handles = [
        Line2D([0], [0], color=DATASET_COLORS[dataset], lw=2.2, label=dataset)
        for dataset in DATASET_ORDER
        if any(point.dataset == dataset and point.x_name == x_name for point in points)
    ]
    metric_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=2.0,
            linestyle=METRIC_STYLES[metric]["linestyle"],
            marker=METRIC_STYLES[metric]["marker"],
            markersize=4.2,
            label=metric,
        )
        for metric in ("SR", "OSR")
    ]
    first_legend = ax.legend(handles=dataset_handles, title="Dataset", loc="lower left")
    ax.add_artist(first_legend)
    ax.legend(handles=metric_handles, title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_summary_bars(plt, records: list[SummaryRecord], output_path: Path) -> None:
    labels = [record.dataset for record in records]
    metrics = ["SR", "OSR", "SPL", "Oracle-SPL"]
    values_by_metric = {
        "SR": [record.sr_pct for record in records],
        "OSR": [record.osr_pct for record in records],
        "SPL": [record.spl_pct for record in records],
        "Oracle-SPL": [record.oracle_spl_pct for record in records],
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    positions = list(range(len(labels)))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    for metric, offset in zip(metrics, offsets):
        bars = ax.bar(
            [position + offset for position in positions],
            values_by_metric[metric],
            width=bar_width,
            label=metric,
            color=BAR_COLORS[metric],
        )
        annotate_bars(ax, bars)

    ax.set_title("SAME Fine Metrics: SR, OSR, SPL, Oracle-SPL")
    ax.set_ylabel("score (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 1.4, 98.0),
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def write_curve_points(path: Path, points: list[CurvePoint]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "x_name",
                "dataset",
                "metric",
                "bin_label",
                "x_center",
                "n",
                "value_pct",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "x_name": point.x_name,
                    "dataset": point.dataset,
                    "metric": point.metric,
                    "bin_label": point.bin_label,
                    "x_center": f"{point.x_center:.6g}",
                    "n": point.n,
                    "value_pct": f"{point.value_pct:.6f}",
                }
            )


def write_summary_csv(path: Path, records: list[SummaryRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "items",
                "sr_pct",
                "osr_pct",
                "spl_pct",
                "oracle_spl_pct",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset": record.dataset,
                    "items": record.items,
                    "sr_pct": f"{record.sr_pct:.6f}",
                    "osr_pct": f"{record.osr_pct:.6f}",
                    "spl_pct": f"{record.spl_pct:.6f}",
                    "oracle_spl_pct": f"{record.oracle_spl_pct:.6f}",
                }
            )


def write_markdown_summary(
    path: Path,
    experiment_id: str,
    wide_csv: Path,
    output_paths: list[Path],
    records: list[SummaryRecord],
    warnings: list[str],
    min_bin_items: int,
    bin_count: int,
) -> None:
    lines = [
        "# SAME Fine-Metric Plot Summary",
        "",
        f"- experiment_id: {experiment_id}",
        f"- generated_at: {dt.datetime.now().astimezone().isoformat()}",
        f"- source: {repo_rel(wide_csv)}",
        f"- bin_count: {bin_count}",
        f"- min_bin_items: {min_bin_items}",
        "",
        "## Figures",
    ]
    for output_path in output_paths:
        lines.append(f"- {repo_rel(output_path)}")

    lines.extend(["", "## Dataset Summary", ""])
    lines.append("| dataset | items | SR | OSR | SPL | Oracle-SPL |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for record in records:
        lines.append(
            f"| {record.dataset} | {record.items} | {record.sr_pct:.2f} | "
            f"{record.osr_pct:.2f} | {record.spl_pct:.2f} | {record.oracle_spl_pct:.2f} |"
        )

    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    fine_metrics_dir = Path(args.fine_metrics_dir).expanduser().resolve()
    wide_csv = (
        Path(args.wide_csv).expanduser().resolve()
        if args.wide_csv
        else fine_metrics_dir / "tables" / "fine_metrics_wide.csv"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(wide_csv)
    experiment_id = experiment_id_from_rows(rows)
    plt, Line2D, style_warning = lazy_import_matplotlib(args.style, args.dpi)
    warnings = [style_warning] if style_warning else []

    all_points: list[CurvePoint] = []
    output_paths: list[Path] = []
    for x_name, x_column, x_label, _file_label in CURVE_SPECS:
        points = aggregate_curve_points(
            rows=rows,
            x_name=x_name,
            x_column=x_column,
            bin_count=args.bin_count,
            min_bin_items=args.min_bin_items,
        )
        if not points:
            warnings.append(f"No curve points generated for {x_column}.")
            continue
        all_points.extend(points)
        output_path = output_dir / f"{experiment_id}_sr_osr_by_{x_name}.png"
        plot_curve(
            plt=plt,
            Line2D=Line2D,
            points=points,
            x_name=x_name,
            x_label=x_label,
            output_path=output_path,
        )
        output_paths.append(output_path)

    summary_records = build_summary_records(rows)
    bar_path = output_dir / f"{experiment_id}_sr_osr_spl_oracle_spl_bar.png"
    plot_summary_bars(plt, summary_records, bar_path)
    output_paths.append(bar_path)

    curve_points_path = output_dir / f"{experiment_id}_fine_metric_curve_points.csv"
    metric_summary_path = output_dir / f"{experiment_id}_fine_metric_summary.csv"
    markdown_summary_path = output_dir / f"{experiment_id}_fine_metric_plots_summary.md"
    write_curve_points(curve_points_path, all_points)
    write_summary_csv(metric_summary_path, summary_records)
    write_markdown_summary(
        path=markdown_summary_path,
        experiment_id=experiment_id,
        wide_csv=wide_csv,
        output_paths=output_paths + [curve_points_path, metric_summary_path],
        records=summary_records,
        warnings=warnings,
        min_bin_items=args.min_bin_items,
        bin_count=args.bin_count,
    )

    print(f"Generated {len(output_paths)} figure(s).")
    for output_path in output_paths:
        print(f"- {output_path}")
    print(f"Curve points: {curve_points_path}")
    print(f"Metric summary: {metric_summary_path}")
    print(f"Markdown summary: {markdown_summary_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
