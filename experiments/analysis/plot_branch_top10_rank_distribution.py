#!/usr/bin/env python3
"""Plot distributions of branch depths needed to cover exact top-10 videos.

Reads files produced by ``export_branch_top100.sh``:

  experiments/branch_rank/top100/<method>/<dataset>.csv

and writes:

  experiments/branch_rank/top100_reports/distribution_<dataset>.png
  experiments/branch_rank/top100_reports/frequency.csv
  experiments/branch_rank/top100_reports/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402


METHOD_ORDER = [
    "GMMFormer",
    "GMMFormer_v2",
    "HLFormer",
    "DreamPRVR",
    "Holmes",
    "BOA",
    "MSC_PRVR",
    "DL_DKD",
    "MS_SL",
    "BGM_Net",
]
METHOD_DISPLAY = {
    "GMMFormer": "GMMFormer",
    "GMMFormer_v2": "GMMFormerV2",
    "HLFormer": "HLFormer",
    "DreamPRVR": "DreamPRVR",
    "Holmes": "Holmes",
    "BOA": "BOA",
    "MSC_PRVR": "MSC-PRVR",
    "DL_DKD": "DL-DKD",
    "MS_SL": "MS-SL",
    "BGM_Net": "BGM-Net",
}
DATASET_ORDER = ["act", "tvr", "cha", "msrvtt"]
BRANCH_COLUMNS = {
    "clip": "exact_top10_clip_max_rank",
    "frame": "exact_top10_frame_max_rank",
}


def percentile(values: list[int], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def read_values(path: Path) -> dict[str, list[int]]:
    values = {branch: [] for branch in BRANCH_COLUMNS}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for branch, column in BRANCH_COLUMNS.items():
                if row.get(column):
                    values[branch].append(int(float(row[column])))
    return values


def write_frequency(path: Path, data: dict[tuple[str, str, str], list[int]]) -> None:
    fields = ["dataset", "method", "branch", "rank", "frequency", "pct", "num_queries"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                for branch in ("clip", "frame"):
                    values = data.get((dataset, method, branch), [])
                    if not values:
                        continue
                    counts = Counter(values)
                    total = len(values)
                    for rank in sorted(counts):
                        writer.writerow({
                            "dataset": dataset,
                            "method": METHOD_DISPLAY.get(method, method),
                            "branch": branch,
                            "rank": rank,
                            "frequency": counts[rank],
                            "pct": 100.0 * counts[rank] / total,
                            "num_queries": total,
                        })


def write_summary(path: Path, data: dict[tuple[str, str, str], list[int]]) -> None:
    fields = [
        "dataset", "method", "branch", "num_queries",
        "min", "mean", "p50", "p90", "p95", "p99", "max",
        "pct_le_10", "pct_le_20", "pct_le_50", "pct_le_100",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                for branch in ("clip", "frame"):
                    values = data.get((dataset, method, branch), [])
                    if not values:
                        continue
                    total = len(values)
                    writer.writerow({
                        "dataset": dataset,
                        "method": METHOD_DISPLAY.get(method, method),
                        "branch": branch,
                        "num_queries": total,
                        "min": min(values),
                        "mean": sum(values) / total,
                        "p50": percentile(values, 50),
                        "p90": percentile(values, 90),
                        "p95": percentile(values, 95),
                        "p99": percentile(values, 99),
                        "max": max(values),
                        "pct_le_10": 100.0 * sum(v <= 10 for v in values) / total,
                        "pct_le_20": 100.0 * sum(v <= 20 for v in values) / total,
                        "pct_le_50": 100.0 * sum(v <= 50 for v in values) / total,
                        "pct_le_100": 100.0 * sum(v <= 100 for v in values) / total,
                    })


def tick_steps(max_value: int) -> tuple[int, int]:
    """Return readable major/minor x tick spacing for rank-depth histograms."""
    if max_value <= 80:
        return 5, 1
    if max_value <= 160:
        return 10, 2
    if max_value <= 400:
        return 25, 5
    if max_value <= 800:
        return 50, 10
    return 100, 20


def plot_dataset(dataset: str, data: dict[tuple[str, str, str], list[int]], output: Path) -> None:
    available_methods = [
        method for method in METHOD_ORDER
        if data.get((dataset, method, "clip")) or data.get((dataset, method, "frame"))
    ]
    if not available_methods:
        return
    fig, axes = plt.subplots(
        nrows=len(available_methods), ncols=2,
        figsize=(12, max(2.2 * len(available_methods), 5)),
        sharey=False,
    )
    if len(available_methods) == 1:
        axes = [axes]
    branch_titles = {"clip": "exact top10 depth in clip-only rank", "frame": "exact top10 depth in frame-only rank"}
    for row_index, method in enumerate(available_methods):
        for col_index, branch in enumerate(("clip", "frame")):
            ax = axes[row_index][col_index]
            values = data.get((dataset, method, branch), [])
            if not values:
                ax.set_axis_off()
                continue
            max_value = max(values)
            bins = min(80, max(10, max_value - min(values) + 1))
            ax.hist(values, bins=bins, color="#4C78A8" if branch == "clip" else "#F58518", alpha=0.88)
            p95_value = percentile(values, 95)
            p99_value = percentile(values, 99)
            ax.axvline(p95_value, color="black", linestyle="--", linewidth=1, label="p95")
            ax.axvline(p99_value, color="#7B2CBF", linestyle="-.", linewidth=1, label="p99")
            ax.axvline(max_value, color="red", linestyle=":", linewidth=1, label="max")
            ax.set_title(
                f"{METHOD_DISPLAY.get(method, method)} / {branch}  "
                f"p95={p95_value:.0f}, p99={p99_value:.0f}, max={max_value}",
                fontsize=9,
            )
            ax.set_xlabel(BRANCH_COLUMNS[branch])
            ax.set_ylabel("frequency")
            major_step, minor_step = tick_steps(max_value)
            right = int(math.ceil(max_value / major_step) * major_step)
            ax.set_xlim(0, right)
            ax.set_xticks(list(range(0, right + 1, major_step)))
            ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
            ax.tick_params(axis="x", labelsize=7, rotation=35)
            ax.grid(which="major", alpha=0.25)
            ax.grid(which="minor", axis="x", alpha=0.12, linestyle=":")
            if row_index == 0:
                ax.text(0.5, 1.28, branch_titles[branch], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.suptitle(f"{dataset}: branch top-L needed to cover exact/base top-10", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    exp_root = Path(os.environ.get("PRVR_EXP_ROOT", project_root / "experiments"))
    parser.add_argument("--input-root", type=Path, default=exp_root / "branch_rank" / "top100")
    parser.add_argument("--output-root", type=Path, default=exp_root / "branch_rank" / "top100_reports")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    data: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for method_dir in sorted(args.input_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for csv_path in sorted(method_dir.glob("*.csv")):
            dataset = csv_path.stem
            values_by_branch = read_values(csv_path)
            for branch, values in values_by_branch.items():
                data[(dataset, method, branch)] = values

    write_frequency(args.output_root / "frequency.csv", data)
    write_summary(args.output_root / "summary.csv", data)
    for dataset in DATASET_ORDER:
        plot_dataset(dataset, data, args.output_root / f"distribution_{dataset}.png")
    print(args.output_root)


if __name__ == "__main__":
    main()
