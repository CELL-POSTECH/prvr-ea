#!/usr/bin/env python3
"""Plot raw depths required to obtain K unique videos per branch.

Reads ``per_query.csv.gz`` files produced by ``analyze_raw_dedup.sh``.  The
default input is the top-3000 raw-representation analysis and the default
metric is the raw depth at which the 20th distinct video is encountered.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402


METHOD_ORDER = [
    "GMMFormer", "GMMFormer_v2", "HLFormer", "DreamPRVR", "Holmes", "BOA",
    "MSC_PRVR", "DL_DKD", "MS_SL", "BGM_Net",
]
METHOD_DISPLAY = {
    "GMMFormer": "GMMFormer", "GMMFormer_v2": "GMMFormer-v2",
    "HLFormer": "HLFormer", "DreamPRVR": "DreamPRVR", "Holmes": "Holmes",
    "BOA": "BOA", "MSC_PRVR": "MSC-PRVR", "DL_DKD": "DL-DKD",
    "MS_SL": "MS-SL", "BGM_Net": "BGM-Net",
}
BRANCH_DISPLAY = {"clip": "Clip", "frame": "Frame", "inheritance": "Inheritance", "exploration": "Exploration"}


def percentile(values: list[int], pct: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def tick_steps(maximum: int) -> tuple[int, int]:
    if maximum <= 80:
        return 5, 1
    if maximum <= 160:
        return 10, 2
    if maximum <= 400:
        return 25, 5
    if maximum <= 800:
        return 50, 10
    if maximum <= 1600:
        return 100, 20
    return 500, 100


def read_depths(path: Path, field: str) -> tuple[list[int], int, str]:
    values: list[int] = []
    total = 0
    repr_per_video = ""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            repr_per_video = row.get("repr_per_video", repr_per_video)
            value = row.get(field, "")
            if value:
                values.append(int(float(value)))
    return values, total, repr_per_video


def write_summary(path: Path, data: dict[tuple[str, str], tuple[list[int], int, str]], dataset: str, unique_k: int) -> None:
    fields = ["dataset", "method", "branch", "unique_k", "repr_per_video", "num_queries", "covered_queries", "coverage_pct", "p95", "p99", "max"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for branch in ("clip", "frame", "inheritance", "exploration"):
                values, total, repr_count = data.get((method, branch), ([], 0, ""))
                if not total:
                    continue
                writer.writerow({
                    "dataset": dataset, "method": METHOD_DISPLAY[method],
                    "branch": BRANCH_DISPLAY.get(branch, branch), "unique_k": unique_k,
                    "repr_per_video": repr_count, "num_queries": total,
                    "covered_queries": len(values), "coverage_pct": 100.0 * len(values) / total,
                    "p95": percentile(values, 95) if values else "",
                    "p99": percentile(values, 99) if values else "",
                    "max": max(values) if values else "",
                })


def plot(dataset: str, unique_k: int, data: dict[tuple[str, str], tuple[list[int], int, str]], output: Path) -> None:
    methods = [method for method in METHOD_ORDER if any(data.get((method, branch), ([], 0, ""))[1] for branch in BRANCH_DISPLAY)]
    fig, axes = plt.subplots(len(methods), 2, figsize=(12, max(2.2 * len(methods), 5)), sharey=False)
    if len(methods) == 1:
        axes = [axes]
    for row, method in enumerate(methods):
        branches = ("inheritance", "exploration") if method == "DL_DKD" else ("clip", "frame")
        for col, branch in enumerate(branches):
            ax = axes[row][col]
            values, total, repr_count = data.get((method, branch), ([], 0, ""))
            if not values:
                ax.set_axis_off()
                continue
            maximum = max(values)
            bins = min(80, max(10, maximum - min(values) + 1))
            ax.hist(values, bins=bins, color="#4C78A8" if col == 0 else "#F58518", alpha=0.88)
            p95, p99 = percentile(values, 95), percentile(values, 99)
            ax.axvline(p95, color="black", linestyle="--", linewidth=1)
            ax.axvline(p99, color="#7B2CBF", linestyle="-.", linewidth=1)
            ax.axvline(maximum, color="red", linestyle=":", linewidth=1)
            coverage = 100.0 * len(values) / total
            ax.set_title(
                f"{METHOD_DISPLAY[method]} / {BRANCH_DISPLAY[branch]}  "
                f"p95={p95:.0f}, p99={p99:.0f}, max={maximum}, cov={coverage:.1f}%",
                fontsize=9,
            )
            ax.set_xlabel(f"raw_depth_unique_{unique_k}")
            ax.set_ylabel("frequency")
            major, minor = tick_steps(maximum)
            right = int(math.ceil(maximum / major) * major)
            # Keep a constant-valued distribution (for example, depth=20 for
            # one-representation-per-video branches) visible instead of
            # clipping its only histogram bin on the right border.
            if right <= maximum:
                right += major
            ax.set_xlim(0, right)
            ax.set_xticks(range(0, right + 1, major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
            ax.tick_params(axis="x", labelsize=7, rotation=35)
            ax.grid(which="major", alpha=0.25)
            ax.grid(which="minor", axis="x", alpha=0.12, linestyle=":")
            if row == 0:
                ax.text(0.5, 1.28, "Clip branch" if col == 0 else "Frame branch",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.suptitle(f"{dataset}: raw top-L needed to obtain {unique_k} unique videos", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    exp_root = Path(os.environ.get("PRVR_EXP_ROOT", project_root / "experiments"))
    parser.add_argument("--input-root", type=Path, default=exp_root / "branch_rank" / "full_repr_top3000")
    parser.add_argument("--output-root", type=Path, default=exp_root / "branch_rank" / "full_repr_top3000" / "reports")
    parser.add_argument("--datasets", nargs="+", default=["act", "tvr"])
    parser.add_argument("--unique-k", type=int, default=20)
    args = parser.parse_args()
    if args.unique_k < 1:
        raise ValueError("--unique-k must be positive")

    field = f"raw_depth_unique_{args.unique_k}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        data: dict[tuple[str, str], tuple[list[int], int, str]] = {}
        for method in METHOD_ORDER:
            source = args.input_root / method / dataset / "per_query.csv.gz"
            if not source.exists():
                continue
            values, total, repr_count = read_depths(source, field)
            with gzip.open(source, "rt", newline="", encoding="utf-8") as handle:
                branches = {row["branch"] for row in csv.DictReader(handle)}
            for branch in branches:
                branch_values, branch_total, branch_repr = read_depths_for_branch(source, field, branch)
                data[(method, branch)] = (branch_values, branch_total, branch_repr)
        png = args.output_root / f"{dataset}_raw_unique{args.unique_k}_depth_distribution.png"
        summary = args.output_root / f"{dataset}_raw_unique{args.unique_k}_depth_distribution_summary.csv"
        plot(dataset, args.unique_k, data, png)
        write_summary(summary, data, dataset, args.unique_k)
        print(png)
        print(summary)


def read_depths_for_branch(path: Path, field: str, branch: str) -> tuple[list[int], int, str]:
    values: list[int] = []
    total = 0
    repr_per_video = ""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["branch"] != branch:
                continue
            total += 1
            repr_per_video = row.get("repr_per_video", repr_per_video)
            value = row.get(field, "")
            if value:
                values.append(int(float(value)))
    return values, total, repr_per_video


if __name__ == "__main__":
    main()
