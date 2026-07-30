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
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402


METHOD_ORDER = [
    "GMMFormer", "GMMFormer_v2", "HLFormer", "DreamPRVR", "Holmes", "BOA",
    "MSC_PRVR", "DL_DKD", "MS_SL", "BGM_Net",
]
METHOD_DISPLAY = {
    "GMMFormer": "GMMFormer", "GMMFormer_v2": "GMMFormer v2",
    "HLFormer": "HLFormer", "DreamPRVR": "DreamPRVR", "Holmes": "Holmes",
    "BOA": "BOA", "MSC_PRVR": "MSC-PRVR", "DL_DKD": "DL-DKD",
    "MS_SL": "MS-SL", "BGM_Net": "BGM-Net",
}
BRANCH_DISPLAY = {"clip": "Clip", "frame": "Frame", "inheritance": "Inheritance", "exploration": "Exploration"}
OUTPUT_DPI = 180


def percentile(values: list[int], pct: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def tick_steps(maximum: int, sparse: bool = False) -> tuple[int, int]:
    if sparse:
        if maximum <= 80:
            return 10, 2
        if maximum <= 400:
            return 50, 10
        if maximum <= 800:
            return 100, 20
        if maximum <= 1600:
            return 250, 50
        return 500, 100
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


def write_summary(
    path: Path,
    data: dict[tuple[str, str], tuple[list[int], int, str]],
    dataset: str,
    unique_k: int,
    methods: list[str],
) -> None:
    fields = ["dataset", "method", "branch", "unique_k", "repr_per_video", "num_queries", "covered_queries", "coverage_pct", "p95", "p99", "max"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in methods:
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


def plot(
    dataset: str,
    unique_k: int,
    data: dict[tuple[str, str], tuple[list[int], int, str]],
    output: Path,
    methods: list[str],
    figure_width: float,
    row_height: float,
    font_size_px: float | None,
    tick_font_size_px: float | None,
    branch_header_font_size_px: float | None,
    panel_title_font_size_px: float | None,
    row_method_label: bool,
    wrap_panel_title: bool,
    sparse_x_ticks: bool,
    show_suptitle: bool,
) -> None:
    methods = [method for method in methods if any(data.get((method, branch), ([], 0, ""))[1] for branch in BRANCH_DISPLAY)]
    if not methods:
        raise ValueError("No selected methods had usable raw-depth data")
    compact = figure_width <= 4.0
    font_size_pt = None if font_size_px is None else font_size_px * 72.0 / OUTPUT_DPI
    tick_font_size_pt = None if tick_font_size_px is None else tick_font_size_px * 72.0 / OUTPUT_DPI
    branch_header_font_size_pt = (
        None if branch_header_font_size_px is None else branch_header_font_size_px * 72.0 / OUTPUT_DPI
    )
    panel_title_font_size_pt = (
        None if panel_title_font_size_px is None else panel_title_font_size_px * 72.0 / OUTPUT_DPI
    )
    if row_method_label:
        # Each method gets an explicit header row spanning both columns.  This
        # establishes the intended visual hierarchy:
        # method -> per-branch statistics -> two histograms.
        fig = plt.figure(figsize=(figure_width, max(row_height * len(methods), 2.2)))
        # Outer spacing separates method groups.  Each group then has its own
        # compact method-header/statistics/histogram sub-layout.
        # Together with the paper figure's row height, this preserves a
        # 20-pixel clear gap between neighbouring method/statistics/histogram
        # groups without making the histograms excessively short.
        outer_grid = fig.add_gridspec(nrows=len(methods), ncols=1, hspace=0.25)
        axes = []
        for row, method in enumerate(methods):
            group_grid = outer_grid[row].subgridspec(
                nrows=2,
                ncols=2,
                # Reserve a little more header room so the Method label is
                # clearly separated from its statistics and histogram.
                height_ratios=(0.342, 1.0),
                hspace=0.12,
                wspace=0.24,
            )
            method_ax = fig.add_subplot(group_grid[0, :])
            method_ax.set_axis_off()
            # Align each Method label with a consistent offset from its own
            # histogram pair.
            method_lift_px = 16
            method_ax.text(
                0.0,
                0.26,
                METHOD_DISPLAY[method],
                ha="left",
                va="bottom",
                fontsize=font_size_pt if font_size_pt is not None else 9,
                transform=method_ax.transAxes + ScaledTranslation(
                    0, method_lift_px / OUTPUT_DPI, fig.dpi_scale_trans
                ),
            )
            axes.append((
                fig.add_subplot(group_grid[1, 0]),
                fig.add_subplot(group_grid[1, 1]),
            ))
    else:
        fig, axes = plt.subplots(
            len(methods), 2,
            figsize=(figure_width, max(row_height * len(methods), 2.2)),
            sharey=False,
            squeeze=False,
        )
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
            # ``stepfilled`` emits one continuous filled path in the PDF.  This
            # avoids hairline seams between adjacent vector bar rectangles in
            # PDF viewers while preserving the histogram geometry.
            ax.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                color="#9BC3ED" if col == 0 else "#F5BF9E",
                edgecolor="none",
                linewidth=0,
                alpha=0.88,
            )
            p95, p99 = percentile(values, 95), percentile(values, 99)
            ax.axvline(p95, color="black", linestyle="--", linewidth=1.25)
            ax.axvline(p99, color="#7B2CBF", linestyle="-.", linewidth=1.25)
            ax.axvline(maximum, color="red", linestyle=":", linewidth=1.25)
            coverage = 100.0 * len(values) / total
            stats_title = f"p95/p99 = {p95:.0f}/{p99:.0f}, max={maximum}"
            if row_method_label:
                # The paper layout places these three markers in one shared
                # legend below the final row instead of repeating statistics
                # over every panel.
                title = None
            elif wrap_panel_title:
                title = (
                    f"{METHOD_DISPLAY[method]}\n"
                    f"p95={p95:.0f}, p99={p99:.0f}\n"
                    f"max={maximum}"
                )
            elif compact:
                title = (
                    f"{METHOD_DISPLAY[method]}\n"
                    f"{stats_title}"
                )
            else:
                title = (
                    f"{METHOD_DISPLAY[method]}  "
                    f"{stats_title}"
                )
            default_font_size = 5.6 if compact else 9
            size = font_size_pt if font_size_pt is not None else default_font_size
            title_x = 0.5
            if title:
                ax.set_title(
                    title,
                    fontsize=panel_title_font_size_pt if panel_title_font_size_pt is not None else size,
                    pad=4 if compact else 8,
                    x=title_x,
                )
            ax.set_xlabel("" if row_method_label else "Retrieval depth", fontsize=size)
            ax.set_ylabel("" if row_method_label else ("Frequency" if col == 0 else ""), fontsize=size)
            major, minor = tick_steps(maximum, sparse_x_ticks)
            right = int(math.ceil(maximum / major) * major)
            # Keep a constant-valued distribution (for example, depth=20 for
            # one-representation-per-video branches) visible instead of
            # clipping its only histogram bin on the right border.
            if right <= maximum:
                right += major
            ax.set_xlim(0, right)
            ax.set_xticks(range(0, right + 1, major))
            # Use a consistent 500-count reference on ordinary panels rather
            # than the automatic 250 tick. The one degenerate single-depth
            # panel can exceed 15k counts, where 500-step labels would be
            # unreadable, so it keeps Matplotlib's coarse tick selection.
            y_top = ax.get_ylim()[1]
            if y_top <= 1500:
                y_top = max(500, y_top)
                ax.set_ylim(0, y_top)
                ax.set_yticks(range(0, int(math.ceil(y_top / 500.0)) * 500 + 1, 500))
            if sparse_x_ticks:
                ax.minorticks_off()
            else:
                ax.xaxis.set_minor_locator(MultipleLocator(minor))
            ax.tick_params(
                axis="both",
                labelsize=tick_font_size_pt if tick_font_size_pt is not None else (
                    font_size_pt if font_size_pt is not None else (5.5 if compact else 7)
                ),
            )
            ax.tick_params(axis="x", rotation=35)
            # Keep the axis spines and ticks, but omit interior grid lines for
            # the paper figure.
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    # Keep the first Method row clear of the figure-level branch headers.
    top_margin = 0.930 if (not compact and row_method_label) else (
        0.90 if not compact else (0.97 if show_suptitle else 1)
    )
    if not compact:
        header_size = (
            branch_header_font_size_pt if branch_header_font_size_pt is not None
            else (font_size_pt if font_size_pt is not None else 11)
        )
        # Figure-level headers reserve their own top margin, so large headers
        # cannot overlap the first row's multi-line panel titles.
        fig.text(0.29, 0.982, "Clip-level branch", ha="center", va="top", fontsize=header_size, fontweight="bold")
        fig.text(0.74, 0.982, "Frame-level branch", ha="center", va="top", fontsize=header_size, fontweight="bold")
    if row_method_label:
        # Keep the shared marker key beside the frame-branch heading, as in
        # the two-column paper layout, rather than consuming a full row.
        legend_size = tick_font_size_pt if tick_font_size_pt is not None else 8
        fig.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.25, label="p95"),
                Line2D([0], [0], color="#7B2CBF", linestyle="-.", linewidth=1.25, label="p99"),
                Line2D([0], [0], color="red", linestyle=":", linewidth=1.25, label="max"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.786, 0.955),
            ncol=3,
            frameon=False,
            fontsize=legend_size,
            handlelength=2.0,
            columnspacing=1.35,
            handletextpad=0.45,
        )
        # A shared x-axis title keeps the two-branch layout compact.
        fig.text(
            (0.10 + 0.965) / 2,
            0.035,
            "Retrieval depth",
            ha="center",
            va="bottom",
            fontsize=font_size_pt if font_size_pt is not None else 9,
        )
        fig.text(
            0.035,
            (0.115 + top_margin) / 2,
            "Frequency",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=font_size_pt if font_size_pt is not None else 9,
        )
    if show_suptitle:
        fig.suptitle(
            f"{dataset}: raw top-L needed to obtain {unique_k} unique videos",
            y=0.995,
            fontsize=font_size_pt if font_size_pt is not None else (9 if compact else 14),
        )
    if row_method_label:
        # Reserve a dedicated strip below the final DL-DKD axes for the
        # shared percentile-marker legend.
        fig.subplots_adjust(left=0.10, right=0.965, bottom=0.115, top=top_margin)
    else:
        fig.tight_layout(rect=(0, 0, 1, top_margin), h_pad=1.05, w_pad=1.05)
    fig.savefig(output, dpi=OUTPUT_DPI)
    fig.savefig(output.with_suffix(".pdf"), format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    exp_root = Path(os.environ.get("PRVR_EXP_ROOT", project_root / "experiments"))
    parser.add_argument("--input-root", type=Path, default=exp_root / "branch_rank" / "full_repr_top3000")
    parser.add_argument("--output-root", type=Path, default=exp_root / "branch_rank" / "full_repr_top3000" / "reports")
    parser.add_argument("--datasets", nargs="+", default=["act", "tvr"])
    parser.add_argument("--unique-k", type=int, default=20)
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_ORDER,
        help="Subset and ordering of methods to plot (default: all methods).",
    )
    parser.add_argument("--figure-width", type=float, default=12.0, help="Figure width in inches.")
    parser.add_argument("--row-height", type=float, default=2.2, help="Height per method row in inches.")
    parser.add_argument("--font-path", type=Path, help="Optional TrueType/OpenType font to use for all text.")
    parser.add_argument(
        "--font-size-px", type=float,
        help=f"Font size in output pixels (PNG is rendered at {OUTPUT_DPI} dpi).",
    )
    parser.add_argument(
        "--tick-font-size-px", type=float,
        help="Optional x/y tick-label size in output pixels (defaults to --font-size-px).",
    )
    parser.add_argument(
        "--branch-header-font-size-px", type=float,
        help="Optional Clip-level/Frame-level header size in output pixels.",
    )
    parser.add_argument(
        "--panel-title-font-size-px", type=float,
        help="Optional method/statistics panel-title size in output pixels.",
    )
    parser.add_argument(
        "--row-method-label", action="store_true",
        help="Show one model name centred above each two-branch row.",
    )
    parser.add_argument(
        "--wrap-panel-title", action="store_true",
        help="Wrap method, p95/p99, and max onto separate title lines.",
    )
    parser.add_argument("--sparse-x-ticks", action="store_true", help="Show only sparse major x-axis ticks.")
    parser.add_argument("--no-suptitle", action="store_true", help="Omit the overall figure title.")
    args = parser.parse_args()
    if args.unique_k < 1:
        raise ValueError("--unique-k must be positive")
    if args.figure_width <= 0 or args.row_height <= 0:
        raise ValueError("--figure-width and --row-height must be positive")
    if args.font_size_px is not None and args.font_size_px <= 0:
        raise ValueError("--font-size-px must be positive")
    if args.tick_font_size_px is not None and args.tick_font_size_px <= 0:
        raise ValueError("--tick-font-size-px must be positive")
    if args.branch_header_font_size_px is not None and args.branch_header_font_size_px <= 0:
        raise ValueError("--branch-header-font-size-px must be positive")
    if args.panel_title_font_size_px is not None and args.panel_title_font_size_px <= 0:
        raise ValueError("--panel-title-font-size-px must be positive")
    if args.font_path:
        if not args.font_path.is_file():
            raise FileNotFoundError(f"Font file not found: {args.font_path}")
        font_manager.fontManager.addfont(str(args.font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=args.font_path).get_name()
        plt.rcParams["pdf.fonttype"] = 42

    field = f"raw_depth_unique_{args.unique_k}"
    methods = args.methods or METHOD_ORDER
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        data: dict[tuple[str, str], tuple[list[int], int, str]] = {}
        for method in methods:
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
        plot(
            dataset, args.unique_k, data, png, methods,
            args.figure_width, args.row_height,
            args.font_size_px, args.tick_font_size_px, args.branch_header_font_size_px,
            args.panel_title_font_size_px,
            args.row_method_label,
            args.wrap_panel_title,
            args.sparse_x_ticks,
            not args.no_suptitle,
        )
        write_summary(summary, data, dataset, args.unique_k, methods)
        print(png)
        print(png.with_suffix(".pdf"))
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
