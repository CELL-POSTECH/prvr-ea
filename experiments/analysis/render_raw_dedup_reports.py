#!/usr/bin/env python3
"""Render compact, dataset-wise raw-dedup reports from per-method summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = [
    "GMMFormer", "GMMFormer-v2", "HLFormer", "DreamPRVR", "Holmes", "BOA",
    "MSC-PRVR", "DL-DKD", "MS-SL", "BGM-Net",
]
SAFE = {name: name.replace("-", "_") for name in METHOD_ORDER}


def read_summary(path: Path) -> dict[tuple[str, str, str], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["branch"], row["metric"], row["statistic"]): row["value"]
                for row in csv.DictReader(handle)}


def value(data: dict[tuple[str, str, str], str], branch: str, metric: str, stat: str) -> str:
    return data.get((branch, metric, stat), "")


def display_branch(model: str, branch: str) -> str:
    if model == "DL-DKD":
        return {"inheritance": "Inheritance", "exploration": "Exploration"}[branch]
    return branch.capitalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    columns = [
        "method", "branch", "repr_per_video", "raw_top_l",
        "dup_at_l5_mean", "dup_at_l5_p95", "dup_at_l5_max",
        "dup_at_l10_mean", "dup_at_l10_p95", "dup_at_l10_max",
        "dup_at_l20_mean", "dup_at_l20_p95", "dup_at_l20_max",
        "dup_at_l30_mean", "dup_at_l30_p95", "dup_at_l30_max",
        "l_at_unique5_min", "l_at_unique5_mean", "l_at_unique5_p50", "l_at_unique5_p90", "l_at_unique5_p95", "l_at_unique5_p99", "l_at_unique5_max",
        "l_at_unique10_min", "l_at_unique10_mean", "l_at_unique10_p50", "l_at_unique10_p90", "l_at_unique10_p95", "l_at_unique10_p99", "l_at_unique10_max",
        "l_at_unique20_min", "l_at_unique20_mean", "l_at_unique20_p50", "l_at_unique20_p90", "l_at_unique20_p95", "l_at_unique20_p99", "l_at_unique20_max", "l_at_unique20_coverage",
        "l_at_unique30_min", "l_at_unique30_mean", "l_at_unique30_p50", "l_at_unique30_p90", "l_at_unique30_p95", "l_at_unique30_p99", "l_at_unique30_max", "l_at_unique30_coverage",
        "video_rank_min", "video_rank_mean", "video_rank_median", "video_rank_p90", "video_rank_p95", "video_rank_p99", "video_rank_max",
        "video_r1", "video_r5", "video_r10",
        "gt_first_raw_rank_min", "gt_first_raw_rank_mean", "gt_first_raw_rank_median", "gt_first_raw_rank_p90", "gt_first_raw_rank_p95", "gt_first_raw_rank_p99", "gt_first_raw_rank_max",
    ]
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        summary = args.root / SAFE[method] / args.dataset / "summary.csv"
        if not summary.exists():
            continue
        data = read_summary(summary)
        branches = ("inheritance", "exploration") if method == "DL-DKD" else ("clip", "frame")
        for branch in branches:
            repr_count = value(data, branch, "raw_depth_for_unique_video_at_1", "p50")
            if not repr_count:
                continue
            rows.append({
                "method": method, "branch": display_branch(method, branch),
                "repr_per_video": next((row["repr_per_video"] for row in csv.DictReader(summary.open(newline="", encoding="utf-8"))
                                          if row["branch"] == branch), ""),
                "raw_top_l": next((row["raw_top_l"] for row in csv.DictReader(summary.open(newline="", encoding="utf-8"))
                                    if row["branch"] == branch), ""),
                "dup_at_l5_mean": value(data, branch, "duplicate_rate_at_l5", "mean"),
                "dup_at_l5_p95": value(data, branch, "duplicate_rate_at_l5", "p95"),
                "dup_at_l5_max": value(data, branch, "duplicate_rate_at_l5", "max"),
                "dup_at_l10_mean": value(data, branch, "duplicate_rate_at_l10", "mean"),
                "dup_at_l10_p95": value(data, branch, "duplicate_rate_at_l10", "p95"),
                "dup_at_l10_max": value(data, branch, "duplicate_rate_at_l10", "max"),
                "dup_at_l20_mean": value(data, branch, "duplicate_rate_at_l20", "mean"),
                "dup_at_l20_p95": value(data, branch, "duplicate_rate_at_l20", "p95"),
                "dup_at_l20_max": value(data, branch, "duplicate_rate_at_l20", "max"),
                "dup_at_l30_mean": value(data, branch, "duplicate_rate_at_l30", "mean"),
                "dup_at_l30_p95": value(data, branch, "duplicate_rate_at_l30", "p95"),
                "dup_at_l30_max": value(data, branch, "duplicate_rate_at_l30", "max"),
                **{f"l_at_unique{k}_{stat}": value(data, branch, f"raw_depth_for_unique_video_at_{k}", stat)
                   for k in (5, 10, 20, 30) for stat in ("min", "mean", "p50", "p90", "p95", "p99", "max")},
                "l_at_unique20_coverage": next((row["coverage_pct"] for row in csv.DictReader(summary.open(newline="", encoding="utf-8"))
                                                if row["branch"] == branch and row["metric"] == "raw_depth_for_unique_video_at_20" and row["statistic"] == "max"), ""),
                "l_at_unique30_coverage": next((row["coverage_pct"] for row in csv.DictReader(summary.open(newline="", encoding="utf-8"))
                                                if row["branch"] == branch and row["metric"] == "raw_depth_for_unique_video_at_30" and row["statistic"] == "max"), ""),
                **{f"video_rank_{stat}": value(data, branch, "video_rank", stat)
                   for stat in ("min", "mean", "median", "p90", "p95", "p99", "max")},
                **{f"video_r{k}": value(data, branch, f"video_recall_at_{k}", "value") for k in (1, 5, 10)},
                **{f"gt_first_raw_rank_{stat}": value(data, branch, "gt_first_raw_rank", stat)
                   for stat in ("min", "mean", "median", "p90", "p95", "p99", "max")},
            })
    report_dir = args.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{args.dataset}_branch_raw_dedup_detail.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
