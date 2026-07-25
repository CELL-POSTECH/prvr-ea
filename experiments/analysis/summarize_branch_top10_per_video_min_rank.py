#!/usr/bin/env python3
"""Summarize per-video oracle branch ranks for exact top-10 videos.

For each exact top-10 video, compute:

  min(exact_top10_clip_rank, exact_top10_frame_rank)

Then summarize the worst case and percentiles by method/dataset.
Input files are produced by ``export_branch_top10.sh``:

  experiments/branch_rank/top10/<method>/<dataset>.csv

Outputs:

  experiments/branch_rank/top10_oracle_per_video_required_rank/summary.csv
  experiments/branch_rank/top10_oracle_per_video_required_rank/<dataset>_per_exact_video.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


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
METHOD_ORDER = list(METHOD_DISPLAY)
DATASET_ORDER = ["act", "tvr", "cha", "msrvtt"]


def percentile(values: list[int], pct: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    exp_root = Path(os.environ.get("PRVR_EXP_ROOT", project_root / "experiments"))
    parser.add_argument("--input-root", type=Path, default=exp_root / "branch_rank" / "top10")
    parser.add_argument("--output-root", type=Path, default=exp_root / "branch_rank" / "top10_oracle_per_video_required_rank")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    per_dataset_rows: dict[str, list[dict[str, object]]] = {dataset: [] for dataset in DATASET_ORDER}
    summary_rows: list[dict[str, object]] = []
    required = {"exact_top10_video_ids", "exact_top10_clip_ranks", "exact_top10_frame_ranks"}

    for method in METHOD_ORDER:
        method_dir = args.input_root / method
        for dataset in DATASET_ORDER:
            source = method_dir / f"{dataset}.csv"
            if not source.exists():
                summary_rows.append({
                    "dataset": dataset, "method": METHOD_DISPLAY[method], "status": "missing",
                    "source": str(source),
                })
                continue
            values: list[int] = []
            clip_best = frame_best = tie = n_queries = 0
            with source.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not required <= set(reader.fieldnames or []):
                    summary_rows.append({
                        "dataset": dataset, "method": METHOD_DISPLAY[method], "status": "missing_columns",
                        "source": str(source),
                    })
                    continue
                for row in reader:
                    n_queries += 1
                    videos = row["exact_top10_video_ids"].split("|")
                    clip_ranks = [int(float(item)) for item in row["exact_top10_clip_ranks"].split("|") if item]
                    frame_ranks = [int(float(item)) for item in row["exact_top10_frame_ranks"].split("|") if item]
                    for exact_rank, (video_id, clip_rank, frame_rank) in enumerate(
                        zip(videos, clip_ranks, frame_ranks), start=1
                    ):
                        min_rank = min(clip_rank, frame_rank)
                        if clip_rank < frame_rank:
                            best_branch = "clip"
                            clip_best += 1
                        elif frame_rank < clip_rank:
                            best_branch = "frame"
                            frame_best += 1
                        else:
                            best_branch = "tie"
                            tie += 1
                        values.append(min_rank)
                        per_dataset_rows[dataset].append({
                            "dataset": dataset,
                            "method": METHOD_DISPLAY[method],
                            "query_id": row.get("query_id", ""),
                            "exact_rank": exact_rank,
                            "video_id": video_id,
                            "clip_rank": clip_rank,
                            "frame_rank": frame_rank,
                            "min_branch_rank": min_rank,
                            "best_branch": best_branch,
                        })
            if not values:
                continue
            total = len(values)
            summary_rows.append({
                "dataset": dataset,
                "method": METHOD_DISPLAY[method],
                "status": "ok",
                "num_queries": n_queries,
                "num_exact_top10_videos": total,
                "per_video_min_rank_max": max(values),
                "per_video_min_rank_p99": percentile(values, 99),
                "per_video_min_rank_p95": percentile(values, 95),
                "per_video_min_rank_p90": percentile(values, 90),
                "per_video_min_rank_p50": percentile(values, 50),
                "per_video_min_rank_mean": sum(values) / total,
                "clip_best_pct": 100.0 * clip_best / total,
                "frame_best_pct": 100.0 * frame_best / total,
                "tie_pct": 100.0 * tie / total,
                "pct_le_10": 100.0 * sum(value <= 10 for value in values) / total,
                "pct_le_20": 100.0 * sum(value <= 20 for value in values) / total,
                "pct_le_50": 100.0 * sum(value <= 50 for value in values) / total,
                "source": str(source),
            })

    summary_fields = [
        "dataset", "method", "status", "num_queries", "num_exact_top10_videos",
        "per_video_min_rank_max", "per_video_min_rank_p99", "per_video_min_rank_p95",
        "per_video_min_rank_p90", "per_video_min_rank_p50", "per_video_min_rank_mean",
        "clip_best_pct", "frame_best_pct", "tie_pct",
        "pct_le_10", "pct_le_20", "pct_le_50", "source",
    ]
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    per_video_fields = [
        "dataset", "method", "query_id", "exact_rank", "video_id",
        "clip_rank", "frame_rank", "min_branch_rank", "best_branch",
    ]
    for dataset, rows in per_dataset_rows.items():
        with (args.output_root / f"{dataset}_per_exact_video.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_video_fields)
            writer.writeheader()
            writer.writerows(rows)
    print(args.output_root / "summary.csv")


if __name__ == "__main__":
    main()
