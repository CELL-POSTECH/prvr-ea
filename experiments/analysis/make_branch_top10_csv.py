#!/usr/bin/env python3
"""Create compact top-10 CSVs from branch top-100 exports.

Input:
  experiments/branch_rank/top100/<method>/<dataset>.csv

Output:
  experiments/branch_rank/top100/<method>/<dataset>_top10.csv

The script does not run model evaluation. It only truncates pipe-separated
top-100 list columns to their first 10 entries and renames ``top100`` columns
to ``top10``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


LIST_COLUMNS = {
    "exact_top100_video_ids",
    "clip_top100_video_ids",
    "frame_top100_video_ids",
    "exact_top100_scores",
    "clip_top100_scores",
    "frame_top100_scores",
    "exact_top100_clip_ranks",
    "exact_top100_frame_ranks",
}


def convert_file(source: Path, output: Path, k: int = 10) -> int:
    with source.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {source}")
        rename = {column: column.replace("top100", f"top{k}") for column in LIST_COLUMNS}
        fields = [rename.get(field, field) for field in reader.fieldnames]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            rows = 0
            for row in reader:
                out = {}
                for field in reader.fieldnames:
                    value = row.get(field, "")
                    if field in LIST_COLUMNS and value:
                        value = "|".join(value.split("|")[:k])
                    out[rename.get(field, field)] = value
                writer.writerow(out)
                rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    exp_root = Path(os.environ.get("PRVR_EXP_ROOT", project_root / "experiments"))
    parser.add_argument("--input-root", type=Path, default=exp_root / "branch_rank" / "top100")
    parser.add_argument("--method", default="all", help="method directory name or all")
    parser.add_argument("--dataset", default="all", choices=("act", "tvr", "cha", "msrvtt", "all"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be positive")

    method_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.method != "all":
        method_dirs = [args.input_root / args.method]
    converted = 0
    for method_dir in method_dirs:
        if not method_dir.is_dir():
            print(f"skip missing method dir: {method_dir}")
            continue
        csv_files = sorted(method_dir.glob("*.csv"))
        csv_files = [path for path in csv_files if not path.stem.endswith(f"_top{args.k}")]
        if args.dataset != "all":
            csv_files = [method_dir / f"{args.dataset}.csv"]
        for source in csv_files:
            if not source.is_file():
                print(f"skip missing CSV: {source}")
                continue
            output = source.with_name(f"{source.stem}_top{args.k}.csv")
            rows = convert_file(source, output, args.k)
            print(f"{output} ({rows} rows)")
            converted += 1
    print(f"converted {converted} files")


if __name__ == "__main__":
    main()
