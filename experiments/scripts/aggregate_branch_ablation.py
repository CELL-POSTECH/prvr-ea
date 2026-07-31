#!/usr/bin/env python3
"""Merge per-model dual-branch ablation CSVs into dataset-level tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = [
    ("GMMFormer", "GMMFormer"),
    ("GMMFormer-v2", "GMMFormerV2"),
    ("HLFormer", "HLFormer"),
    ("DreamPRVR", "DreamPRVR"),
    ("Holmes", "Holmes"),
    ("BOA", "BOA"),
    ("MSC-PRVR", "MSC-PRVR"),
    ("DL-DKD", "DL-DKD"),
    ("MS-SL", "MS-SL"),
    ("BGM-Net", "BGMNet"),
]

FIELDS = [
    "Method",
    "Base",
    "Clip-level branch",
    "Frame-level branch",
    "Branch Mean Pooling",
    "Weighed Branch Mean Pooling",
]
LONG_FIELDS = ["method", "dataset", "condition", "r1", "r5", "r10", "num_queries", "checkpoint"]


def read_first_row(path: Path) -> dict[str, str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="branch_ablation/clip root")
    parser.add_argument("--dataset", required=True, choices=("act", "tvr"))
    args = parser.parse_args()

    root = Path(args.root)
    runs = root / "runs" / args.dataset
    summary_path = root / f"{args.dataset}.csv"
    long_path = root / f"{args.dataset}_long.csv"

    rows: list[dict[str, str]] = []
    long_rows: list[dict[str, str]] = []
    for internal_name, display_name in METHOD_ORDER:
        safe = internal_name.replace("-", "_")
        row = read_first_row(runs / f"{safe}.csv")
        if row:
            row["Method"] = display_name
            rows.append({field: row.get(field, "") for field in FIELDS})
        long_file = runs / f"{safe}_long.csv"
        if long_file.exists():
            with long_file.open(newline="", encoding="utf-8") as handle:
                for long_row in csv.DictReader(handle):
                    long_row["method"] = display_name
                    long_rows.append({field: long_row.get(field, "") for field in LONG_FIELDS})

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(long_rows)

    print(f"summary CSV: {summary_path}")
    print(f"long CSV:    {long_path}")


if __name__ == "__main__":
    main()
