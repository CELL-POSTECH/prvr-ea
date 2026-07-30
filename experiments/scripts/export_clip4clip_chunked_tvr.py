#!/usr/bin/env python3
"""Upsert a CLIP4Clip TVR chunked zero-shot result from its log file."""

import argparse
import csv
import re
from pathlib import Path


FIELDS = [
    "method", "dataset", "max_frames", "chunk_size", "r1", "r5", "r10",
    "median_rank", "mean_rank", "log", "run_dir",
]


def parse_metrics(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(
        r"Text-to-Video:\s*.*?R@1:\s*([0-9.]+)\s*-\s*R@5:\s*([0-9.]+)"
        r"\s*-\s*R@10:\s*([0-9.]+)\s*-\s*Median R:\s*([0-9.]+)"
        r"\s*-\s*Mean R:\s*([0-9.]+)",
        text, flags=re.DOTALL,
    ))
    if not matches:
        raise RuntimeError(f"No completed single-GT Text-to-Video metrics in {log_path}")
    match = matches[-1]
    return [match.group(i) for i in range(1, 6)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--max-frames", default=128, type=int)
    args = parser.parse_args()

    r1, r5, r10, median_rank, mean_rank = parse_metrics(args.log)
    row = {
        "method": "CLIP4Clip", "dataset": "tvr", "max_frames": str(args.max_frames),
        "chunk_size": str(args.chunk_size), "r1": r1, "r5": r5, "r10": r10,
        "median_rank": median_rank, "mean_rank": mean_rank,
        "log": str(args.log), "run_dir": args.run_dir,
    }

    rows = []
    if args.csv.exists():
        with args.csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [old for old in rows if not (
        old.get("method") == row["method"] and old.get("dataset") == row["dataset"]
        and old.get("max_frames") == row["max_frames"] and old.get("chunk_size") == row["chunk_size"]
    )]
    rows.append(row)
    rows.sort(key=lambda item: int(item["chunk_size"]))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated {args.csv}: chunk_size={args.chunk_size}, R@1/5/10={r1}/{r5}/{r10}")


if __name__ == "__main__":
    main()
