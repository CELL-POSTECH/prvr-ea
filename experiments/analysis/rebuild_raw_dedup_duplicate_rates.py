#!/usr/bin/env python3
"""Repair/rebuild duplicate-rate fields from existing raw-dedup diagnostics.

The raw-dedup collector stores the unique-video count at each raw checkpoint,
so duplicate rates can be reconstructed without re-running model evaluation.
This also corrects historical summaries produced before checkpoint collisions
were handled correctly for large-representation branches.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "method", "dataset", "branch", "metric", "statistic", "value",
    "repr_per_video", "raw_top_l", "num_queries", "coverage_pct",
]
UNIQUE_FIELD = re.compile(r"^unique_videos_at_l(\d+)$")


def percentile(values: list[float], pct: int) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    index = (len(values) - 1) * pct / 100
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(values: list[float]) -> list[tuple[str, float | str]]:
    if not values:
        return [(name, "") for name in ("min", "mean", "p50", "median", "p90", "p95", "p99", "max")]
    return [
        ("min", min(values)), ("mean", sum(values) / len(values)),
        ("p50", percentile(values, 50)), ("median", percentile(values, 50)),
        ("p90", percentile(values, 90)), ("p95", percentile(values, 95)),
        ("p99", percentile(values, 99)), ("max", max(values)),
    ]


def atomic_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with gzip.open(temp, "wt", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temp, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def repair(directory: Path) -> bool:
    per_query = directory / "per_query.csv.gz"
    summary = directory / "summary.csv"
    if not per_query.exists() or not summary.exists():
        return False
    with gzip.open(per_query, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    unique_fields = [(field, int(match.group(1))) for field in fields if (match := UNIQUE_FIELD.match(field))]
    if not rows or not unique_fields:
        return False
    rates: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        repr_count = int(row["repr_per_video"])
        raw_top_l = int(row["raw_top_l"])
        branch = row["branch"]
        for unique_field, k in unique_fields:
            value = row.get(unique_field, "")
            if not value:
                continue
            checkpoint = min((k - 1) * repr_count + 1, raw_top_l)
            rate = 100.0 * (checkpoint - float(value)) / checkpoint
            row[f"duplicate_rate_at_l{k}"] = str(rate)
            rates[(branch, k)].append(rate)
    atomic_gzip_csv(per_query, fields, rows)

    with summary.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary_rows = [row for row in summary_rows if not row["metric"].startswith("duplicate_rate_at_l")]
    sample = rows[0]
    for (branch, k), values in sorted(rates.items()):
        branch_rows = [row for row in rows if row["branch"] == branch]
        for statistic, value in stats(values):
            summary_rows.append({
                "method": sample["method"], "dataset": sample["dataset"], "branch": branch,
                "metric": f"duplicate_rate_at_l{k}", "statistic": statistic, "value": str(value),
                "repr_per_video": branch_rows[0]["repr_per_video"], "raw_top_l": branch_rows[0]["raw_top_l"],
                "num_queries": str(len(branch_rows)), "coverage_pct": "100.0",
            })
    atomic_csv(summary, summary_rows)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repaired = sum(repair(path.parent) for path in args.root.rglob("per_query.csv.gz"))
    print(f"repaired {repaired} raw-dedup result directories")


if __name__ == "__main__":
    main()
