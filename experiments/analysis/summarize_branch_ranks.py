#!/usr/bin/env python3
"""Merge per-query branch-rank CSVs into a compact model/dataset summary."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def ratio(values, predicate):
    total = 0
    matched = 0
    for value in values:
        total += 1
        matched += bool(predicate(value))
    return 100.0 * matched / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in sorted(Path(args.input_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = tuple(row[k] for k in ("model", "dataset", "feature", "checkpoint", "left_name", "right_name", "left_weight", "right_weight"))
                groups[key].append(row)

    fields = [
        "model", "dataset", "feature", "checkpoint", "left_name", "right_name", "left_weight", "right_weight", "queries",
        "exact_r1", "exact_r5", "exact_r10", "exact_r100", "exact_medr", "exact_meanr", "exact_mrr",
        "left_r1", "left_r5", "left_r10", "left_r100", "left_medr", "left_meanr", "left_mrr",
        "right_r1", "right_r5", "right_r10", "right_r100", "right_medr", "right_meanr", "right_mrr",
        "exact_beats_left_pct", "exact_beats_right_pct", "left_beats_exact_pct", "right_beats_exact_pct",
        "exact_ties_left_pct", "exact_ties_right_pct", "fusion_rescue_r1_pct", "fusion_rescue_r5_pct", "fusion_rescue_r10_pct",
        "top1_agreement_exact_left_pct", "top1_agreement_exact_right_pct", "left_delta_mean", "left_delta_median", "right_delta_mean", "right_delta_median",
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            exact = [int(r["rank_exact"]) for r in rows]
            left = [int(r["rank_left"]) for r in rows]
            right = [int(r["rank_right"]) for r in rows]
            left_delta = [int(r["delta_left"]) for r in rows]
            right_delta = [int(r["delta_right"]) for r in rows]
            model, dataset, feature, checkpoint, left_name, right_name, left_weight, right_weight = key
            row = dict(zip(fields[:9], [model, dataset, feature, checkpoint, left_name, right_name, left_weight, right_weight, len(rows)]))
            for prefix, ranks in (("exact", exact), ("left", left), ("right", right)):
                for k in (1, 5, 10, 100):
                    row[f"{prefix}_r{k}"] = f"{ratio(ranks, lambda r, k=k: r <= k):.6f}"
                row[f"{prefix}_medr"] = statistics.median(ranks)
                row[f"{prefix}_meanr"] = f"{statistics.mean(ranks):.6f}"
                row[f"{prefix}_mrr"] = f"{statistics.mean(1.0 / r for r in ranks):.6f}"
            row.update({
                "exact_beats_left_pct": f"{ratio(zip(exact, left), lambda p: p[0] < p[1]):.6f}",
                "exact_beats_right_pct": f"{ratio(zip(exact, right), lambda p: p[0] < p[1]):.6f}",
                "left_beats_exact_pct": f"{ratio(zip(exact, left), lambda p: p[1] < p[0]):.6f}",
                "right_beats_exact_pct": f"{ratio(zip(exact, right), lambda p: p[1] < p[0]):.6f}",
                "exact_ties_left_pct": f"{ratio(zip(exact, left), lambda p: p[0] == p[1]):.6f}",
                "exact_ties_right_pct": f"{ratio(zip(exact, right), lambda p: p[0] == p[1]):.6f}",
                "top1_agreement_exact_left_pct": f"{ratio(rows, lambda r: r['exact_top1_video_id'] == r['left_top1_video_id']):.6f}",
                "top1_agreement_exact_right_pct": f"{ratio(rows, lambda r: r['exact_top1_video_id'] == r['right_top1_video_id']):.6f}",
                "left_delta_mean": f"{statistics.mean(left_delta):.6f}",
                "left_delta_median": statistics.median(left_delta),
                "right_delta_mean": f"{statistics.mean(right_delta):.6f}",
                "right_delta_median": statistics.median(right_delta),
            })
            for k in (1, 5, 10):
                row[f"fusion_rescue_r{k}_pct"] = f"{ratio(zip(exact, left, right), lambda p, k=k: p[0] <= k and p[1] > k and p[2] > k):.6f}"
            writer.writerow(row)


if __name__ == "__main__":
    main()
