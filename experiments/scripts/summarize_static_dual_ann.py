#!/usr/bin/env python3
"""Merge static dual-branch ANN summaries into one ActivityNet CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ["GMMFormer", "GMMFormer-v2", "HLFormer", "DreamPRVR", "Holmes", "BOA", "MSC-PRVR", "DL-DKD"]
SAFE = {method: method.replace("-", "_") for method in METHODS}


def normalized(method: str, path: Path) -> dict:
    data = json.loads(path.read_text())
    # GMMFormer-v2 predates this shared benchmark and keeps its established
    # field names. Normalize both formats in the final report.
    get = lambda *keys: next((data[key] for key in keys if key in data), "")
    return {
        "method": method,
        "condition": path.stem.removesuffix("_summary").replace("_cross_branch_only", ""),
        "clip_search_ms": get("left_search_ms_mean", "clip_search_ms_mean"),
        "frame_search_ms": get("right_search_ms_mean", "frame_search_ms_mean"),
        "dedup_ms": get("dedup_ms_mean"),
        "clip_fetch_maxsim_ms": get("left_fetch_maxsim_ms_mean", "clip_fetch_maxsim_ms_mean"),
        "frame_fetch_maxsim_ms": get("right_fetch_maxsim_ms_mean", "frame_fetch_maxsim_ms_mean"),
        "fusion_top10_ms": get("fusion_top10_ms_mean"),
        "total_ms": get("total_ms_mean"),
        "r1": get("r1"), "r5": get("r5"), "r10": get("r10"),
        "num_queries": get("num_queries"),
        "backend": get("faiss_backend", "search_backend"),
        "left_k": get("left", "ann_clip_raw_k"),
        "right_k": get("right", "ann_frame_raw_k"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="act")
    parser.add_argument("--condition", default=None,
                        help="include only one requested condition (also omits old smoke/legacy files)")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for method in METHODS:
        directory = root / SAFE[method] / args.dataset
        if not directory.is_dir():
            continue
        for summary in sorted(directory.glob("*_summary.json")):
            row = normalized(method, summary)
            # A diagnostic run may carry an output suffix (for example
            # ``ivf-gpu-x2_rerun``).  Keep it out of the canonical merged
            # condition report unless explicitly summarized without a filter.
            if args.condition and row["condition"] != args.condition:
                continue
            rows.append(row)
    fields = ["method", "condition", "clip_search_ms", "frame_search_ms", "dedup_ms",
              "clip_fetch_maxsim_ms", "frame_fetch_maxsim_ms", "fusion_top10_ms", "total_ms",
              "r1", "r5", "r10", "num_queries", "backend", "left_k", "right_k"]
    suffix = f"_{args.condition}" if args.condition else ""
    output = root / f"static_dual_ann_{args.dataset}{suffix}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} benchmark summaries)")


if __name__ == "__main__":
    main()
