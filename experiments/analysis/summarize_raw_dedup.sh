#!/usr/bin/env bash
# Render compact and full-statistics reports from completed raw-dedup summaries.
# This is CPU-only: it never loads a checkpoint or runs model inference.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

usage() {
    cat <<'EOF'
Usage: bash experiments/scripts/summarize_raw_dedup.sh <dataset|all>

Reads:
  experiments/branch_rank/full_repr/<method>/<dataset>/summary.csv

Writes, for each selected dataset:
  experiments/branch_rank/full_repr/reports/<dataset>_branch_raw_dedup_detail.csv
  experiments/branch_rank/full_repr/reports/<dataset>_branch_raw_dedup_all_stats.csv

The detail file contains the comparison-table fields (duplicate rates, raw
depth for unique top-5/top-10, video rank and raw GT-rank statistics). The
all-stats file additionally includes min/mean/median/p90/p95/p99/max for each
available metric and branch.
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -eq 1 ]] || { usage >&2; exit 2; }

DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }
ROOT="$EXP_ROOT/branch_rank/full_repr"
REPORT_DIR="$ROOT/reports"
mkdir -p "$REPORT_DIR"

if [[ "$DATASET" == all ]]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$DATASET")
fi

for dataset in "${DATASETS[@]}"; do
    "$PYTHON_BIN" "$SCRIPT_DIR/render_raw_dedup_reports.py" --root "$ROOT" --dataset "$dataset"
done

"$PYTHON_BIN" - "$ROOT" "${DATASETS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
datasets = sys.argv[2:]
methods = [
    "GMMFormer", "GMMFormer-v2", "HLFormer", "DreamPRVR", "Holmes", "BOA",
    "MSC-PRVR", "DL-DKD", "MS-SL", "BGM-Net",
]
safe = {method: method.replace("-", "_") for method in methods}

fields = ["method", "branch", "repr_per_video", "raw_top_l"]
for prefix in ("dup_at_l5", "dup_at_l10"):
    fields += [f"{prefix}_{stat}" for stat in ("min", "mean", "median", "p90", "p95", "p99", "max")]
for k in (5, 10):
    fields += [f"l_at_unique{k}_{stat}" for stat in ("min", "mean", "p50", "p90", "p95", "p99", "max")]
for metric in ("video_rank", "gt_first_raw_rank"):
    fields += [f"{metric}_{stat}" for stat in ("min", "mean", "median", "p90", "p95", "p99", "max")]
fields += ["video_r1", "video_r5", "video_r10"]

for dataset in datasets:
    output_rows = []
    for method in methods:
        summary = root / safe[method] / dataset / "summary.csv"
        if not summary.exists():
            continue
        with summary.open(newline="", encoding="utf-8") as handle:
            data = {(row["branch"], row["metric"], row["statistic"]): row
                    for row in csv.DictReader(handle)}
        branches = ("inheritance", "exploration") if method == "DL-DKD" else ("clip", "frame")
        for branch in branches:
            sample = next((row for (saved_branch, _, _), row in data.items()
                           if saved_branch == branch), None)
            if sample is None:
                continue
            row = {"method": method, "branch": branch,
                   "repr_per_video": sample["repr_per_video"], "raw_top_l": sample["raw_top_l"]}
            for l in (5, 10):
                for stat in ("min", "mean", "median", "p90", "p95", "p99", "max"):
                    row[f"dup_at_l{l}_{stat}"] = data.get(
                        (branch, f"duplicate_rate_at_l{l}", stat), {}).get("value", "")
                for stat in ("min", "mean", "p50", "p90", "p95", "p99", "max"):
                    row[f"l_at_unique{l}_{stat}"] = data.get(
                        (branch, f"raw_depth_for_unique_video_at_{l}", stat), {}).get("value", "")
            for metric in ("video_rank", "gt_first_raw_rank"):
                for stat in ("min", "mean", "median", "p90", "p95", "p99", "max"):
                    row[f"{metric}_{stat}"] = data.get((branch, metric, stat), {}).get("value", "")
            for k in (1, 5, 10):
                row[f"video_r{k}"] = data.get((branch, f"video_recall_at_{k}", "value"), {}).get("value", "")
            output_rows.append(row)
    output = root / "reports" / f"{dataset}_branch_raw_dedup_all_stats.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(output)
PY
