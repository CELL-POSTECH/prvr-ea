#!/usr/bin/env bash
# Create top-10 CSVs from existing branch top-100 CSVs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../runner_lib.sh"

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/make_branch_top10_csv.sh [dataset|all] [method|all]

dataset: act | tvr | cha | msrvtt | all  (default: all)
method:  top100 method directory name, e.g. GMMFormer_v2, BOA, BGM_Net, or all

Input:
  experiments/branch_rank/top100/<method>/<dataset>.csv

Output:
  experiments/branch_rank/top100/<method>/<dataset>_top10.csv
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -le 2 ]] || { usage >&2; exit 2; }

DATASET="${1:-all}"
METHOD="${2:-all}"
case "$DATASET" in act|tvr|cha|msrvtt|all) ;; *) echo "unknown dataset: $DATASET" >&2; exit 2 ;; esac

exec "$PYTHON_BIN" "$SCRIPT_DIR/make_branch_top10_csv.py" \
  --input-root "$EXP_ROOT/branch_rank/top100" \
  --dataset "$DATASET" \
  --method "$METHOD" \
  --k 10
