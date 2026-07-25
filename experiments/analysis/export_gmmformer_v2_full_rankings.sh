#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PRVR_PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/export_gmmformer_v2_full_rankings.sh <dataset|all> <gpu> [top_l] [query_batch_size]

dataset: act | tvr | cha | msrvtt | all
top_l: number of rank-aligned exact/video and raw clip/frame rows per test query (default: 500)
query_batch_size: evaluation query batch size (default: 16; reduce it if GPU memory is constrained)

Outputs, one directory per dataset:
  experiments/branch_rank/full_repr/GMMFormer_v2_<dataset>_clip_top<top_l>/
EOF
}

[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
case "$1" in act|tvr|cha|msrvtt|all) ;; *) usage >&2; exit 2 ;; esac
[[ "$2" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }
[[ "${3:-500}" =~ ^[0-9]+$ && "${4:-16}" =~ ^[0-9]+$ ]] || { echo "top_l and query_batch_size must be positive integers" >&2; exit 2; }

datasets=("$1")
[[ "$1" != "all" ]] || datasets=(act tvr cha msrvtt)
for dataset in "${datasets[@]}"; do
  "$PYTHON_BIN" "$SCRIPT_DIR/export_gmmformer_v2_full_rankings.py" \
    --dataset "$dataset" --gpu "$2" --top-l "${3:-500}" --query-batch-size "${4:-16}"
done
