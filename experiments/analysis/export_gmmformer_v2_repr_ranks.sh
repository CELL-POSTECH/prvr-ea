#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PRVR_PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/export_gmmformer_v2_repr_ranks.sh <dataset> <gpu> [num_queries] [branch]

dataset: act | tvr | cha | msrvtt
The default export samples 50 test queries with seed 9527 and writes raw
top-500 representations plus top-500 deduplicated video ranks. branch is
clip (default), frame, or exact. exact exports the original weighted fusion.
EOF
}

[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
case "$1" in act|tvr|cha|msrvtt) ;; *) usage >&2; exit 2 ;; esac
[[ "$2" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }
case "${4:-clip}" in clip|frame|exact) ;; *) echo "branch must be clip, frame, or exact" >&2; exit 2 ;; esac

exec "$PYTHON_BIN" "$SCRIPT_DIR/export_gmmformer_v2_repr_ranks.py" \
  --dataset "$1" --gpu "$2" --num-queries "${3:-50}" --branch "${4:-clip}" --raw-topk 500 --dedup-topk 500
