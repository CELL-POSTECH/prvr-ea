#!/usr/bin/env bash
# Benchmark GMMFormer-v2 ActivityNet CLIP retrieval without changing normal eval.
set -Eeuo pipefail

[[ $# -ge 2 && $# -le 4 ]] || {
  echo "usage: $0 <origin|flat_full|ivf|ivf-gpu|hnsw> <gpu> [checkpoint] [max_queries: 0=all]" >&2
  exit 2
}

INDEX="$1"
GPU="$2"
ROOT="${PRVR_PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PRVR_PYTHON:-python}"
CKPT="${3:-$ROOT/all_prvr/GMMFormer_v2/results/clip/activitynet/gmmformer_v2/best.ckpt}"
LIMIT="${4:-0}"
OUT="$ROOT/experiments/ann_benchmark/GMMFormer_v2/act"
mkdir -p "$OUT"
EXTRA_ARGS=()
OUTPUT_INDEX="$INDEX"

case "$INDEX" in
  origin|flat_full|ivf|ivf-gpu|hnsw) ;;
  *) echo "invalid index: $INDEX (use origin, flat_full, ivf, ivf-gpu, or hnsw)" >&2; exit 2 ;;
esac

# FAISS GPU supports top-k selection only through k=2048.  The CPU-equivalent
# frame candidate depth is 2948, so GPU IVF uses a clearly labeled capped run.
if [[ "$INDEX" == "ivf-gpu" ]]; then
  EXTRA_ARGS+=(--ann_frame_raw_k 2048)
  OUTPUT_INDEX="ivf-gpu_frame2048"
fi

cd "$ROOT/all_prvr/GMMFormer_v2"

# IVF/HNSW use a query-independent bank built once.  ``origin`` intentionally
# re-encodes test contexts through the normal evaluator on every run.
# PRVR_PYTHON="$PY" "$ROOT/experiments/scripts/build_gmmformer_v2_context_bank.sh" act "$GPU" "$CKPT"

"$PY" src/main.py -d act_clip --gpu "$GPU" --eval --resume "$CKPT" \
  --ann_benchmark --ann_index "$INDEX" --ann_max_queries "$LIMIT" \
  "${EXTRA_ARGS[@]}" \
  --ann_output "$OUT/${OUTPUT_INDEX}_cross_branch_only.csv"
