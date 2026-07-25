#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -ge 2 && $# -le 3 ]] || { echo "usage: $0 <dataset> <gpu> [checkpoint]" >&2; exit 2; }
DATASET="$1"; GPU="$2"
ROOT="${PRVR_PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PRVR_PYTHON:-python}"
CKPT="${3:-$ROOT/all_prvr/GMMFormer_v2/results/clip/activitynet/gmmformer_v2/best.ckpt}"
cd "$ROOT/all_prvr/GMMFormer_v2"
"$PY" src/main.py -d "${DATASET}_clip" --gpu "$GPU" --eval --resume "$CKPT" --ann_build_context_bank
