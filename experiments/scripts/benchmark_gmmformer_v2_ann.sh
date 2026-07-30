#!/usr/bin/env bash
# Benchmark GMMFormer-v2 ActivityNet CLIP retrieval without changing normal eval.
set -Eeuo pipefail

[[ $# -ge 2 ]] || {
  echo "usage: $0 <origin|flat_full|ivf|ivf-gpu|hnsw> <gpu> [checkpoint] [max_queries: 0=all] [ANN options]" >&2
  exit 2
}

INDEX="$1"
GPU="$2"
shift 2
ROOT="${PRVR_PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PRVR_PYTHON:-python}"
CKPT="$ROOT/all_prvr/GMMFormer_v2/results/clip/activitynet/gmmformer_v2/best.ckpt"
LIMIT=0
CLIP_RAW_K=832
FRAME_RAW_K=2948
CLIP_NLIST=0
FRAME_NLIST=0
CLIP_NPROBE=64
FRAME_NPROBE=64
CLIP_EF_SEARCH=256
FRAME_EF_SEARCH=256
HNSW_M=128
HNSW_EF_CONSTRUCTION=256
POSITIONAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clip)
      CLIP_RAW_K="${2:?--clip requires a positive integer}"
      shift 2
      ;;
    --frame)
      FRAME_RAW_K="${2:?--frame requires a positive integer}"
      shift 2
      ;;
    --nlist)
      CLIP_NLIST="${2:?--nlist requires a non-negative integer}"
      FRAME_NLIST="$CLIP_NLIST"
      shift 2
      ;;
    --clip-nlist)
      CLIP_NLIST="${2:?--clip-nlist requires a non-negative integer}"
      shift 2
      ;;
    --frame-nlist)
      FRAME_NLIST="${2:?--frame-nlist requires a non-negative integer}"
      shift 2
      ;;
    --nprobe)
      CLIP_NPROBE="${2:?--nprobe requires a positive integer}"
      FRAME_NPROBE="$CLIP_NPROBE"
      shift 2
      ;;
    --clip-nprobe)
      CLIP_NPROBE="${2:?--clip-nprobe requires a positive integer}"
      shift 2
      ;;
    --frame-nprobe)
      FRAME_NPROBE="${2:?--frame-nprobe requires a positive integer}"
      shift 2
      ;;
    --ef-search)
      CLIP_EF_SEARCH="${2:?--ef-search requires a positive integer}"
      FRAME_EF_SEARCH="$CLIP_EF_SEARCH"
      shift 2
      ;;
    --clip-ef-search)
      CLIP_EF_SEARCH="${2:?--clip-ef-search requires a positive integer}"
      shift 2
      ;;
    --frame-ef-search)
      FRAME_EF_SEARCH="${2:?--frame-ef-search requires a positive integer}"
      shift 2
      ;;
    --hnsw-m)
      HNSW_M="${2:?--hnsw-m requires a positive integer}"
      shift 2
      ;;
    --hnsw-ef-construction)
      HNSW_EF_CONSTRUCTION="${2:?--hnsw-ef-construction requires a positive integer}"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
usage: $0 <origin|flat_full|ivf|ivf-gpu|hnsw> <gpu> [checkpoint] [max_queries: 0=all] [ANN options]

ANN options:
  --clip K --frame K                         raw output k for index.search
  --nlist L | --clip-nlist L --frame-nlist L IVF partitions (0=auto)
  --nprobe P | --clip-nprobe P --frame-nprobe P
  --ef-search E | --clip-ef-search E --frame-ef-search E
  --hnsw-m M --hnsw-ef-construction E
EOF
      exit 0
      ;;
    --*)
      echo "unknown option: $1" >&2
      exit 2
      ;;
    *)
      case "$POSITIONAL" in
        0) CKPT="$1" ;;
        1) LIMIT="$1" ;;
        *) echo "unexpected positional argument: $1" >&2; exit 2 ;;
      esac
      POSITIONAL=$((POSITIONAL + 1))
      shift
      ;;
  esac
done

[[ "$CLIP_RAW_K" =~ ^[1-9][0-9]*$ ]] || { echo "--clip must be a positive integer" >&2; exit 2; }
[[ "$FRAME_RAW_K" =~ ^[1-9][0-9]*$ ]] || { echo "--frame must be a positive integer" >&2; exit 2; }
for value in "$CLIP_NLIST" "$FRAME_NLIST"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "nlist values must be non-negative integers" >&2; exit 2; }
done
for value in "$CLIP_NPROBE" "$FRAME_NPROBE" "$CLIP_EF_SEARCH" "$FRAME_EF_SEARCH" "$HNSW_M" "$HNSW_EF_CONSTRUCTION"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ANN search parameters must be positive integers" >&2; exit 2; }
done
OUT="$ROOT/experiments/ann_benchmark/GMMFormer_v2/act"
mkdir -p "$OUT"
EXTRA_ARGS=()
OUTPUT_INDEX="$INDEX"

case "$INDEX" in
  origin|flat_full|ivf|ivf-gpu|hnsw) ;;
  *) echo "invalid index: $INDEX (use origin, flat_full, ivf, ivf-gpu, or hnsw)" >&2; exit 2 ;;
esac

# FAISS GPU supports top-k selection only through k=2048.  Cap either branch
# if needed, and encode the effective depths in the output filename.
if [[ "$INDEX" == "ivf-gpu" ]]; then
  if (( CLIP_RAW_K > 2048 )); then
    echo "ivf-gpu: clip raw K ${CLIP_RAW_K} exceeds FAISS GPU limit; using 2048" >&2
    CLIP_RAW_K=2048
  fi
  if (( FRAME_RAW_K > 2048 )); then
    echo "ivf-gpu: frame raw K ${FRAME_RAW_K} exceeds FAISS GPU limit; using 2048" >&2
    FRAME_RAW_K=2048
  fi
  OUTPUT_INDEX="ivf-gpu_frame2048"
fi
if (( CLIP_RAW_K != 832 || FRAME_RAW_K != 2948 || CLIP_NLIST != 0 || FRAME_NLIST != 0 || CLIP_NPROBE != 64 || FRAME_NPROBE != 64 || CLIP_EF_SEARCH != 256 || FRAME_EF_SEARCH != 256 || HNSW_M != 128 || HNSW_EF_CONSTRUCTION != 256 )); then
  OUTPUT_INDEX="${INDEX}_k${CLIP_RAW_K}-${FRAME_RAW_K}_nl${CLIP_NLIST}-${FRAME_NLIST}_np${CLIP_NPROBE}-${FRAME_NPROBE}_ef${CLIP_EF_SEARCH}-${FRAME_EF_SEARCH}"
fi
# The orchestrator uses stable condition names when it gathers the seven
# benchmark settings.  Standalone invocations retain the descriptive default.
[[ -z "${PRVR_ANN_OUTPUT_LABEL:-}" ]] || OUTPUT_INDEX="$PRVR_ANN_OUTPUT_LABEL"
EXTRA_ARGS+=(
  --ann_clip_raw_k "$CLIP_RAW_K"
  --ann_frame_raw_k "$FRAME_RAW_K"
  --ann_clip_nlist "$CLIP_NLIST"
  --ann_frame_nlist "$FRAME_NLIST"
  --ann_clip_nprobe "$CLIP_NPROBE"
  --ann_frame_nprobe "$FRAME_NPROBE"
  --ann_clip_ef_search "$CLIP_EF_SEARCH"
  --ann_frame_ef_search "$FRAME_EF_SEARCH"
  --ann_hnsw_m "$HNSW_M"
  --ann_hnsw_ef_construction "$HNSW_EF_CONSTRUCTION"
)

cd "$ROOT/all_prvr/GMMFormer_v2"

# IVF/HNSW use a query-independent bank built once.  ``origin`` intentionally
# re-encodes test contexts through the normal evaluator on every run.
# PRVR_PYTHON="$PY" "$ROOT/experiments/scripts/build_gmmformer_v2_context_bank.sh" act "$GPU" "$CKPT"

"$PY" src/main.py -d act_clip --gpu "$GPU" --eval --resume "$CKPT" \
  --ann_benchmark --ann_index "$INDEX" --ann_max_queries "$LIMIT" \
  "${EXTRA_ARGS[@]}" \
  --ann_output "$OUT/${OUTPUT_INDEX}_cross_branch_only.csv"
