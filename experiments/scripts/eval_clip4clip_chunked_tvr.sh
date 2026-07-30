#!/usr/bin/env bash
# TVR CLIP4Clip ViT-B/32 raw-frame zero-shot evaluation. The original ordered
# raw-frame sequence is split into fixed-size chunks before any max-frame
# sampling; the parent-video score is max over chunk scores.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../runner_lib.sh"

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/eval_clip4clip_chunked_tvr.sh <gpu> <10|20|30|all>

Examples:
  bash experiments/scripts/eval_clip4clip_chunked_tvr.sh 1 10
  bash experiments/scripts/eval_clip4clip_chunked_tvr.sh 1 all
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -eq 2 ]] || { usage >&2; exit 2; }
GPU_ID="$1"
REQUESTED_CHUNK="$2"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }

case "$REQUESTED_CHUNK" in
  10|20|30) CHUNKS=("$REQUESTED_CHUNK") ;;
  all) CHUNKS=(10 20 30) ;;
  *) usage >&2; exit 2 ;;
esac

REPO="$PRVR_ROOT/CLIP4Clip"
RUNNER="$REPO/scripts/run_prvr_rawframes.sh"
CSV="$EXP_ROOT/recall_results/recall_clip4clip_tvr_chunked.csv"
EXPORTER="$SCRIPT_DIR/export_clip4clip_chunked_tvr.py"
FAILURES=0

for CHUNK_SIZE in "${CHUNKS[@]}"; do
  FEATURE="rawframes128_chunk${CHUNK_SIZE}"
  LOG="$(new_log_path CLIP4Clip tvr "$FEATURE")"
  RUN_DIR="$REPO/results/rawframes/tvr/zeroshot_f128_chunk${CHUNK_SIZE}"
  COMMAND="cd $REPO && MAX_FRAMES=128 CHUNK_SIZE=$CHUNK_SIZE MAX_EVAL_SAMPLES=0 BATCH_SIZE_VAL=32 NUM_WORKERS=2 PYTHON_BIN=$PYTHON_BIN bash scripts/run_prvr_rawframes.sh zeroshot tvr $GPU_ID"
  echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: CLIP4Clip TVR zero-shot max_frames=128 chunk_size=$CHUNK_SIZE"
  if (
      cd "$REPO"
      MAX_FRAMES=128 CHUNK_SIZE="$CHUNK_SIZE" MAX_EVAL_SAMPLES=0 BATCH_SIZE_VAL=32 NUM_WORKERS=2 \
        PYTHON_BIN="$PYTHON_BIN" bash "$RUNNER" zeroshot tvr "$GPU_ID"
  ) >"$LOG" 2>&1; then
    "$PYTHON_BIN" "$EXPORTER" --csv "$CSV" --log "$LOG" --run-dir "$RUN_DIR" \
      --chunk-size "$CHUNK_SIZE" --max-frames 128
    record CLIP4Clip tvr "$FEATURE" ok "$LOG" "$RUN_DIR" "$COMMAND"
    echo "  complete; log: $LOG"
  else
    RC=$?
    record CLIP4Clip tvr "$FEATURE" "eval_failed:$RC" "$LOG" "$RUN_DIR" "$COMMAND"
    echo "  FAILED (rc=$RC); log: $LOG" >&2
    FAILURES=1
  fi
done

exit "$FAILURES"
