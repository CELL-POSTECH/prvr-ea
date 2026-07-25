#!/usr/bin/env bash
# Synthetic CLIP-feature retrieval-latency benchmark.
#
# The reported E2E latency is: query encoder + retrieval over an already
# encoded gallery.  One-time context encoding is retained as gallery_prepare
# metadata in latency_detail.csv and is deliberately excluded from E2E.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PRVR_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PRVR_PYTHON:-/venv/prvr_c4c/bin/python}"
PRVR_ROOT="${PRVR_ROOT:-$ROOT_DIR/all_prvr}"
SYNTHETIC_ROOT="${LATENCY_SYNTHETIC_ROOT:-$ROOT_DIR/experiments/latency_synthetic}"
RESULT_ROOT="${LATENCY_RESULT_ROOT:-$ROOT_DIR/experiments/latency_results}"
DETAIL_CSV="$RESULT_ROOT/latency_detail.csv"
GPU_ID="${CUDA_GPU:-0}"
SEARCH_VECTOR_BUDGET="${SEARCH_VECTOR_BUDGET:-7237555}"
METHOD_FILTER="all"

declare -a CORPORA=()

usage() {
  cat <<'EOF'
Usage:
  bash experiments/scripts/latency.sh [options] <100000|500000|1000000|5000000> [...]

Options:
  --gpu N              CUDA GPU index (default: 0)
  --method NAME        one method, or all (default: all)
  --search-vector-budget N
                       raw representation count per GPU search chunk
                       (default: 7237555)
  --100k|--500k|--1m|--5m
                       convenient corpus-size aliases

Methods:
  CLIP4Clip, AMDNet, GMMFormer, GMMFormerV2, HLFormer, Holmes,
  DreamPRVR, BOA, DL-DKD, MSC-PRVR, MS-SL, BGMNet, all

Examples:
  bash experiments/scripts/latency.sh --gpu 0 100000
  bash experiments/scripts/latency.sh --gpu 1 --method GMMFormerV2 100000 500000
  bash experiments/scripts/latency.sh --gpu 0 --100k --500k

The script auto-creates each deterministic [V,1,512] synthetic gallery when
missing. Results are appended to experiments/latency_results/latency_detail.csv
and the latest successful QPS per corpus/method is exported to qps.csv.
EOF
}

canonical_method() {
  case "$1" in
    clip4clip|CLIP4Clip) echo CLIP4Clip ;;
    amdnet|AMDNet) echo AMDNet ;;
    gmmformer|GMMFormer) echo GMMFormer ;;
    gmmformerv2|GMMFormerV2|GMMFormer-v2) echo GMMFormerV2 ;;
    hlformer|HLFormer) echo HLFormer ;;
    holmes|Holmes) echo Holmes ;;
    dreamprvr|DreamPRVR) echo DreamPRVR ;;
    boa|BOA) echo BOA ;;
    dldkd|DL-DKD) echo DL-DKD ;;
    msc-prvr|MSC-PRVR|mscprvr) echo MSC-PRVR ;;
    ms-sl|MS-SL|mssl) echo MS-SL ;;
    bgmnet|BGMNet|BGM-Net) echo BGMNet ;;
    all) echo all ;;
    *) return 1 ;;
  esac
}

valid_corpus() {
  case "$1" in 100000|500000|1000000|5000000) return 0 ;; *) return 1 ;; esac
}

record_failure() {
  local corpus="$1" method="$2" error="$3"
  ROOT_DIR="$ROOT_DIR" "$PYTHON_BIN" - "$DETAIL_CSV" "$corpus" "$method" "$error" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["ROOT_DIR"]) / "experiments"))
from latency.results import append_detail, export_wide_qps
output = Path(sys.argv[1])
append_detail(output, {"corpus": sys.argv[2], "method": sys.argv[3],
                       "status": "failed", "error": sys.argv[4]})
export_wide_qps(output, output.parent / "qps.csv")
PY
}

latest_best() {
  local directory="$1"
  find "$directory" -type f -name best.ckpt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}

latest_model() {
  local directory="$1"
  find "$directory" -type f -name model.ckpt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}

run_checked() {
  local corpus="$1" method="$2"; shift 2
  echo "[$(date -u +%FT%TZ)] V=$corpus GPU=$GPU_ID: $method"
  if "$@"; then
    return 0
  fi
  local rc=$?
  echo "  FAILED ($method, rc=$rc)" >&2
  record_failure "$corpus" "$method" "worker_exit_$rc"
  return "$rc"
}

run_method() {
  local method="$1" corpus="$2" directory="$3" checkpoint repo config_method
  case "$method" in
    CLIP4Clip)
      repo="$PRVR_ROOT/CLIP4Clip"
      run_checked "$corpus" "CLIP4Clip" "$PYTHON_BIN" "$ROOT_DIR/experiments/latency/clip4clip_latency.py" \
        --repo "$repo" --variant flat --corpus-dir "$directory" --output "$DETAIL_CSV" \
        --gpu "$GPU_ID" --search-vector-budget "$SEARCH_VECTOR_BUDGET"
      ;;
    AMDNet|GMMFormer|GMMFormerV2|HLFormer|Holmes|DreamPRVR|BOA|MSC-PRVR)
      case "$method" in
        AMDNet) repo="$PRVR_ROOT/AMDNet"; config_method=AMDNet ;;
        GMMFormer) repo="$PRVR_ROOT/GMMFormer"; config_method=GMMFormer ;;
        GMMFormerV2) repo="$PRVR_ROOT/GMMFormer_v2"; config_method=GMMFormer-v2 ;;
        HLFormer) repo="$PRVR_ROOT/ICCV25-HLFormer"; config_method=HLFormer ;;
        Holmes) repo="$PRVR_ROOT/ICML26-Holmes"; config_method=Holmes ;;
        DreamPRVR) repo="$PRVR_ROOT/CVPR26-DreamPRVR"; config_method=DreamPRVR ;;
        BOA) repo="$PRVR_ROOT/BOA"; config_method=BOA ;;
        MSC-PRVR) repo="$PRVR_ROOT/MSC_PRVR"; config_method=MSC-PRVR ;;
      esac
      checkpoint="$(latest_best "$repo/results/clip/activitynet")"
      if [[ -z "$checkpoint" ]]; then
        echo "missing CLIP checkpoint: $method" >&2
        record_failure "$corpus" "$method" "missing_checkpoint"
        return 2
      fi
      run_checked "$corpus" "$method" "$PYTHON_BIN" "$ROOT_DIR/experiments/latency/config_feature_latency.py" \
        --repo "$repo" --method "$config_method" --checkpoint "$checkpoint" --corpus-dir "$directory" \
        --output "$DETAIL_CSV" --gpu "$GPU_ID" --search-vector-budget "$SEARCH_VECTOR_BUDGET"
      ;;
    DL-DKD)
      repo="$PRVR_ROOT/DL-DKD"
      checkpoint="$(latest_model "$repo/results/resnet/activitynet")"
      if [[ -z "$checkpoint" ]]; then
        echo "missing DL-DKD original dual-feature checkpoint" >&2
        record_failure "$corpus" "$method" "missing_checkpoint"
        return 2
      fi
      run_checked "$corpus" "$method" "$PYTHON_BIN" "$ROOT_DIR/experiments/latency/dldkd_latency.py" \
        --repo "$repo" --checkpoint "$checkpoint" --corpus-dir "$directory" --output "$DETAIL_CSV" --gpu "$GPU_ID" \
        --search-vector-budget "$SEARCH_VECTOR_BUDGET"
      ;;
    MS-SL|BGMNet)
      if [[ "$method" == MS-SL ]]; then repo="$PRVR_ROOT/ms-sl"; config_method="MS-SL"; else repo="$PRVR_ROOT/BGM-Net"; config_method="BGM-Net"; fi
      checkpoint="$(latest_model "$repo/results/clip/activitynet")"
      if [[ -z "$checkpoint" ]]; then
        echo "missing CLIP checkpoint: $method" >&2
        record_failure "$corpus" "$method" "missing_checkpoint"
        return 2
      fi
      run_checked "$corpus" "$method" "$PYTHON_BIN" "$ROOT_DIR/experiments/latency/auto_feature_latency.py" \
        --repo "$repo" --method "$config_method" --checkpoint "$checkpoint" --corpus-dir "$directory" \
        --output "$DETAIL_CSV" --gpu "$GPU_ID" --search-vector-budget "$SEARCH_VECTOR_BUDGET"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpu) GPU_ID="${2:?--gpu requires a value}"; shift 2 ;;
    --method) METHOD_FILTER="$(canonical_method "${2:?--method requires a value}")" || { echo "unknown method: $2" >&2; exit 2; }; shift 2 ;;
    --search-vector-budget) SEARCH_VECTOR_BUDGET="${2:?--search-vector-budget requires a value}"; shift 2 ;;
    --100|--100k) CORPORA+=(100000); shift ;;
    --500|--500k) CORPORA+=(500000); shift ;;
    --1m|--1000k) CORPORA+=(1000000); shift ;;
    --5m|--5000k) CORPORA+=(5000000); shift ;;
    *) valid_corpus "$1" || { echo "unsupported corpus: $1" >&2; usage >&2; exit 2; }; CORPORA+=("$1"); shift ;;
  esac
done

[[ ${#CORPORA[@]} -gt 0 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "PRVR Python not found: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$RESULT_ROOT"

ALL_METHODS=(CLIP4Clip AMDNet GMMFormer GMMFormerV2 HLFormer Holmes DreamPRVR BOA DL-DKD MSC-PRVR MS-SL BGMNet)
STATUS=0
for corpus in "${CORPORA[@]}"; do
  directory="$($PYTHON_BIN "$ROOT_DIR/experiments/latency/synthetic_corpus.py" --root "$SYNTHETIC_ROOT" --videos "$corpus" | tail -n1)" || exit $?
  echo "Synthetic corpus ready: $directory"
  for method in "${ALL_METHODS[@]}"; do
    [[ "$METHOD_FILTER" == all || "$METHOD_FILTER" == "$method" ]] || continue
    run_method "$method" "$corpus" "$directory" || STATUS=1
  done
done

ROOT_DIR="$ROOT_DIR" "$PYTHON_BIN" - "$DETAIL_CSV" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["ROOT_DIR"]) / "experiments"))
from latency.results import export_wide_qps
detail = Path(sys.argv[1]); export_wide_qps(detail, detail.parent / "qps.csv")
PY
echo "Detailed results: $DETAIL_CSV"
echo "QPS summary:      $RESULT_ROOT/qps.csv"
exit "$STATUS"
