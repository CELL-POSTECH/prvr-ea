#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/jaewoo/anaconda3/envs/prvr/bin/python" ]]; then
    PYTHON_BIN="/home/jaewoo/anaconda3/envs/prvr/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
GPU="${GPU:-0}"
DATASET_ARG="${1:-all}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_DATA_ROOT="${PRVR_DATA_ROOT:-${ROOT}/datasets}"

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "$DATASET_ARG" in
  act)
    DATASET_ARG="activitynet"
    ;;
  all|activitynet|tvr|charades|msrvtt)
    ;;
  *)
    echo "Usage: bash scripts/run_candidate_mining.sh [all|activitynet|act|tvr|charades|msrvtt] [--dry-run] [--gpu N]" >&2
    exit 1
    ;;
esac

declare -A DATA_ROOTS=(
  [activitynet]="${PRVR_ACTIVITYNET_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
  [tvr]="${PRVR_TVR_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
  [charades]="${PRVR_CHARADES_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
  [msrvtt]="${PRVR_MSRVTT_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
)

declare -A DREAMPRVR_CKPTS=(
  [activitynet]="${ROOT}/all_prvr/CVPR26-DreamPRVR/results/clip/activitynet/DreamPRVR/best.ckpt"
  [tvr]="${ROOT}/all_prvr/CVPR26-DreamPRVR/results/clip/tvr/DreamPRVR/best.ckpt"
  [charades]="${ROOT}/all_prvr/CVPR26-DreamPRVR/results/clip/charades/DreamPRVR/best.ckpt"
  [msrvtt]="${ROOT}/all_prvr/CVPR26-DreamPRVR/results/clip/msrvtt/DreamPRVR/best.ckpt"
)

declare -A GMMFORMER_CKPTS=(
  [activitynet]="${ROOT}/all_prvr/GMMFormer_v2/results/clip/activitynet/gmmformer_v2/best.ckpt"
  [tvr]="${ROOT}/all_prvr/GMMFormer_v2/results/clip/tvr/gmmformer_v2/best.ckpt"
  [charades]="${ROOT}/all_prvr/GMMFormer_v2/results/clip/charades/gmmformer_v2/best.ckpt"
  [msrvtt]="${ROOT}/all_prvr/GMMFormer_v2/results/clip/msrvtt/gmmformer_v2/best.ckpt"
)

declare -A HLFORMER_CKPTS=(
  [activitynet]="${ROOT}/all_prvr/ICCV25-HLFormer/results/clip/activitynet/HLFormer/best.ckpt"
  [tvr]="${ROOT}/all_prvr/ICCV25-HLFormer/results/clip/tvr/HLFormer/best.ckpt"
  [charades]="${ROOT}/all_prvr/ICCV25-HLFormer/results/clip/charades/HLFormer/best.ckpt"
  [msrvtt]="${ROOT}/all_prvr/ICCV25-HLFormer/results/clip/msrvtt/HLFormer/best.ckpt"
)

declare -A HOLMES_CKPTS=(
  [activitynet]="${ROOT}/all_prvr/ICML26-Holmes/results/clip/activitynet/Holmes/20260721-142334/best.ckpt"
  [tvr]="${ROOT}/all_prvr/ICML26-Holmes/results/clip/tvr/Holmes/20260721-155445/best.ckpt"
  [charades]="${ROOT}/all_prvr/ICML26-Holmes/results/clip/charades/Holmes/20260721-151633/best.ckpt"
  [msrvtt]="${ROOT}/all_prvr/ICML26-Holmes/results/clip/msrvtt/Holmes/20260721-160641/best.ckpt"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

run_candidates() {
  local dataset="$1"
  local data_root="${DATA_ROOTS[$dataset]}"

  require_file "${DREAMPRVR_CKPTS[$dataset]}"
  require_file "${GMMFORMER_CKPTS[$dataset]}"
  require_file "${HLFORMER_CKPTS[$dataset]}"
  require_file "${HOLMES_CKPTS[$dataset]}"

  local cmd=(
    "$PYTHON_BIN" "${ROOT}/scripts/build_pseudo_gt_from_ckpts.py"
    --dataset "$dataset"
    --prvr-model "DreamPRVR=dreamprvr=${DREAMPRVR_CKPTS[$dataset]}"
    --prvr-model "GMMFormerv2=gmmformer=${GMMFORMER_CKPTS[$dataset]}"
    --prvr-model "HLFormer=hlformer=${HLFORMER_CKPTS[$dataset]}"
    --prvr-model "Holmes=holmes=${HOLMES_CKPTS[$dataset]}"
    --output "${ROOT}/outputs/upstream/pseudo_gt_candidates.${dataset}.jsonl"
    --gpu "$GPU"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  echo "===== ${dataset} ====="
  PRVR_DATA_ROOT="$data_root" "${cmd[@]}"
}

if [[ "$DATASET_ARG" == "all" ]]; then
  for dataset in activitynet tvr charades msrvtt; do
    run_candidates "$dataset"
  done
else
  run_candidates "$DATASET_ARG"
fi
