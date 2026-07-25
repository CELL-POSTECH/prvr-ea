#!/usr/bin/env bash
# Export exact/branch top-10 video lists and exact-top10 branch ranks.
#
# This runs model evaluation once and writes compact per-query CSVs directly:
#   experiments/branch_rank/top10/<method>/<dataset>.csv
#
# Unlike make_branch_top10_csv.sh, this can compute exact_top10_clip_ranks and
# exact_top10_frame_ranks even when the exact videos are outside branch top-10,
# because it observes the full branch score matrices during evaluation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA DL-DKD MSC-PRVR MS-SL BGM-Net)

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/export_branch_top10.sh <dataset|all> <gpu> [model|all]

dataset: act | tvr | cha | msrvtt | all
gpu:     physical CUDA GPU index
model:   optional; one of GMMFormer, GMMFormer-v2, HLFormer, DreamPRVR,
         Holmes, BOA, DL-DKD, MSC-PRVR, MS-SL, BGM-Net, or all

Outputs:
  experiments/branch_rank/top10/<method>/<dataset>.csv

Each row is one query and contains:
  exact_top10_video_ids, clip_top10_video_ids, frame_top10_video_ids
  exact_top10_scores, clip_top10_scores, frame_top10_scores
  exact_top10_clip_ranks, exact_top10_frame_ranks
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }
GPU_ID="$2"
REQUESTED="${3:-all}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }

if [[ "$REQUESTED" != all ]]; then
  found=0
  for model in "${MODELS[@]}"; do [[ "$model" == "$REQUESTED" ]] && found=1; done
  [[ "$found" -eq 1 ]] || { echo "unsupported dual-branch model: $REQUESTED" >&2; exit 2; }
fi

export PYTHONPATH="$EXP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$DATASET" == all ]] && DATASETS=("${ALL_DATASETS[@]}") || DATASETS=("$DATASET")
[[ "$REQUESTED" == all ]] && SELECTED=("${MODELS[@]}") || SELECTED=("$REQUESTED")

checkpoint_for() {
  local model="$1" dataset="$2" collection spec repo_rel
  collection="$(collection_for "$dataset")"
  case "$model" in
    DL-DKD) latest_model_checkpoint "$PRVR_ROOT/DL-DKD" "$dataset" resnet ;;
    MS-SL) latest_model_checkpoint "$PRVR_ROOT/ms-sl" "$dataset" clip ;;
    BGM-Net) latest_model_checkpoint "$PRVR_ROOT/BGM-Net" "$dataset" clip ;;
    *)
      spec="$(config_spec "$model")"
      IFS='|' read -r repo_rel _ _ <<< "$spec"
      latest_checkpoint "$PRVR_ROOT/$repo_rel/results/clip/$collection"
      ;;
  esac
}

failures=0
for dataset in "${DATASETS[@]}"; do
  for model in "${SELECTED[@]}"; do
    checkpoint="$(checkpoint_for "$model" "$dataset")"
    if [[ -z "$checkpoint" ]]; then
      echo "skip $model/$dataset: no compatible checkpoint" >&2
      failures=$((failures + 1))
      continue
    fi
    safe_model="${model//-/_}"
    output_dir="$EXP_ROOT/branch_rank/top10/$safe_model"
    output="$output_dir/$dataset.csv"
    partial="$output.partial"
    mkdir -p "$output_dir"

    export PRVR_BRANCH_TOPK_OUTPUT="$partial"
    export PRVR_BRANCH_TOPK_MODEL="$model"
    export PRVR_BRANCH_TOPK_DATASET="$dataset"
    export PRVR_BRANCH_TOPK_FEATURE="clip"
    export PRVR_BRANCH_TOPK_CHECKPOINT="$checkpoint"
    export PRVR_BRANCH_TOPK_K=10
    export PRVR_BRANCH_TOPK_EXACT_K=10

    echo "[$(date -u +%FT%TZ)] export branch top-10 $model/$dataset -> $output"
    run_feature=clip
    [[ "$model" == DL-DKD ]] && run_feature=resnet
    if run_one eval "$model" "$dataset" "$run_feature"; then
      if [[ -s "$partial" ]]; then
        mv "$partial" "$output"
      else
        echo "top-10 collector did not create output: $partial" >&2
        failures=$((failures + 1))
      fi
    else
      echo "failed: $model/$dataset" >&2
      echo "partial output retained: $partial" >&2
      failures=$((failures + 1))
    fi

    unset PRVR_BRANCH_TOPK_OUTPUT PRVR_BRANCH_TOPK_MODEL PRVR_BRANCH_TOPK_DATASET
    unset PRVR_BRANCH_TOPK_FEATURE PRVR_BRANCH_TOPK_CHECKPOINT PRVR_BRANCH_TOPK_K
    unset PRVR_BRANCH_TOPK_EXACT_K
  done
done

[[ "$failures" -eq 0 ]]
