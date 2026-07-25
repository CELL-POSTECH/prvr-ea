#!/usr/bin/env bash
# Export exact/branch top-100 video lists for every query.
#
# For each query this writes:
#   - exact/base top-100 videos from the model's original fused score
#   - clip-branch top-100 videos
#   - frame-branch top-100 videos
#   - how deep clip/frame ranks must go to cover exact top-10/top-100
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA DL-DKD MSC-PRVR MS-SL BGM-Net)

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/export_branch_top100.sh <dataset|all> <gpu> [model|all] [top_k]

dataset: act | tvr | cha | msrvtt | all
gpu:     physical CUDA GPU index
model:   optional; one of GMMFormer, GMMFormer-v2, HLFormer, DreamPRVR,
         Holmes, BOA, DL-DKD, MSC-PRVR, MS-SL, BGM-Net, or all
top_k:   optional; default 100

Outputs:
  experiments/branch_rank/top<top_k>/<method>/<dataset>.csv

Each row is one query and contains exact/clip/frame top-k video-id lists plus
the full-rank depth needed for clip/frame to cover exact top-10.
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }
GPU_ID="$2"
REQUESTED="${3:-all}"
TOP_K="${4:-100}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }
[[ "$TOP_K" =~ ^[0-9]+$ && "$TOP_K" -ge 10 ]] || { echo "top_k must be an integer >= 10" >&2; exit 2; }

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
    # Keep exports at different depths separate.  A top-3000 export is a
    # score cache and must never overwrite the existing top-100 analysis.
    output_dir="$EXP_ROOT/branch_rank/top$TOP_K/$safe_model"
    output="$output_dir/$dataset.csv"
    partial="$output.partial"
    mkdir -p "$output_dir"

    export PRVR_BRANCH_TOPK_OUTPUT="$partial"
    export PRVR_BRANCH_TOPK_MODEL="$model"
    export PRVR_BRANCH_TOPK_DATASET="$dataset"
    export PRVR_BRANCH_TOPK_FEATURE="clip"
    export PRVR_BRANCH_TOPK_CHECKPOINT="$checkpoint"
    export PRVR_BRANCH_TOPK_K="$TOP_K"
    export PRVR_BRANCH_TOPK_EXACT_K=10

    echo "[$(date -u +%FT%TZ)] export branch top-$TOP_K $model/$dataset -> $output"
    run_feature=clip
    [[ "$model" == DL-DKD ]] && run_feature=resnet
    if run_one eval "$model" "$dataset" "$run_feature"; then
      if [[ -s "$partial" ]]; then
        mv "$partial" "$output"
      else
        echo "top-k collector did not create output: $partial" >&2
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
