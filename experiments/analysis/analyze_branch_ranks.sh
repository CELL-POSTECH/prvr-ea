#!/usr/bin/env bash
# Evaluate existing CLIP checkpoints and record exact/branch ranks per query.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA DL-DKD MSC-PRVR MS-SL BGM-Net)

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/analyze_branch_ranks.sh <dataset|all> <gpu> [model|all]

Uses the newest existing CLIP checkpoint for each requested model.  It never
trains or overwrites checkpoints.

Examples:
  bash experiments/scripts/analyze_branch_ranks.sh act 0 GMMFormer-v2
  bash experiments/scripts/analyze_branch_ranks.sh all 1 all
EOF
}

[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || { usage; exit 0; }
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
RUN_DIR="$EXP_ROOT/branch_rank/per_query"
SUMMARY="$EXP_ROOT/branch_rank/branch_rank_summary.csv"
mkdir -p "$RUN_DIR"

[[ "$DATASET" == all ]] && DATASETS=("${ALL_DATASETS[@]}") || DATASETS=("$DATASET")
[[ "$REQUESTED" == all ]] && SELECTED=("${MODELS[@]}") || SELECTED=("$REQUESTED")

failures=0
for model in "${SELECTED[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    collection="$(collection_for "$dataset")"
    analysis_feature=clip
    case "$model" in
      # DL-DKD's original student is ResNet/I3D; CLIP is its teacher rather
      # than a replacement student input.
      DL-DKD) analysis_feature=resnet_teacher_clip; checkpoint="$(latest_model_checkpoint "$PRVR_ROOT/DL-DKD" "$dataset" resnet)" ;;
      MS-SL) checkpoint="$(latest_model_checkpoint "$PRVR_ROOT/ms-sl" "$dataset" clip)" ;;
      BGM-Net) checkpoint="$(latest_model_checkpoint "$PRVR_ROOT/BGM-Net" "$dataset" clip)" ;;
      *)
        spec="$(config_spec "$model")"
        IFS='|' read -r repo_rel _ _ <<< "$spec"
        checkpoint="$(latest_checkpoint "$PRVR_ROOT/$repo_rel/results/clip/$collection")"
        ;;
    esac
    if [[ -z "$checkpoint" ]]; then
      echo "skip $model $dataset: no compatible checkpoint" >&2
      failures=$((failures + 1))
      continue
    fi
    safe_model="${model//-/_}"
    output="$RUN_DIR/${safe_model}_${dataset}_${analysis_feature}.csv"
    export PRVR_BRANCH_RANK_OUTPUT="${output}.partial"
    export PRVR_BRANCH_RANK_MODEL="$model"
    export PRVR_BRANCH_RANK_DATASET="$dataset"
    export PRVR_BRANCH_RANK_FEATURE="$analysis_feature"
    export PRVR_BRANCH_RANK_CHECKPOINT="$checkpoint"
    echo "[$(date -u +%FT%TZ)] analyze $model $dataset (checkpoint: $checkpoint)"
    run_feature=clip
    [[ "$model" == DL-DKD ]] && run_feature=resnet
    if ! run_one eval "$model" "$dataset" "$run_feature"; then
      echo "failed: $model $dataset" >&2
      failures=$((failures + 1))
      echo "partial analysis retained at ${output}.partial" >&2
    else
      mv "${output}.partial" "$output"
    fi
    unset PRVR_BRANCH_RANK_OUTPUT PRVR_BRANCH_RANK_MODEL PRVR_BRANCH_RANK_DATASET PRVR_BRANCH_RANK_FEATURE PRVR_BRANCH_RANK_CHECKPOINT
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_branch_ranks.py" --input-dir "$RUN_DIR" --output "$SUMMARY"
echo "per-query CSV: $RUN_DIR"
echo "summary CSV:   $SUMMARY"
[[ "$failures" -eq 0 ]]
