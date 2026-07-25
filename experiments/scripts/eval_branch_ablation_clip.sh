#!/usr/bin/env bash
# Dual-branch ablation for CLIP-trained PRVR checkpoints on ActivityNet/TVR.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA DL-DKD MSC-PRVR MS-SL BGM-Net)

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/eval_branch_ablation_clip.sh <act|tvr|all> <gpu> [model|all] [--dry-run]

Runs existing CLIP-feature checkpoints for dual-branch ablation.  DL-DKD uses
its original dual-feature checkpoint, as in the recall CLIP table.

Output:
  experiments/branch_ablation/clip/act.csv
  experiments/branch_ablation/clip/act_long.csv
  experiments/branch_ablation/clip/tvr.csv
  experiments/branch_ablation/clip/tvr_long.csv

Summary cells are formatted as R@1/R@5/R@10.
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }
GPU_ID="$2"
REQUESTED="${3:-all}"
DRY_RUN=0
[[ "${4:-}" != --dry-run ]] || DRY_RUN=1
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }
[[ "$DATASET" == all || "$DATASET" == act || "$DATASET" == tvr ]] || {
  echo "branch ablation currently supports only act, tvr, or all" >&2
  exit 2
}

case "$REQUESTED" in
  GMMFormerV2) REQUESTED=GMMFormer-v2 ;;
  BGMNet|BGMNet*) REQUESTED=BGM-Net ;;
esac

if [[ "$REQUESTED" != all ]]; then
  found=0
  for model in "${MODELS[@]}"; do [[ "$model" == "$REQUESTED" ]] && found=1; done
  [[ "$found" -eq 1 ]] || { echo "unsupported dual-branch model: $REQUESTED" >&2; exit 2; }
fi

export PYTHONPATH="$EXP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$DATASET" == all ]] && DATASETS=(act tvr) || DATASETS=("$DATASET")
[[ "$REQUESTED" == all ]] && SELECTED=("${MODELS[@]}") || SELECTED=("$REQUESTED")

checkpoint_for() {
  local model="$1" dataset="$2" collection repo_rel spec
  collection="$(collection_for "$dataset")"
  case "$model" in
    DL-DKD) latest_model_checkpoint "$PRVR_ROOT/DL-DKD" "$dataset" resnet ;;
    MS-SL) latest_model_checkpoint "$PRVR_ROOT/ms-sl" "$dataset" clip ;;
    BGM-Net) latest_model_checkpoint "$PRVR_ROOT/BGM-Net" "$dataset" clip ;;
    *)
      spec="$(config_spec "$model")"; IFS='|' read -r repo_rel _ _ <<< "$spec"
      latest_checkpoint "$PRVR_ROOT/$repo_rel/results/clip/$collection"
      ;;
  esac
}

branch_map_for() {
  local model="$1" dataset="$2"
  case "$model" in
    GMMFormer) echo '{"32":"clip","1":"frame"}' ;;
    GMMFormer-v2|HLFormer|DreamPRVR|Holmes|BOA) echo '{"32":"clip","128":"frame"}' ;;
    MS-SL) echo '{"528":"clip","1":"frame"}' ;;
    BGM-Net)
      [[ "$dataset" == tvr ]] && echo '{"528":"clip","1":"frame"}' || echo '{"1176":"clip","1":"frame"}'
      ;;
    *) echo '{}' ;;
  esac
}

sequence_for() {
  case "$1" in
    MSC-PRVR) echo '["clip","frame"]' ;;
    *) echo '' ;;
  esac
}

failures=0
for dataset in "${DATASETS[@]}"; do
  for model in "${SELECTED[@]}"; do
    checkpoint="$(checkpoint_for "$model" "$dataset")"
    if [[ -z "$checkpoint" ]]; then
      echo "missing checkpoint: $model/$dataset" >&2
      failures=$((failures + 1))
      continue
    fi
    safe_model="${model//-/_}"
    out_dir="$EXP_ROOT/branch_ablation/clip/runs/$dataset"
    summary="$out_dir/${safe_model}.csv"
    long="$out_dir/${safe_model}_long.csv"
    partial_summary="$summary.partial"
    partial_long="$long.partial"
    branches="$(branch_map_for "$model" "$dataset")"
    sequence="$(sequence_for "$model")"
    echo "[$(date -u +%FT%TZ)] branch-ablation $model/$dataset -> $summary"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "  checkpoint: $checkpoint"
      echo "  branches:   $branches"
      [[ -z "$sequence" ]] || echo "  sequence:   $sequence"
      continue
    fi
    mkdir -p "$out_dir"
    export PRVR_BRANCH_ABLATION_OUTPUT="$partial_summary"
    export PRVR_BRANCH_ABLATION_LONG_OUTPUT="$partial_long"
    export PRVR_BRANCH_ABLATION_METHOD="$model"
    export PRVR_BRANCH_ABLATION_DATASET="$dataset"
    export PRVR_BRANCH_ABLATION_CHECKPOINT="$checkpoint"
    export PRVR_BRANCH_ABLATION_BRANCHES="$branches"
    if [[ -n "$sequence" ]]; then
      export PRVR_BRANCH_ABLATION_SEQUENCE="$sequence"
    else
      unset PRVR_BRANCH_ABLATION_SEQUENCE
    fi
    [[ "$model" == BGM-Net && "$dataset" == act ]] && export PRVR_RAW_DEDUP_EVAL_QUERY_BSZ=7 || unset PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
    run_feature=clip
    [[ "$model" == DL-DKD ]] && run_feature=resnet
    if run_one eval "$model" "$dataset" "$run_feature"; then
      if [[ ! -s "$partial_summary" || ! -s "$partial_long" ]]; then
        echo "ablation collector did not create output for $model/$dataset" >&2
        failures=$((failures + 1))
      else
        mv "$partial_summary" "$summary"
        mv "$partial_long" "$long"
      fi
    else
      echo "failed branch ablation: $model/$dataset" >&2
      failures=$((failures + 1))
    fi
    unset PRVR_BRANCH_ABLATION_OUTPUT PRVR_BRANCH_ABLATION_LONG_OUTPUT PRVR_BRANCH_ABLATION_METHOD PRVR_BRANCH_ABLATION_DATASET PRVR_BRANCH_ABLATION_CHECKPOINT PRVR_BRANCH_ABLATION_BRANCHES PRVR_BRANCH_ABLATION_SEQUENCE PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
  done
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_branch_ablation.py" \
      --root "$EXP_ROOT/branch_ablation/clip" --dataset "$dataset"
  fi
done

[[ "$failures" -eq 0 ]]
