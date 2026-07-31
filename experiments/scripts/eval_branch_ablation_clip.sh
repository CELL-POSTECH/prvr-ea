#!/usr/bin/env bash
# Dual-branch ablation for CLIP-trained PRVR checkpoints on ActivityNet/TVR.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# DL-DKD reports only Base and its two original branches. Its
# inheritance/exploration query encoders are independent, so MP/WMP is not
# defined and is intentionally omitted for that model.
MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA MSC-PRVR DL-DKD MS-SL BGM-Net)

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/eval_branch_ablation_clip.sh <act|tvr|all> <gpu> [model|all] [--dry-run]

Runs existing CLIP-feature checkpoints for dual-branch ablation. Mean-pooling
conditions pool branch representation vectors before cosine scoring. DL-DKD
reports Base, inheritance-only, and exploration-only only; its MP/WMP cells
are intentionally empty because its branch query encoders are independent.

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
    MS-SL) latest_model_checkpoint "$PRVR_ROOT/ms-sl" "$dataset" clip ;;
    BGM-Net) latest_model_checkpoint "$PRVR_ROOT/BGM-Net" "$dataset" clip ;;
    DL-DKD) latest_model_checkpoint "$PRVR_ROOT/DL-DKD" "$dataset" resnet ;;
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

weights_for() {
  local model="$1" dataset="$2"
  case "$model/$dataset" in
    DreamPRVR/act) echo '0.6 0.4' ;;
    DreamPRVR/tvr) echo '0.5 0.5' ;;
    Holmes/*) echo '0.6 0.4' ;;
    BOA/*) echo '0.5 0.5' ;;
    MSC-PRVR/*) echo '0.4 0.6' ;;
    *) echo '0.7 0.3' ;;
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
    read -r left_weight right_weight <<< "$(weights_for "$model" "$dataset")"
    echo "[$(date -u +%FT%TZ)] branch-ablation $model/$dataset -> $summary"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "  checkpoint: $checkpoint"
      echo "  branches:   $branches"
      echo "  weights:    $left_weight / $right_weight"
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
    export PRVR_BRANCH_ABLATION_LEFT_WEIGHT="$left_weight"
    export PRVR_BRANCH_ABLATION_RIGHT_WEIGHT="$right_weight"
    if [[ "$model" == DL-DKD ]]; then
      export PRVR_BRANCH_ABLATION_POOLING=0
    else
      unset PRVR_BRANCH_ABLATION_POOLING
    fi
    if [[ -n "$sequence" ]]; then
      export PRVR_BRANCH_ABLATION_SEQUENCE="$sequence"
    else
      unset PRVR_BRANCH_ABLATION_SEQUENCE
    fi
    [[ "$model" == BGM-Net && "$dataset" == act ]] && export PRVR_RAW_DEDUP_EVAL_QUERY_BSZ=7 || unset PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
    if run_one eval "$model" "$dataset" clip; then
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
    unset PRVR_BRANCH_ABLATION_OUTPUT PRVR_BRANCH_ABLATION_LONG_OUTPUT PRVR_BRANCH_ABLATION_METHOD PRVR_BRANCH_ABLATION_DATASET PRVR_BRANCH_ABLATION_CHECKPOINT PRVR_BRANCH_ABLATION_BRANCHES PRVR_BRANCH_ABLATION_LEFT_WEIGHT PRVR_BRANCH_ABLATION_RIGHT_WEIGHT PRVR_BRANCH_ABLATION_SEQUENCE PRVR_BRANCH_ABLATION_POOLING PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
  done
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_branch_ablation.py" \
      --root "$EXP_ROOT/branch_ablation/clip" --dataset "$dataset"
  fi
done

[[ "$failures" -eq 0 ]]
