#!/usr/bin/env bash
# Summary-only raw-representation dedup analysis. No raw rank CSV is written.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA MSC-PRVR DL-DKD MS-SL BGM-Net)

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/analyze_raw_dedup.sh <dataset|all> <gpu> [model|all] [top_l] [unique_ks] [--dry-run]

Runs existing CLIP checkpoints (DL-DKD uses its original dual-feature checkpoint)
and writes analysis-only artifacts to:
  experiments/branch_rank/full_repr/<method>/<dataset>/summary.csv
  experiments/branch_rank/full_repr/<method>/<dataset>/per_query.csv.gz

If top_l is provided, outputs are written to:
  experiments/branch_rank/full_repr_top<top_l>/<method>/<dataset>/

Examples:
  bash experiments/scripts/analyze_raw_dedup.sh act 0 GMMFormer-v2
  bash experiments/scripts/analyze_raw_dedup.sh act 0 GMMFormer-v2 3000 1,5,10,20
  bash experiments/scripts/analyze_raw_dedup.sh all 0 all 3000 1,5,10,20

``summary.csv`` contains detailed depth, duplicate-rate, video-rank, raw-GT
rank, and branch Recall@1/5/10 statistics. ``per_query.csv.gz`` contains the
underlying scalar values for every query; it never contains full raw rankings.

AMDNet and CLIP4Clip are excluded because they are single-branch baselines.
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 6 ]] || { usage >&2; exit 2; }
DATASET="$(normalize_dataset "$1")" || { echo "unknown dataset: $1" >&2; exit 2; }
GPU_ID="$2"
REQUESTED="${3:-all}"
TOP_L=""
UNIQUE_KS=""
DRY_RUN=0
for arg in "${@:4}"; do
  if [[ "$arg" == --dry-run ]]; then
    DRY_RUN=1
  elif [[ -z "$TOP_L" ]]; then
    TOP_L="$arg"
  elif [[ -z "$UNIQUE_KS" ]]; then
    UNIQUE_KS="$arg"
  else
    usage >&2
    exit 2
  fi
done
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; exit 2; }
if [[ -n "$TOP_L" ]]; then
  [[ "$TOP_L" =~ ^[0-9]+$ && "$TOP_L" -ge 20 ]] || { echo "top_l must be an integer >= 20" >&2; exit 2; }
  UNIQUE_KS="${UNIQUE_KS:-1,5,10,20}"
fi

if [[ "$REQUESTED" != all ]]; then
  found=0
  for model in "${MODELS[@]}"; do [[ "$model" == "$REQUESTED" ]] && found=1; done
  [[ "$found" -eq 1 ]] || { echo "unsupported multi-branch model: $REQUESTED" >&2; exit 2; }
fi

export PYTHONPATH="$EXP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$DATASET" == all ]] && DATASETS=("${ALL_DATASETS[@]}") || DATASETS=("$DATASET")
[[ "$REQUESTED" == all ]] && SELECTED=("${MODELS[@]}") || SELECTED=("$REQUESTED")
OUTPUT_ROOT="$EXP_ROOT/branch_rank/full_repr"
[[ -z "$TOP_L" ]] || OUTPUT_ROOT="$EXP_ROOT/branch_rank/full_repr_top$TOP_L"

checkpoint_for() {
  local model="$1" dataset="$2" collection repo_rel
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

expected_branches_for() {
  case "$1" in
    DL-DKD) echo 'inheritance exploration' ;;
    *)      echo 'clip frame' ;;
  esac
}

validate_summary() {
  local summary="$1" model="$2" expected branch
  [[ -s "$summary" ]] || {
    echo "raw-dedup collector did not create a summary: $summary" >&2
    return 1
  }
  for branch in $(expected_branches_for "$model"); do
    # A branch must have at least one statistics row; this makes an adapter
    # shape mismatch a hard failure instead of silently writing a partial CSV.
    if ! awk -F, -v branch="$branch" 'NR > 1 && $3 == branch { found=1; exit } END { exit !found }' "$summary"; then
      echo "raw-dedup summary missing branch '$branch': $summary" >&2
      return 1
    fi
  done
}

for dataset in "${DATASETS[@]}"; do
  for model in "${SELECTED[@]}"; do
    checkpoint="$(checkpoint_for "$model" "$dataset")"
    [[ -n "$checkpoint" ]] || { echo "missing checkpoint: $model/$dataset" >&2; continue; }
    safe_model="${model//-/_}"
    output_dir="$OUTPUT_ROOT/$safe_model/$dataset"
    summary="$output_dir/summary.csv"
    case "$model" in
      GMMFormer) branches='{"32":"clip","1":"frame"}' ;;
      GMMFormer-v2|HLFormer|DreamPRVR|Holmes|BOA) branches='{"32":"clip","128":"frame"}' ;;
      MSC-PRVR) branches='{}' ;;
      DL-DKD) branches='{}' ;;
      MS-SL) branches='{"528":"clip","1":"frame"}' ;;
      BGM-Net)
        # Its documented TVR configuration uses map_size=32 (528 temporal
        # proposals); the other original configurations use map_size=48
        # (1176 proposals).
        [[ "$dataset" == tvr ]] && branches='{"528":"clip","1":"frame"}' || branches='{"1176":"clip","1":"frame"}'
        ;;
    esac
    echo "[$(date -u +%FT%TZ)] raw-dedup $model/$dataset -> $summary"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "  checkpoint: $checkpoint; branch map: $branches"
      [[ -z "$TOP_L" ]] || echo "  raw top_l: $TOP_L; unique ks: $UNIQUE_KS; output root: $OUTPUT_ROOT"
      continue
    fi
    mkdir -p "$output_dir"
    partial_summary="$summary.partial"
    per_query="$output_dir/per_query.csv.gz"
    partial_per_query="$per_query.partial"
    collection="$(collection_for "$dataset")"
    export PRVR_RAW_DEDUP_SUMMARY="$partial_summary"
    export PRVR_RAW_DEDUP_PER_QUERY="$partial_per_query"
    export PRVR_RAW_DEDUP_METHOD="$model"
    export PRVR_RAW_DEDUP_DATASET="$dataset"
    export PRVR_RAW_DEDUP_BRANCHES="$branches"
    export PRVR_RAW_DEDUP_QUERY_FILE="$DATA_ROOT/$collection/TextData/${collection}test.caption.txt"
    if [[ -n "$TOP_L" ]]; then
      export PRVR_RAW_DEDUP_TOP_L="$TOP_L"
      export PRVR_RAW_DEDUP_UNIQUE_KS="$UNIQUE_KS"
    else
      unset PRVR_RAW_DEDUP_TOP_L PRVR_RAW_DEDUP_UNIQUE_KS
    fi
    unset PRVR_RAW_DEDUP_SEQUENCE
    [[ "$model" != MSC-PRVR ]] || export PRVR_RAW_DEDUP_SEQUENCE='["clip","frame"]'
    # BGM-Net materializes [query, proposal, video] before reducing proposal
    # scores.  Query chunking is mathematically exact and keeps its 1176-proposal
    # ActivityNet analysis within a 24 GiB GPU.
    # ActivityNet has 15,753 test queries. Seven avoids both the original
    # 50-query OOM and a singleton final batch (the upstream BGM code squeezes
    # that batch to 2-D before its unchanged score computation).
    [[ "$model" == BGM-Net ]] && export PRVR_RAW_DEDUP_EVAL_QUERY_BSZ=7 || unset PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
    run_feature=clip
    [[ "$model" != DL-DKD ]] || run_feature=resnet
    if ! run_one eval "$model" "$dataset" "$run_feature"; then
      echo "failed: $model/$dataset" >&2
      echo "partial summary retained: $partial_summary" >&2
      continue
    fi
    unset PRVR_RAW_DEDUP_SUMMARY PRVR_RAW_DEDUP_PER_QUERY PRVR_RAW_DEDUP_METHOD PRVR_RAW_DEDUP_DATASET PRVR_RAW_DEDUP_BRANCHES PRVR_RAW_DEDUP_QUERY_FILE PRVR_RAW_DEDUP_SEQUENCE PRVR_RAW_DEDUP_EVAL_QUERY_BSZ PRVR_RAW_DEDUP_TOP_L PRVR_RAW_DEDUP_UNIQUE_KS
    if ! validate_summary "$partial_summary" "$model"; then
      echo "failed raw-dedup validation: $model/$dataset" >&2
      echo "partial summary retained: $partial_summary" >&2
      continue
    fi
    mv "$partial_summary" "$summary"
    [[ -s "$partial_per_query" ]] || { echo "raw-dedup collector did not create per-query CSV: $partial_per_query" >&2; continue; }
    mv "$partial_per_query" "$per_query"
    "$PYTHON_BIN" "$SCRIPT_DIR/render_raw_dedup_reports.py" --root "$OUTPUT_ROOT" --dataset "$dataset"
  done
done
