#!/usr/bin/env bash
# Static dual-branch ANN retrieval benchmark on ActivityNet/TVR CLIP features.
#
# Models with a query-dependent frame branch (MS-SL and BGM-Net) are
# deliberately excluded: indexing such a branch would change its original
# retrieval definition.  All other dual-branch PRVR models are covered.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage: benchmark_static_dual_ann.sh <condition> <gpu> [model|all] [max_queries]

condition:
  origin | ivf | ivf-x2 | ivf-gpu | ivf-gpu-x2 | hnsw | hnsw-x2

model:
  GMMFormer | GMMFormer-v2 | HLFormer | DreamPRVR | Holmes | BOA |
  MSC-PRVR | DL-DKD | all

Runs ActivityNet CLIP evaluation by default. Set
``PRVR_STATIC_ANN_DATASET=tvr`` to use TVR CLIP checkpoints and its L@20
candidate-depth configuration.
`max_queries=0` evaluates the full test split (default).

Notes:
  - IVF uses auto nlist=2^floor(log2(sqrt(raw branch corpus))) and nprobe
    64 (128 for *-x2).
  - HNSW uses M=128, efConstruction=256, efSearch=raw candidate K
    (2*K for hnsw-x2).
  - GPU IVF has FAISS's K<=2048 limit, so either branch's requested K is
    capped to 2048 and the effective values are saved in its summary.
EOF
}

[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
CONDITION="$1"; GPU_ID="$2"; TARGET="${3:-all}"; MAX_QUERIES="${4:-0}"
case "$CONDITION" in origin|ivf|ivf-x2|ivf-gpu|ivf-gpu-x2|hnsw|hnsw-x2) ;; *) usage >&2; exit 2;; esac
[[ "$GPU_ID" =~ ^[0-9]+$ && "$MAX_QUERIES" =~ ^[0-9]+$ ]] || { echo "gpu/max_queries must be non-negative integers" >&2; exit 2; }
DATASET="${PRVR_STATIC_ANN_DATASET:-act}"
OUTPUT_TAG="${PRVR_STATIC_ANN_OUTPUT_TAG:-}"
case "$DATASET" in
  act) COLLECTION=activitynet; DATASET_ARG=act_clip; CANDIDATE_K=30 ;;
  tvr) COLLECTION=tvr; DATASET_ARG=tvr_clip; CANDIDATE_K=20 ;;
  *) echo "PRVR_STATIC_ANN_DATASET must be act or tvr (got: $DATASET)" >&2; exit 2 ;;
esac

MODELS=(GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA MSC-PRVR DL-DKD)
if [[ "$TARGET" == all ]]; then
  SELECTED=("${MODELS[@]}")
else
  SELECTED=("$TARGET")
  found=0; for model in "${MODELS[@]}"; do [[ "$model" == "$TARGET" ]] && found=1; done
  (( found )) || { echo "unsupported static dual-branch model: $TARGET" >&2; exit 2; }
fi

branch_depths() {
  if [[ "$DATASET" == act ]]; then
    case "$1" in
      GMMFormer)    echo '848 30' ;;
      GMMFormer-v2) echo '832 2948' ;;
      HLFormer)     echo '834 3011' ;;
      DreamPRVR)    echo '830 2912' ;;
      Holmes)       echo '828 2883' ;;
      BOA)          echo '835 3169' ;;
      MSC-PRVR)     echo '737 3230' ;;
      DL-DKD)       echo '3103 2596' ;;
    esac
  else
    # TVR L@20 raw depths.  After dedup, retain 20 unique videos/branch.
    case "$1" in
      GMMFormer)    echo '334 20' ;;
      GMMFormer-v2) echo '291 444' ;;
      HLFormer)     echo '304 452' ;;
      DreamPRVR)    echo '306 404' ;;
      Holmes)       echo '316 396' ;;
      BOA)          echo '304 1108' ;;
      MSC-PRVR)     echo '242 1164' ;;
      DL-DKD)       echo '560 436' ;;
    esac
  fi
}

safe_name() { echo "$1" | tr '-' '_'; }

run_config_model() {
  local model="$1" left="$2" right="$3" spec repo_rel cwd_rel entry cwd collection ds_arg checkpoint out index_dir mode nprobe ef_left ef_right
  spec="$(config_spec "$model")"; IFS='|' read -r repo_rel cwd_rel entry <<< "$spec"
  cwd="$PRVR_ROOT/$cwd_rel"; collection="$COLLECTION"; ds_arg="$DATASET_ARG"
  checkpoint="$(latest_checkpoint "$PRVR_ROOT/$repo_rel/results/clip/$collection")"
  [[ -n "$checkpoint" ]] || { echo "missing checkpoint: $model" >&2; return 2; }
  out="$EXP_ROOT/ann_benchmark/$(safe_name "$model")/$DATASET/${CONDITION}${OUTPUT_TAG}.csv"
  index_dir="$(dirname "$checkpoint")/ann_static_indices"
  mode="$CONDITION"; nprobe=64; ef_left="$left"; ef_right="$right"
  case "$CONDITION" in
    origin) mode=origin ;;
    ivf) mode=ivf ;;
    ivf-x2) mode=ivf; nprobe=128 ;;
    ivf-gpu) mode=ivf-gpu ;;
    ivf-gpu-x2) mode=ivf-gpu; nprobe=128 ;;
    hnsw) mode=hnsw ;;
    hnsw-x2) mode=hnsw; ef_left=$((left * 2)); ef_right=$((right * 2)) ;;
  esac
  echo "[$(date -u +%FT%TZ)] $model: $CONDITION (raw K $left/$right)"
  (
    export PYTHONPATH="$EXP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PRVR_STATIC_ANN_MODE="$mode"
    export PRVR_STATIC_ANN_METHOD="$model"
    export PRVR_STATIC_ANN_CHECKPOINT="$checkpoint"
    export PRVR_STATIC_ANN_OUTPUT="$out"
    export PRVR_STATIC_ANN_INDEX_DIR="$index_dir"
    export PRVR_STATIC_ANN_LEFT_K="$left" PRVR_STATIC_ANN_RIGHT_K="$right"
    export PRVR_STATIC_ANN_LEFT_NPROBE="$nprobe" PRVR_STATIC_ANN_RIGHT_NPROBE="$nprobe"
    export PRVR_STATIC_ANN_LEFT_EF_SEARCH="$ef_left" PRVR_STATIC_ANN_RIGHT_EF_SEARCH="$ef_right"
    export PRVR_STATIC_ANN_HNSW_M=128 PRVR_STATIC_ANN_HNSW_EF_CONSTRUCTION=256
    export PRVR_STATIC_ANN_CANDIDATE_K="$CANDIDATE_K" PRVR_STATIC_ANN_MAX_QUERIES="$MAX_QUERIES"
    cd "$cwd"
    "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" --eval --resume "$checkpoint"
  )
}

run_dldkd() {
  local left="$1" right="$2" checkpoint model_dir mode nprobe ef_left ef_right out
  checkpoint="$(latest_model_checkpoint "$PRVR_ROOT/DL-DKD" "$DATASET" resnet)"
  [[ -n "$checkpoint" ]] || { echo 'missing checkpoint: DL-DKD' >&2; return 2; }
  model_dir="$(dirname "$checkpoint")"; out="$EXP_ROOT/ann_benchmark/DL_DKD/$DATASET/${CONDITION}${OUTPUT_TAG}.csv"
  mode="$CONDITION"; nprobe=64; ef_left="$left"; ef_right="$right"
  case "$CONDITION" in
    origin) mode=origin ;;
    ivf) mode=ivf ;;
    ivf-x2) mode=ivf; nprobe=128 ;;
    ivf-gpu) mode=ivf-gpu ;;
    ivf-gpu-x2) mode=ivf-gpu; nprobe=128 ;;
    hnsw) mode=hnsw ;;
    hnsw-x2) mode=hnsw; ef_left=$((left * 2)); ef_right=$((right * 2)) ;;
  esac
  echo "[$(date -u +%FT%TZ)] DL-DKD: $CONDITION (raw K $left/$right)"
  (
    export PYTHONPATH="$EXP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PRVR_STATIC_ANN_MODE="$mode" PRVR_STATIC_ANN_METHOD=DL-DKD
    export PRVR_STATIC_ANN_CHECKPOINT="$checkpoint" PRVR_STATIC_ANN_OUTPUT="$out"
    export PRVR_STATIC_ANN_INDEX_DIR="$model_dir/ann_static_indices"
    export PRVR_STATIC_ANN_LEFT_K="$left" PRVR_STATIC_ANN_RIGHT_K="$right"
    export PRVR_STATIC_ANN_LEFT_NPROBE="$nprobe" PRVR_STATIC_ANN_RIGHT_NPROBE="$nprobe"
    export PRVR_STATIC_ANN_LEFT_EF_SEARCH="$ef_left" PRVR_STATIC_ANN_RIGHT_EF_SEARCH="$ef_right"
    export PRVR_STATIC_ANN_HNSW_M=128 PRVR_STATIC_ANN_HNSW_EF_CONSTRUCTION=256
    export PRVR_STATIC_ANN_CANDIDATE_K="$CANDIDATE_K" PRVR_STATIC_ANN_MAX_QUERIES="$MAX_QUERIES"
    "$PYTHON_BIN" "$SCRIPT_DIR/eval_auto.py" --repo "$PRVR_ROOT/DL-DKD" --model-dir "$model_dir" --gpu "$GPU_ID"
  )
}

for model in "${SELECTED[@]}"; do
  read -r left right <<< "$(branch_depths "$model")"
  if [[ "$model" == DL-DKD ]]; then
    run_dldkd "$left" "$right"
  else
    run_config_model "$model" "$left" "$right"
  fi
done

if [[ -z "$OUTPUT_TAG" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/summarize_static_dual_ann.py" --root "$EXP_ROOT/ann_benchmark" --dataset "$DATASET" --condition "$CONDITION"
fi
