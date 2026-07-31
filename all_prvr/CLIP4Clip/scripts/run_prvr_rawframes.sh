#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_prvr_rawframes.sh <zeroshot|train|eval> <dataset> <gpu> [checkpoint]
#
# Environment overrides: MAX_FRAMES, BATCH_SIZE, BATCH_SIZE_VAL, EPOCHS,
# NUM_WORKERS, OUTPUT_ROOT, PYTHON_BIN, MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES,
# MULTI_GT=1 (dense multi-positive evaluation), CHUNK_SIZE (>0 for chunked
# zero-shot raw-frame retrieval before max-frame sampling), and
# RAW_FRAME_CPU_THREADS (default: 1, avoiding per-image thread oversubscription
# on high-core-count hosts).

MODE="${1:-}"
DATASET="${2:-}"
GPU="${3:-}"
CHECKPOINT="${4:-}"

if [[ -z "${MODE}" || -z "${DATASET}" || -z "${GPU}" ]]; then
  echo "Usage: $0 <zeroshot|train|eval> <msrvtt|activitynet|tvr|charades> <gpu> [checkpoint]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PRVR_PROJECT_ROOT:-$(cd "${ROOT_DIR}/../.." && pwd)}"
DATA_ROOT="${PRVR_DATA_ROOT:-${PROJECT_ROOT}/datasets}"
PYTHON_BIN="${PYTHON_BIN:-${PRVR_PYTHON:-python}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/rawframes}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-5}"
MAX_FRAMES="${MAX_FRAMES:-128}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
MULTI_GT="${MULTI_GT:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
RAW_FRAME_CPU_THREADS="${RAW_FRAME_CPU_THREADS:-1}"

if ! [[ "${CHUNK_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "CHUNK_SIZE must be a non-negative integer, got: ${CHUNK_SIZE}" >&2
  exit 2
fi
if ! [[ "${RAW_FRAME_CPU_THREADS}" =~ ^[0-9]+$ ]] || (( RAW_FRAME_CPU_THREADS < 1 )); then
  echo "RAW_FRAME_CPU_THREADS must be a positive integer, got: ${RAW_FRAME_CPU_THREADS}" >&2
  exit 2
fi

# PIL/torch preprocessing is performed once per raw image.  Letting every
# image use all host CPU threads is dramatically slower on high-core-count
# machines (and starves the frame loader).  Respect explicit user settings.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${RAW_FRAME_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${RAW_FRAME_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${RAW_FRAME_CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${RAW_FRAME_CPU_THREADS}}"

case "${DATASET}" in
  msrvtt)
    DATATYPE=raw_msrvtt
    DATA_PATH="${DATA_ROOT}/msrvtt"
    FRAME_ROOT="${DATA_PATH}/raw_frames"
    MAX_WORDS=32
    ;;
  activitynet)
    DATATYPE=raw_activitynet
    DATA_PATH="${DATA_ROOT}/activitynet"
    FRAME_ROOT="${DATA_PATH}/raw_frames"
    MAX_WORDS=64
    ;;
  tvr)
    DATATYPE=raw_tvr
    DATA_PATH="${DATA_ROOT}/tvr"
    FRAME_ROOT="${DATA_PATH}/raw_frames/frames_hq"
    MAX_WORDS=32
    ;;
  charades)
    DATATYPE=raw_charades
    DATA_PATH="${DATA_ROOT}/charades"
    FRAME_ROOT="${DATA_PATH}/raw_frames"
    MAX_WORDS=32
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

if (( MAX_FRAMES >= 128 )); then
  BATCH_SIZE="${BATCH_SIZE:-8}"
  BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-32}"
else
  BATCH_SIZE="${BATCH_SIZE:-32}"
  BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-32}"
fi

case "${MODE}" in
  zeroshot)
    RUN_ARGS=(--do_eval)
    OUT_DIR="${OUTPUT_ROOT}/${DATASET}/zeroshot_f${MAX_FRAMES}"
    ;;
  train)
    RUN_ARGS=(--do_train)
    OUT_DIR="${OUTPUT_ROOT}/${DATASET}/train_f${MAX_FRAMES}"
    ;;
  eval)
    if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
      echo "eval requires a checkpoint as the fourth argument." >&2
      exit 2
    fi
    RUN_ARGS=(--do_eval --init_model "${CHECKPOINT}")
    OUT_DIR="${OUTPUT_ROOT}/${DATASET}/eval_f${MAX_FRAMES}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

if (( CHUNK_SIZE > 0 )); then
  if [[ "${MODE}" != "zeroshot" ]]; then
    echo "CHUNK_SIZE is supported only for zero-shot evaluation." >&2
    exit 2
  fi
  OUT_DIR="${OUT_DIR}_chunk${CHUNK_SIZE}"
  CHUNK_ARGS=(--chunk_size "${CHUNK_SIZE}")
else
  CHUNK_ARGS=()
fi

if [[ "${MULTI_GT}" == "1" ]]; then
  OUT_DIR="${OUT_DIR}_multiGT"
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

MULTI_GT_ARGS=()
if [[ "${MULTI_GT}" == "1" ]]; then
  MULTI_GT_ARGS=(--multi_gt_eval)
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node=1 main_task_retrieval.py \
  "${RUN_ARGS[@]}" \
  --datatype "${DATATYPE}" \
  --data_path "${DATA_PATH}" \
  --caption_root "${DATA_PATH}/TextData" \
  --frame_root "${FRAME_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_size_val "${BATCH_SIZE_VAL}" \
  --num_thread_reader "${NUM_WORKERS}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  "${MULTI_GT_ARGS[@]}" \
  --max_words "${MAX_WORDS}" \
  --max_frames "${MAX_FRAMES}" \
  "${CHUNK_ARGS[@]}" \
  --feature_framerate 1 \
  --coef_lr 1e-3 \
  --freeze_layer_num 0 \
  --slice_framepos 2 \
  --loose_type --linear_patch 2d --sim_header meanP \
  --pretrained_clip_name ViT-B/32
