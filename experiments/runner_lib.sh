#!/usr/bin/env bash
# Shared helpers for the maintained train/eval entrypoints. This file is
# sourced, not executed.

set -Eeuo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PRVR_PROJECT_ROOT:-$(cd "$RUNNER_DIR/.." && pwd)}"
EXP_ROOT="${PRVR_EXP_ROOT:-$PROJECT_ROOT/experiments}"
PRVR_ROOT="${PRVR_ROOT:-$PROJECT_ROOT/all_prvr}"
DATA_ROOT="${PRVR_DATA_ROOT:-$PROJECT_ROOT/datasets}"
PYTHON_BIN="${PRVR_PYTHON:-python}"
CSV_PATH="$EXP_ROOT/recall_clip.csv"
LOG_DIR="$EXP_ROOT/logs"

export PRVR_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$PRVR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_DIR"

collection_for() {
    case "$1" in
        act) echo activitynet ;;
        tvr) echo tvr ;;
        cha) echo charades ;;
        msrvtt) echo msrvtt ;;
        *) echo "Unknown dataset alias: $1" >&2; return 2 ;;
    esac
}

dataset_arg_for() {
    if [[ "$2" == clip ]]; then
        printf '%s_clip\n' "$1"
    else
        printf '%s\n' "$1"
    fi
}

new_log_path() {
    local model="$1" dataset="$2" feature="$3"
    printf '%s/%s_gpu%s_%s_%s_%s.log\n' "$LOG_DIR" "$(date -u +%Y%m%dT%H%M%SZ)" "$GPU_ID" "$model" "$dataset" "$feature"
}

record() {
    local model="$1" dataset="$2" feature="$3" status="$4" log="$5" run_dir="$6" command="$7"
    "$PYTHON_BIN" "$EXP_ROOT/record_result.py" \
        --csv "$CSV_PATH" --gpu "$GPU_ID" --model "$model" --dataset "$dataset" \
        --feature "$feature" --status "$status" --log "$log" --run-dir "$run_dir" --command "$command"
}

run_clip4clip_eval_job() {
    # CLIP4Clip is a raw-frame, zero-shot T2VR baseline.  It deliberately
    # remains separate from pre-extracted-feature model code, but participates
    # in the ordered CLIP evaluation matrix as its first method.
    local dataset="$1" collection repo runner log run_dir command rc record_feature
    collection="$(collection_for "$dataset")"
    repo="$PRVR_ROOT/CLIP4Clip"
    runner="$repo/scripts/run_prvr_rawframes.sh"
    log="$(new_log_path CLIP4Clip "$dataset" rawframes128)"
    run_dir="$repo/results/rawframes/$collection/zeroshot_f128"
    record_feature=rawframes128
    if [[ "${MULTIGT:-0}" == 1 ]]; then
        record_feature=rawframes128_multiGT
        run_dir="${run_dir}_multiGT"
    fi
    command="cd $repo && MAX_FRAMES=128 MAX_EVAL_SAMPLES=0 BATCH_SIZE_VAL=32 NUM_WORKERS=2 MULTI_GT=${MULTIGT:-0} PYTHON_BIN=$PYTHON_BIN bash scripts/run_prvr_rawframes.sh zeroshot $collection $GPU_ID"

    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: CLIP4Clip zero-shot $dataset/rawframes128"
    if (
        cd "$repo"
        MAX_FRAMES=128 MAX_EVAL_SAMPLES=0 BATCH_SIZE_VAL=32 NUM_WORKERS=2 MULTI_GT="${MULTIGT:-0}" \
        PYTHON_BIN="$PYTHON_BIN" bash "$runner" zeroshot "$collection" "$GPU_ID"
    ) >"$log" 2>&1; then
        record CLIP4Clip "$dataset" "$record_feature" ok "$log" "$run_dir" "$command"
        echo "  complete; log: $log"
        return 0
    else
        rc=$?
    fi
    record CLIP4Clip "$dataset" "$record_feature" "eval_failed:$rc" "$log" "$run_dir" "$command"
    echo "  FAILED (rc=$rc); log: $log" >&2
    return "$rc"
}

run_in_dir() {
    local log="$1" cwd="$2"
    shift 2
    printf '\n===== %s =====\n' "$(printf '%q ' "$@")" >> "$log"
    set +e
    (cd "$cwd" && "$@") >> "$log" 2>&1
    local rc=$?
    set -e
    return "$rc"
}

latest_checkpoint() {
    local root="$1"
    find "$root" -type f -name best.ckpt -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-
}

# Config-based projects validate during training but do not always run the test
# split.  Run their documented --eval command after training and record that
# final test metric.  The result root is unique for model/dataset/feature, so
# no runner shares a checkpoint directory with another runner.
run_config_job() {
    local model="$1" repo="$2" cwd="$3" entry="$4" dataset="$5" feature="$6"
    local ds_arg collection log command checkpoint rc status
    local -a extra_args=()
    ds_arg="$(dataset_arg_for "$dataset" "$feature")"
    collection="$(collection_for "$dataset")"
    # MSC-PRVR's original RKD-angle formula is evaluated exactly in chunks for
    # MSRVTT.  MSRVTT has 20 captions/video, so the original dense N^3 tensor
    # cannot fit on a 24 GiB GPU at the unchanged batch size of 128.
    if [[ "$model" == "MSC-PRVR" && "$dataset" == "msrvtt" ]]; then
        extra_args+=(--rkd_angle_chunk_size 32)
    fi
    log="$(new_log_path "$model" "$dataset" "$feature")"
    command="cd $cwd && $PYTHON_BIN $entry -d $ds_arg --gpu $GPU_ID ${extra_args[*]}"

    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: $model $dataset/$feature"
    if run_in_dir "$log" "$cwd" "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" "${extra_args[@]}"; then
        :
    else
        rc=$?
        status="train_failed:$rc"
        record "$model" "$dataset" "$feature" "$status" "$log" "" "$command"
        echo "  FAILED (training); log: $log" >&2
        return 0
    fi

    checkpoint="$(latest_checkpoint "$repo/results/$feature/$collection")"
    if [[ -z "$checkpoint" ]]; then
        status="eval_skipped:no_best_ckpt"
        record "$model" "$dataset" "$feature" "$status" "$log" "" "$command"
        echo "  FAILED (best.ckpt not found); log: $log" >&2
        return 0
    fi
    if run_in_dir "$log" "$cwd" "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" "${extra_args[@]}" --eval --resume "$checkpoint"; then
        :
    else
        rc=$?
        status="eval_failed:$rc"
        record "$model" "$dataset" "$feature" "$status" "$log" "$checkpoint" "$command --eval --resume $checkpoint"
        echo "  FAILED (evaluation); log: $log" >&2
        return 0
    fi
    record "$model" "$dataset" "$feature" ok "$log" "$checkpoint" "$command --eval --resume $checkpoint"
    echo "  complete; log: $log"
}

# Resume a previously interrupted config-based run, then evaluate and record
# it with the same result handling as run_config_job.
run_config_resume_job() {
    local model="$1" repo="$2" cwd="$3" entry="$4" dataset="$5" feature="$6" resume_ckpt="$7"
    local ds_arg collection log command checkpoint rc status
    ds_arg="$(dataset_arg_for "$dataset" "$feature")"
    collection="$(collection_for "$dataset")"
    log="$(new_log_path "$model" "$dataset" "$feature")"
    command="cd $cwd && $PYTHON_BIN $entry -d $ds_arg --gpu $GPU_ID --resume $resume_ckpt"

    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: $model $dataset/$feature (resume)"
    if run_in_dir "$log" "$cwd" "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" --resume "$resume_ckpt"; then
        :
    else
        rc=$?
        status="train_failed:$rc"
        record "$model" "$dataset" "$feature" "$status" "$log" "$resume_ckpt" "$command"
        echo "  FAILED (training); log: $log" >&2
        return 0
    fi

    checkpoint="$(latest_checkpoint "$repo/results/$feature/$collection")"
    if [[ -z "$checkpoint" ]]; then
        status="eval_skipped:no_best_ckpt"
        record "$model" "$dataset" "$feature" "$status" "$log" "" "$command"
        echo "  FAILED (best.ckpt not found); log: $log" >&2
        return 0
    fi
    if run_in_dir "$log" "$cwd" "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" --eval --resume "$checkpoint"; then
        :
    else
        rc=$?
        status="eval_failed:$rc"
        record "$model" "$dataset" "$feature" "$status" "$log" "$checkpoint" "$command --eval --resume $checkpoint"
        echo "  FAILED (evaluation); log: $log" >&2
        return 0
    fi
    record "$model" "$dataset" "$feature" ok "$log" "$checkpoint" "$command --eval --resume $checkpoint"
    echo "  complete; log: $log"
}

# BGM-Net, MS-SL, and DL-DKD already invoke their original test evaluation at
# the end of method/train.py.  Keep that behavior rather than adding another
# evaluator with potentially different arguments.
run_auto_eval_job() {
    local model="$1" cwd="$2" dataset="$3" feature="$4"
    shift 4
    local log command rc
    log="$(new_log_path "$model" "$dataset" "$feature")"
    command="cd $cwd && $(printf '%q ' "$@")"
    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: $model $dataset/$feature"
    if run_in_dir "$log" "$cwd" "$@"; then
        :
    else
        rc=$?
        record "$model" "$dataset" "$feature" "train_or_eval_failed:$rc" "$log" "" "$command"
        echo "  FAILED; log: $log" >&2
        return 0
    fi
    record "$model" "$dataset" "$feature" ok "$log" "" "$command"
    echo "  complete; log: $log"
}

run_bgm_job() {
    local dataset="$1" feature="$2"
    case "$dataset" in
        act) run_auto_eval_job BGM-Net "$PRVR_ROOT/BGM-Net" act "$feature" "$PYTHON_BIN" method/train.py --collection activitynet --dset_name activitynet --root_path "$DATA_ROOT" --visual_feature i3d --feature_mode "$feature" --output_root "$PRVR_ROOT/BGM-Net/results" --exp_id "act_${feature}" --device_ids "$GPU_ID" --use_matcher_start_epoch 20 --map_size 48 --smp_rate 1.0 --bsz 128 ;;
        tvr) run_auto_eval_job BGM-Net "$PRVR_ROOT/BGM-Net" tvr "$feature" "$PYTHON_BIN" method/train.py --collection tvr --dset_name tvr --root_path "$DATA_ROOT" --visual_feature i3d_resnet --feature_mode "$feature" --output_root "$PRVR_ROOT/BGM-Net/results" --exp_id "tvr_${feature}" --device_ids "$GPU_ID" --margin 0.1 --bsz 128 --lr 0.00025 --use_matcher_start_epoch 0 --map_size 32 --smp_rate 0.01 ;;
        cha) run_auto_eval_job BGM-Net "$PRVR_ROOT/BGM-Net" cha "$feature" "$PYTHON_BIN" method/train.py --collection charades --dset_name charades --root_path "$DATA_ROOT" --visual_feature i3d_rgb_lgi --feature_mode "$feature" --output_root "$PRVR_ROOT/BGM-Net/results" --exp_id "cha_${feature}" --device_ids "$GPU_ID" --clip_scale_w 0.6 --frame_scale_w 0.4 --use_matcher_start_epoch 5 --map_size 48 --smp_rate 1.0 --bsz 16 ;;
        msrvtt) run_auto_eval_job BGM-Net "$PRVR_ROOT/BGM-Net" msrvtt "$feature" "$PYTHON_BIN" method/train.py --collection msrvtt --dset_name msrvtt --root_path "$DATA_ROOT" --visual_feature resnext101-resnet152 --feature_mode "$feature" --output_root "$PRVR_ROOT/BGM-Net/results" --exp_id "msrvtt_${feature}" --device_ids "$GPU_ID" --use_matcher_start_epoch 20 --map_size 48 --smp_rate 1.0 --bsz 128 ;;
        *) echo "Unknown BGM-Net dataset: $dataset" >&2; return 2 ;;
    esac
}

run_ms_sl_job() {
    local dataset="$1" feature="$2"
    case "$dataset" in
        act) run_auto_eval_job MS-SL "$PRVR_ROOT/ms-sl" act "$feature" "$PYTHON_BIN" method/train.py --collection activitynet --dset_name activitynet --root_path "$DATA_ROOT" --visual_feature i3d --feature_mode "$feature" --output_root "$PRVR_ROOT/ms-sl/results" --exp_id "act_${feature}" --device_ids "$GPU_ID" ;;
        tvr) run_auto_eval_job MS-SL "$PRVR_ROOT/ms-sl" tvr "$feature" "$PYTHON_BIN" method/train.py --collection tvr --dset_name tvr --root_path "$DATA_ROOT" --visual_feature i3d_resnet --feature_mode "$feature" --output_root "$PRVR_ROOT/ms-sl/results" --exp_id "tvr_${feature}" --device_ids "$GPU_ID" --margin 0.1 ;;
        cha) run_auto_eval_job MS-SL "$PRVR_ROOT/ms-sl" cha "$feature" "$PYTHON_BIN" method/train.py --collection charades --dset_name charades --root_path "$DATA_ROOT" --visual_feature i3d_rgb_lgi --feature_mode "$feature" --output_root "$PRVR_ROOT/ms-sl/results" --exp_id "cha_${feature}" --device_ids "$GPU_ID" --clip_scale_w 0.5 --frame_scale_w 0.5 ;;
        msrvtt) run_auto_eval_job MS-SL "$PRVR_ROOT/ms-sl" msrvtt "$feature" "$PYTHON_BIN" method/train.py --collection msrvtt --dset_name msrvtt --root_path "$DATA_ROOT" --visual_feature resnext101-resnet152 --feature_mode "$feature" --output_root "$PRVR_ROOT/ms-sl/results" --exp_id "msrvtt_${feature}" --device_ids "$GPU_ID" ;;
        *) echo "Unknown MS-SL dataset: $dataset" >&2; return 2 ;;
    esac
}

run_dldkd_job() {
    local dataset="$1" result_label="${2:-resnet}"
    case "$dataset" in
        act) run_auto_eval_job DL-DKD "$PRVR_ROOT/DL-DKD" act "$result_label" "$PYTHON_BIN" method/train.py --collection activitynet --dset_name activitynet --root_path "$DATA_ROOT" --visual_feature i3d --results_root "$PRVR_ROOT/DL-DKD/results/resnet" --exp_id act_resnet --model_name DLDKD --device_ids "$GPU_ID" --distill_loss_decay exp --double_branch --drop 0.25 --input_drop 0.25 --q_feat_size 1024 --label_style soft ;;
        tvr) run_auto_eval_job DL-DKD "$PRVR_ROOT/DL-DKD" tvr "$result_label" "$PYTHON_BIN" method/train.py --collection tvr --dset_name tvr --root_path "$DATA_ROOT" --visual_feature i3d_resnet --results_root "$PRVR_ROOT/DL-DKD/results/resnet" --exp_id tvr_resnet --model_name DLDKD --device_ids "$GPU_ID" --q_feat_size 768 --margin 0.1 --n_heads 4 --lr 0.0003 --distill_loss_decay exp --double_branch --drop 0.2 --input_drop 0.2 --label_style soft ;;
        cha) run_auto_eval_job DL-DKD "$PRVR_ROOT/DL-DKD" cha "$result_label" "$PYTHON_BIN" method/train.py --collection charades --dset_name charades --root_path "$DATA_ROOT" --visual_feature i3d_rgb_lgi --results_root "$PRVR_ROOT/DL-DKD/results/resnet" --exp_id cha_resnet --model_name DLDKD --device_ids "$GPU_ID" --lr 0.00024 --distill_loss_decay exp --double_branch --q_feat_size 1024 --drop 0.15 --input_drop 0.15 --label_style soft ;;
        msrvtt) run_auto_eval_job DL-DKD "$PRVR_ROOT/DL-DKD" msrvtt "$result_label" "$PYTHON_BIN" method/train.py --collection msrvtt --dset_name msrvtt --root_path "$DATA_ROOT" --visual_feature resnext101-resnet152 --results_root "$PRVR_ROOT/DL-DKD/results/resnet" --exp_id msrvtt_resnet --model_name DLDKD --device_ids "$GPU_ID" --distill_loss_decay exp --double_branch --drop 0.25 --input_drop 0.25 --q_feat_size 1024 --label_style soft ;;
        *) echo "Unknown DL-DKD dataset: $dataset" >&2; return 2 ;;
    esac
}
