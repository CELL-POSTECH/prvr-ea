#!/usr/bin/env bash
# Shared implementation for the PRVR recall reproduction entrypoints.
# This file is sourced by train_*.sh and eval_*.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../runner_lib.sh"

ALL_DATASETS=(act tvr cha msrvtt)
# Keep the execution order aligned with the paper-comparison tables.  DL-DKD
# remains the same original ResNet-student + CLIP-teacher run under both labels.
RESNET_MODELS=(AMDNet GMMFormer GMMFormer-v2 HLFormer Holmes DreamPRVR BOA DL-DKD MS-SL BGM-Net)
CLIP_MODELS=(CLIP4Clip AMDNet GMMFormer GMMFormer-v2 HLFormer DreamPRVR Holmes BOA MSC-PRVR DL-DKD MS-SL BGM-Net)
ALL_MODELS=(CLIP4Clip AMDNet GMMFormer GMMFormer-v2 HLFormer Holmes DreamPRVR BOA MSC-PRVR DL-DKD MS-SL BGM-Net)

# Set by eval_multigt_{resnet,clip}.sh.  Training and ordinary eval keep the
# original score/target behavior unchanged.
MULTIGT="${MULTIGT:-0}"

evaluation_feature() {
    local feature="$1"
    if [[ "$MULTIGT" == 1 ]]; then
        printf '%s_multiGT\n' "$feature"
    else
        printf '%s\n' "$feature"
    fi
}

usage() {
    local action="$1" expected_feature="$2"
    cat <<EOF
Usage: $(basename "$0") <dataset|all> <gpu> [model]

  dataset: act | tvr | cha | msrvtt | all
  gpu:     physical CUDA GPU index, e.g. 0
  feature: $expected_feature (selected by this script's name)
  model:   optional; one of: ${ALL_MODELS[*]}

Examples:
  bash $(basename "$0") act 0 AMDNet
  bash $(basename "$0") all 1
EOF
}

normalize_dataset() {
    case "$1" in
        act|activitynet) echo act ;;
        tvr) echo tvr ;;
        cha|charades) echo cha ;;
        msrvtt) echo msrvtt ;;
        all) echo all ;;
        *) return 1 ;;
    esac
}

check_model() {
    local candidate="$1" model
    for model in "${ALL_MODELS[@]}"; do
        [[ "$candidate" == "$model" ]] && return 0
    done
    return 1
}

# DL-DKD is an original dual-feature model: its ResNet/I3D student and
# CLIP-B/32 teacher are always used together.  Both script labels therefore
# run the same original configuration.  MSC-PRVR is CLIP-only.
supports_feature() {
    local model="$1" feature="$2"
    [[ "$model" == CLIP4Clip && "$feature" == resnet ]] && return 1
    [[ "$model" == MSC-PRVR && "$feature" == resnet ]] && return 1
    return 0
}

models_for_feature() {
    case "$1" in
        resnet) printf '%s\n' "${RESNET_MODELS[@]}" ;;
        clip)   printf '%s\n' "${CLIP_MODELS[@]}" ;;
        *) return 2 ;;
    esac
}

config_spec() {
    # Prints: <repo-relative-path>|<working-directory-relative-path>|<entry>
    case "$1" in
        AMDNet)       echo 'AMDNet|AMDNet|main.py' ;;
        BOA)          echo 'BOA|BOA|src/main.py' ;;
        DreamPRVR)    echo 'CVPR26-DreamPRVR|CVPR26-DreamPRVR/src|main.py' ;;
        GMMFormer)    echo 'GMMFormer|GMMFormer/src|main.py' ;;
        GMMFormer-v2) echo 'GMMFormer_v2|GMMFormer_v2/src|main.py' ;;
        HLFormer)     echo 'ICCV25-HLFormer|ICCV25-HLFormer/src|main.py' ;;
        Holmes)       echo 'ICML26-Holmes|ICML26-Holmes/src|main.py' ;;
        MSC-PRVR)     echo 'MSC_PRVR|MSC_PRVR/src|main.py' ;;
        *) return 1 ;;
    esac
}

config_extra_args() {
    # Keep MSC's exact chunked RKD-angle implementation safe on 24 GiB GPUs.
    if [[ "$1" == MSC-PRVR && "$2" == msrvtt ]]; then
        printf '%s\n' --rkd_angle_chunk_size 32
    fi
}

run_config_train_only() {
    local model="$1" dataset="$2" feature="$3"
    local spec repo_rel cwd_rel entry repo cwd ds_arg log rc
    local -a extra_args=()
    spec="$(config_spec "$model")"
    IFS='|' read -r repo_rel cwd_rel entry <<< "$spec"
    repo="$PRVR_ROOT/$repo_rel"
    cwd="$PRVR_ROOT/$cwd_rel"
    ds_arg="$(dataset_arg_for "$dataset" "$feature")"
    mapfile -t extra_args < <(config_extra_args "$model" "$dataset")
    log="$(new_log_path "$model" "$dataset" "$feature")"

    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: train $model $dataset/$feature"
    if run_in_dir "$log" "$cwd" "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" "${extra_args[@]}"; then
        echo "  training complete; log: $log"
        return 0
    else
        rc=$?
    fi
    echo "  FAILED (training, rc=$rc); log: $log" >&2
    return "$rc"
}

run_config_eval_only() {
    local model="$1" dataset="$2" feature="$3"
    local spec repo_rel cwd_rel entry repo cwd collection ds_arg checkpoint log rc command
    local -a extra_args=() eval_args=()
    spec="$(config_spec "$model")"
    IFS='|' read -r repo_rel cwd_rel entry <<< "$spec"
    repo="$PRVR_ROOT/$repo_rel"
    cwd="$PRVR_ROOT/$cwd_rel"
    collection="$(collection_for "$dataset")"
    ds_arg="$(dataset_arg_for "$dataset" "$feature")"
    mapfile -t extra_args < <(config_extra_args "$model" "$dataset")
    [[ "$MULTIGT" == 1 ]] && eval_args+=(--multiGT)
    checkpoint="$(latest_checkpoint "$repo/results/$feature/$collection")"
    if [[ -z "$checkpoint" ]]; then
        echo "  no best.ckpt for $model $dataset/$feature" >&2
        return 2
    fi
    log="$(new_log_path "$model" "$dataset" "$feature")"
    command="cd $cwd && $PYTHON_BIN $entry -d $ds_arg --gpu $GPU_ID ${extra_args[*]} ${eval_args[*]} --eval --resume $checkpoint"
    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: eval $model $dataset/$feature"
    if run_in_dir "$log" "$cwd" "$PYTHON_BIN" "$entry" -d "$ds_arg" --gpu "$GPU_ID" "${extra_args[@]}" "${eval_args[@]}" --eval --resume "$checkpoint"; then
        record "$model" "$dataset" "$(evaluation_feature "$feature")" ok "$log" "$checkpoint" "$command"
        echo "  evaluation complete; log: $log"
        return 0
    else
        rc=$?
    fi
    record "$model" "$dataset" "$(evaluation_feature "$feature")" "eval_failed:$rc" "$log" "$checkpoint" "$command"
    echo "  FAILED (evaluation, rc=$rc); log: $log" >&2
    return "$rc"
}

latest_model_checkpoint() {
    local repo="$1" dataset="$2" feature="$3" collection
    collection="$(collection_for "$dataset")"
    find "$repo/results/$feature/$collection" -type f -name model.ckpt -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_auto_train() {
    local model="$1" dataset="$2" feature="$3"
    # These repositories keep their original train-then-test-eval behavior.
    case "$model" in
        BGM-Net) run_bgm_job "$dataset" "$feature" ;;
        MS-SL)   run_ms_sl_job "$dataset" "$feature" ;;
        DL-DKD)  run_dldkd_job "$dataset" "$feature" ;;
        *) return 2 ;;
    esac
}

run_auto_eval_only() {
    local model="$1" dataset="$2" feature="$3" repo checkpoint checkpoint_feature model_dir log rc command
    local -a eval_args=()
    case "$model" in
        BGM-Net) repo="$PRVR_ROOT/BGM-Net" ;;
        MS-SL)   repo="$PRVR_ROOT/ms-sl" ;;
        DL-DKD)  repo="$PRVR_ROOT/DL-DKD" ;;
        *) return 2 ;;
    esac
    # DL-DKD has one original dual-feature checkpoint tree (results/resnet).
    # The clip script is an experiment label for the same architecture, not a
    # request to replace its ResNet student with CLIP features.
    checkpoint_feature="$feature"
    [[ "$model" == DL-DKD ]] && checkpoint_feature=resnet
    checkpoint="$(latest_model_checkpoint "$repo" "$dataset" "$checkpoint_feature")"
    if [[ -z "$checkpoint" ]]; then
        echo "  no model.ckpt for $model $dataset/$checkpoint_feature" >&2
        return 2
    fi
    model_dir="$(dirname "$checkpoint")"
    [[ "$MULTIGT" == 1 ]] && eval_args+=(--multiGT)
    log="$(new_log_path "$model" "$dataset" "$feature")"
    command="cd $repo && $PYTHON_BIN $SCRIPT_DIR/eval_auto.py --repo $repo --model-dir $model_dir --gpu $GPU_ID ${eval_args[*]}"
    echo "[$(date -u +%FT%TZ)] GPU $GPU_ID: eval $model $dataset/$feature"
    if [[ "$model" == BGM-Net && "$dataset" == act ]]; then
        # 7 still peaks above 24 GiB while normalizing ActivityNet proposal
        # representations. Two queries preserve the original score computation
        # while leaving headroom for temporary normalization buffers.
        export PRVR_RAW_DEDUP_EVAL_QUERY_BSZ="${PRVR_RAW_DEDUP_EVAL_QUERY_BSZ:-2}"
    fi
    if run_in_dir "$log" "$repo" "$PYTHON_BIN" "$SCRIPT_DIR/eval_auto.py" --repo "$repo" --model-dir "$model_dir" --gpu "$GPU_ID" "${eval_args[@]}"; then
        record "$model" "$dataset" "$(evaluation_feature "$feature")" ok "$log" "$checkpoint" "$command"
        echo "  evaluation complete; log: $log"
        [[ "$model" == BGM-Net && "$dataset" == act ]] && unset PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
        return 0
    else
        rc=$?
    fi
    [[ "$model" == BGM-Net && "$dataset" == act ]] && unset PRVR_RAW_DEDUP_EVAL_QUERY_BSZ
    record "$model" "$dataset" "$(evaluation_feature "$feature")" "eval_failed:$rc" "$log" "$checkpoint" "$command"
    echo "  FAILED (evaluation, rc=$rc); log: $log" >&2
    return "$rc"
}

run_one() {
    local mode="$1" model="$2" dataset="$3" feature="$4"
    if [[ "$model" == CLIP4Clip ]]; then
        if [[ "$mode" == train ]]; then
            echo "  skip CLIP4Clip $dataset/$feature (zero-shot evaluation only)"
            return 0
        fi
        run_clip4clip_eval_job "$dataset"
        return $?
    fi
    if ! supports_feature "$model" "$feature"; then
        echo "  skip $model $dataset/$feature (not an original supported condition)"
        return 0
    fi
    case "$model" in
        BGM-Net|MS-SL|DL-DKD)
            if [[ "$mode" == train ]]; then
                run_auto_train "$model" "$dataset" "$feature"
            else
                run_auto_eval_only "$model" "$dataset" "$feature"
            fi
            ;;
        *)
            if [[ "$mode" == train ]]; then
                run_config_train_only "$model" "$dataset" "$feature"
            else
                run_config_eval_only "$model" "$dataset" "$feature"
            fi
            ;;
    esac
}

run_matrix() {
    local mode="$1" feature="$2" dataset_input="$3" gpu_input="$4" requested_model="${5:-all}"
    local dataset model failures=0
    dataset="$(normalize_dataset "$dataset_input")" || { echo "unknown dataset: $dataset_input" >&2; return 2; }
    [[ "$gpu_input" =~ ^[0-9]+$ ]] || { echo "gpu must be a non-negative integer" >&2; return 2; }
    GPU_ID="$gpu_input"

    if [[ "$requested_model" != all ]] && ! check_model "$requested_model"; then
        echo "unknown model: $requested_model" >&2
        return 2
    fi

    local -a datasets models
    [[ "$dataset" == all ]] && datasets=("${ALL_DATASETS[@]}") || datasets=("$dataset")
    if [[ "$requested_model" == all ]]; then
        mapfile -t models < <(models_for_feature "$feature")
    else
        models=("$requested_model")
    fi

    # Dataset outer loop makes each dataset's CSV block follow the table's
    # method order, even when `all` is requested.
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            if ! run_one "$mode" "$model" "$dataset" "$feature"; then
                echo "  continuing after failure: $model $dataset/$feature" >&2
                failures=$((failures + 1))
            fi
        done
    done
    [[ "$failures" -eq 0 ]]
}
