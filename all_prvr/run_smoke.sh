#!/usr/bin/env bash
# Run one full PRVR smoke job: one training epoch followed by its test recall.
# Usage: ./run_smoke.sh <model> <activitynet|tvr|charades|msrvtt> <resnet|clip> <gpu>
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PRVR_PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"
root="${PRVR_ROOT:-$project_root/all_prvr}"
python="${PRVR_PYTHON:-python}"
model=${1:?model is required}
collection=${2:?collection is required}
mode=${3:?feature mode is required}
gpu=${4:?gpu is required}

case "$collection" in
  activitynet) cli=act; visual=i3d ;;
  tvr) cli=tvr; visual=i3d_resnet ;;
  charades) cli=cha; visual=i3d_rgb_lgi ;;
  msrvtt) cli=msrvtt; visual=resnext101-resnet152 ;;
  *) echo "Unknown collection: $collection" >&2; exit 2 ;;
esac
case "$mode" in
  resnet|clip) ;;
  *) echo "Feature mode must be resnet or clip" >&2; exit 2 ;;
esac
if [[ "$mode" == clip ]]; then cli=${cli}_clip; fi
if [[ "$model" == msc && "$mode" != clip ]]; then
  echo "MSC-PRVR is intentionally CLIP-only." >&2
  exit 2
fi

export PRVR_DATA_ROOT="${PRVR_DATA_ROOT:-$project_root/datasets}"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

run_config() {
  local repo=$1 workdir=$2
  cd "$root/$repo/$workdir"
  "$python" -u main.py -d "$cli" --gpu "$gpu" --n_epoch 1 --num_workers 4
  local ckpt
  ckpt=$(find "$root/$repo/results/$mode/$collection" -name best.ckpt -type f -print -quit)
  test -n "$ckpt"
  "$python" -u main.py -d "$cli" --gpu "$gpu" --num_workers 4 --eval --resume "$ckpt"
}

run_script() {
  local repo=$1 kind=$2
  cd "$root/$repo"
  export PYTHONPATH="$PWD:$root${PYTHONPATH:+:$PYTHONPATH}"
  local out="$root/$repo/results/smoke"
  if [[ "$kind" == dldkd ]]; then
    "$python" -u method/train.py --collection "$collection" --dset_name "$collection" \
      --root_path "$PRVR_DATA_ROOT" --visual_feature "$visual" --feature_mode "$mode" \
      --results_root "$out/$mode" --exp_id "${collection}_${mode}_smoke" --model_name DLDKD \
      --device_ids "$gpu" --n_epoch 1 --num_workers 4
  else
    "$python" -u method/train.py --collection "$collection" --dset_name "$collection" \
      --root_path "$PRVR_DATA_ROOT" --visual_feature "$visual" --feature_mode "$mode" \
      --output_root "$out" --exp_id "${collection}_${mode}_smoke" \
      --device_ids "$gpu" --n_epoch 1 --num_workers 4
  fi
}

case "$model" in
  amd) run_config AMDNet . ;;
  boa) run_config BOA src ;;
  gmm) run_config GMMFormer src ;;
  gmmv2) run_config GMMFormer_v2 src ;;
  hlformer) run_config ICCV25-HLFormer src ;;
  dream) run_config CVPR26-DreamPRVR src ;;
  holmes) run_config ICML26-Holmes src ;;
  msc) run_config MSC_PRVR src ;;
  bgm) run_script BGM-Net bgm ;;
  mssl) run_script ms-sl mssl ;;
  dldkd) run_script DL-DKD dldkd ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
