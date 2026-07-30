#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-qwen3vl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0+cu124}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0+cu124}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0+cu124}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.4}"
MAX_JOBS="${MAX_JOBS:-4}"
RECREATE="${RECREATE:-0}"

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  for candidate in \
    /opt/miniforge3/bin/conda \
    /opt/conda/bin/conda \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda)" || {
  echo "conda not found. Install Miniforge/Miniconda first or set CONDA_EXE." >&2
  exit 1
}

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  if [[ "${RECREATE}" == "1" ]]; then
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "Using existing conda env: ${ENV_NAME}"
  fi
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --extra-index-url "${TORCH_INDEX_URL}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

pip install packaging ninja
pip uninstall -y flash-attn || true

install_flash_attn() {
  MAX_JOBS="${MAX_JOBS}" \
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir
}

if ! install_flash_attn; then
  echo "flash-attn install failed. Installing cuda-toolkit=${CUDA_TOOLKIT_VERSION} and retrying with nvcc." >&2
  conda install -c nvidia "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" -y
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  nvcc --version
  pip uninstall -y flash-attn || true
  MAX_JOBS="${MAX_JOBS}" FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir
fi

python - <<'PY'
import torch
import flash_attn
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
PY

pip install \
  accelerate \
  safetensors \
  tokenizers \
  huggingface-hub \
  qwen-vl-utils==0.0.14 \
  av \
  decord \
  pillow \
  tqdm \
  numpy \
  requests \
  PyYAML

pip install -U "transformers>=4.57.0"
pip install matplotlib

python - <<'PY'
import torch
import flash_attn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import qwen_vl_utils

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print("flash_attn:", flash_attn.__version__)
print("Qwen3VL import ok")
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
