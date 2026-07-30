#!/usr/bin/env bash
set -euo pipefail

# Build a conda environment for the PRVR/CLIP4Clip raw-frame code.
# This script does not require an already activated conda environment.
#
# Usage:
#   ./scripts/setup_prvr_conda_env.sh
#   ENV_NAME=my_prvr ./scripts/setup_prvr_conda_env.sh
#   PRVR_PYTHON_VERSION=3.10 ENV_NAME=my_prvr ./scripts/setup_prvr_conda_env.sh
#   FORCE_RECREATE=1 ENV_NAME=my_prvr ./scripts/setup_prvr_conda_env.sh

ENV_NAME="${ENV_NAME:-prvr}"
PYTHON_VERSION="${PRVR_PYTHON_VERSION:-3.9}"

# Defaults mirror the existing /venv/prvr environment on this machine.
TORCH_VERSION="${TORCH_VERSION:-1.13.1+cu117}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.14.1+cu117}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-0.13.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"

FORCE_RECREATE="${FORCE_RECREATE:-0}"
INSTALL_TORCHAUDIO="${INSTALL_TORCHAUDIO:-0}"
INSTALL_PRVR_EXTRAS="${INSTALL_PRVR_EXTRAS:-0}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  for candidate in \
    /opt/miniforge3/bin/conda \
    /opt/miniconda3/bin/conda \
    /opt/anaconda3/bin/conda \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

CONDA_EXE="$(find_conda)" || {
  echo "conda executable not found. Install Miniforge/Miniconda or add conda to PATH." >&2
  exit 1
}

env_exists() {
  "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

run_in_env() {
  "${CONDA_EXE}" run -n "${ENV_NAME}" "$@"
}

python_major_minor() {
  printf '%s\n' "${1}" | awk -F. '{print $1 "." $2}'
}

echo "Using conda: ${CONDA_EXE}"
echo "Target env: ${ENV_NAME}"

if env_exists; then
  if [[ "${FORCE_RECREATE}" == "1" ]]; then
    echo "Removing existing env: ${ENV_NAME}"
    "${CONDA_EXE}" env remove -y -n "${ENV_NAME}"
  else
    echo "Env already exists: ${ENV_NAME}"
    echo "Set FORCE_RECREATE=1 to rebuild it from scratch."
  fi
fi

if ! env_exists; then
  echo "Creating conda env with Python ${PYTHON_VERSION}"
  "${CONDA_EXE}" create -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" \
    pip \
    setuptools \
    wheel
fi

EXPECTED_PYTHON_VERSION="$(python_major_minor "${PYTHON_VERSION}")"
CURRENT_PYTHON_VERSION="$(run_in_env python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${CURRENT_PYTHON_VERSION}" != "${EXPECTED_PYTHON_VERSION}" ]]; then
  cat >&2 <<EOF
Python version mismatch in env '${ENV_NAME}'.
  expected: ${EXPECTED_PYTHON_VERSION}
  current:  ${CURRENT_PYTHON_VERSION}

PyTorch ${TORCH_VERSION} does not provide wheels for every Python version
(for example, Python 3.12 is too new for torch 1.13.1+cu117).

Recreate this env with:
  FORCE_RECREATE=1 ENV_NAME=${ENV_NAME} $0

Or choose a new env name:
  ENV_NAME=${ENV_NAME}_py39 $0
EOF
  exit 2
fi

echo "Upgrading pip tooling"
run_in_env python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch stack"
run_in_env python -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

if [[ "${INSTALL_TORCHAUDIO}" == "1" ]]; then
  run_in_env python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
fi

echo "Installing numeric/data packages"
run_in_env python -m pip install \
  "numpy==1.26.4" \
  "pandas==2.3.3" \
  "scipy==1.13.1" \
  "scikit-learn==1.6.1" \
  "h5py==3.14.0"

echo "Installing video/image packages"
run_in_env python -m pip install \
  "pillow==11.3.0" \
  "opencv-python==4.10.0.84" \
  "ffmpeg-python==0.2.0"

echo "Installing text/model utility packages"
run_in_env python -m pip install \
  "ftfy==6.3.1" \
  "regex==2026.1.15" \
  "tqdm==4.68.3" \
  "requests==2.32.5" \
  "PyYAML==6.0.3" \
  "transformers==4.41.2"

echo "Installing optional PRVR utility packages"
run_in_env python -m pip install \
  "easydict==1.13" \
  "einops==0.8.2" \
  "tensorboard==2.21.0" \
  "tensorboard-logger==0.1.0" \
  "tabulate==0.9.0" \
  "yacs==0.1.8"

if [[ "${INSTALL_PRVR_EXTRAS}" == "1" ]]; then
  echo "Installing larger PRVR extras"
  run_in_env python -m pip install \
    "matplotlib==3.9.4" \
    "seaborn==0.13.2" \
    "ipython==8.18.1" \
    "ipdb==0.13.13" \
    "fvcore==0.1.5.post20221221" \
    "iopath==0.1.10" \
    "spacy==3.7.5" \
    "thop==0.1.1.post2209072238" \
    "geoopt==0.5.0"
fi

echo "Re-pinning NumPy for torch/torchvision compatibility"
run_in_env python -m pip install --force-reinstall --no-deps "numpy==1.26.4"

echo "Verifying core imports"
run_in_env python -c '
import sys
import torch
import torchvision
import numpy
import pandas
import cv2
import PIL
import h5py
import ftfy
import regex
import tqdm
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("cv2", cv2.__version__)
print("PIL", PIL.__version__)
print("h5py", h5py.__version__)
print("ftfy", ftfy.__version__)
print("regex", regex.__version__)
print("tqdm", tqdm.__version__)
print("transformers", transformers.__version__)
'

cat <<EOF

Done.

Activate with:
  conda activate ${ENV_NAME}

If conda is not initialized in your shell, run:
  source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
EOF