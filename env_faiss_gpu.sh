#!/usr/bin/env bash
set -euo pipefail

# Create the PRVR environment used by the ANN latency benchmarks.
# It first installs the standard PRVR stack through env.sh, then adds
# FAISS-GPU from conda-forge.
#
# Usage:
#   bash env_faiss_gpu.sh
#   ENV_NAME=my_prvr_faiss bash env_faiss_gpu.sh
#   SKIP_BASE_SETUP=1 ENV_NAME=prvr_faiss_gpu bash env_faiss_gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-prvr_faiss_gpu}"
FAISS_VERSION="${FAISS_VERSION:-1.7.4}"
SKIP_BASE_SETUP="${SKIP_BASE_SETUP:-0}"

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

if [[ "${SKIP_BASE_SETUP}" != "1" ]]; then
  ENV_NAME="${ENV_NAME}" bash "${ROOT_DIR}/env.sh"
fi

echo "Installing FAISS-GPU ${FAISS_VERSION} in conda env: ${ENV_NAME}"
"${CONDA_EXE}" install -y -n "${ENV_NAME}" -c conda-forge "faiss-gpu=${FAISS_VERSION}"

echo "Verifying FAISS GPU support"
"${CONDA_EXE}" run -n "${ENV_NAME}" python - <<'PY'
import faiss
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("faiss", faiss.__version__, "gpu_count", faiss.get_num_gpus())
if not torch.cuda.is_available() or faiss.get_num_gpus() < 1:
    raise SystemExit("FAISS-GPU verification failed: no CUDA GPU is visible.")
PY

cat <<EOF

Done.

Activate with:
  conda activate ${ENV_NAME}

For PRVR + ANN latency commands:
  export PRVR_PYTHON="python"
EOF
