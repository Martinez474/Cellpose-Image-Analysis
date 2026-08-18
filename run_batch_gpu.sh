#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/home/mars/Documents/school/thesis/Input Micrographs"
RESULTS_DIR="/home/mars/Documents/school/thesis/results"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

# Required when ImageJ/Python is launched on the integrated GPU.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd "${PROJECT_DIR}"
echo "Checking CUDA device..."
"${PYTHON}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable; continuing on CPU")'
echo "Running batch predictions..."
"${PYTHON}" tools/batch_predict.py "${INPUT_DIR}" "${RESULTS_DIR}"
