#!/usr/bin/env bash
# One-time cluster setup. Run from the repository root.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo ">> Creating virtualenv at ${VENV}"
"${PYTHON}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo ">> Upgrading pip"
pip install --upgrade pip wheel

echo ">> Installing PyTorch (CUDA 12.1 build by default; override TORCH_INDEX_URL if needed)"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
pip install "torch>=2.3.0" --index-url "${TORCH_INDEX_URL}"

echo ">> Installing project requirements"
pip install -r requirements.txt

echo ">> Verifying CUDA availability"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo ">> Done. Activate with: source ${VENV}/bin/activate"
