#!/usr/bin/env bash
# One-time cluster setup. Run from the repository root.
#
# torch>=2.6 is required: recent transformers builds import `torch.accelerator`
# at load time, which does not exist in older torch releases.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

# H100 supports CUDA 12.4 wheels. Override for other drivers, e.g.
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo ">> Creating virtualenv at ${VENV}"
"${PYTHON}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo ">> Upgrading pip"
pip install --upgrade pip wheel

echo ">> Installing PyTorch (torch>=2.6) from ${TORCH_INDEX_URL}"
pip install "torch>=2.6.0" --index-url "${TORCH_INDEX_URL}"

echo ">> Installing project requirements"
pip install -r requirements.txt

echo ">> Verifying the environment"
python scripts/check_env.py

echo ">> Done. Activate with: source ${VENV}/bin/activate"
