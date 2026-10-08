#!/usr/bin/env bash
# Launch RLCD Stage 1 training (single H100 by default).
set -euo pipefail

CONFIG="${CONFIG:-configs/audit_qwen.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rlcd_audit}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# Reduce fragmentation with long-context allocations.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(--config "${CONFIG}" --output_dir "${OUTPUT_DIR}")
if [[ -n "${DATA_PATH:-}" ]]; then ARGS+=(--data_path "${DATA_PATH}"); fi
if [[ -n "${MODEL:-}" ]]; then ARGS+=(--model "${MODEL}"); fi

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES}" train.py "${ARGS[@]}" "$@"
else
  python train.py "${ARGS[@]}" "$@"
fi
