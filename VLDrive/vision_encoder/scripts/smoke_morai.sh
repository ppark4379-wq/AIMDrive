#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ENCODER_ROOT}/../.." && pwd)"
export TORCH_HOME="${ENCODER_ROOT}/.cache/torch"

DEFAULT_PYTHON="/opt/miniforge3/envs/vldrive-ve/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DATASET_ROOT="${1:-${REPO_ROOT}/MoraiDrive/dataset_local/competition_train}"
SPLIT_FILE="${2:-${ENCODER_ROOT}/configs/morai_full_loop_80_20.json}"
OUTPUT_ROOT="${3:-${ENCODER_ROOT}/output_morai/smoke}"

cd "${ENCODER_ROOT}"
"${PYTHON_BIN}" train_morai_pretrain.py \
  "${DATASET_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --output "${OUTPUT_ROOT}" \
  --epochs 1 \
  --warmup-epochs 0 \
  --batch-size 1 \
  --accumulation-steps 1 \
  --workers 0 \
  --max-train-batches 1 \
  --max-validation-batches 1 \
  --lidar-axis-policy morai_flu_to_lmdrive_fru
