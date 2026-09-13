#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ENCODER_ROOT}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniforge3/envs/vldrive-ve/bin/python}"
DATASET_ROOT="${1:-${REPO_ROOT}/MoraiDrive/dataset_local/competition_train}"
SPLIT_FILE="${2:-${ENCODER_ROOT}/configs/morai_full_loop_80_20.json}"
OUTPUT_ROOT="${3:-${ENCODER_ROOT}/output_morai/full_loop_expert}"
PID_FILE="${OUTPUT_ROOT}/train.pid"
LOG_FILE="${OUTPUT_ROOT}/train.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment does not exist: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Training is already running: pid=${OLD_PID}" >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -e "${OUTPUT_ROOT}/checkpoint-last.pth.tar" ]] \
  || [[ -e "${OUTPUT_ROOT}/model-best.pth.tar" ]] \
  || [[ -e "${OUTPUT_ROOT}/metrics.jsonl" ]]; then
  echo "Output already contains a training run: ${OUTPUT_ROOT}" >&2
  exit 1
fi

cd "${ENCODER_ROOT}"
nohup env \
  TORCH_HOME="${ENCODER_ROOT}/.cache/torch" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" train_morai_pretrain.py \
    "${DATASET_ROOT}" \
    --split-file "${SPLIT_FILE}" \
    --output "${OUTPUT_ROOT}" \
    --epochs 25 \
    --warmup-epochs 5 \
    --batch-size 1 \
    --accumulation-steps 32 \
    --workers 2 \
    --reference-lr 7.5e-4 \
    --reference-rgb-lr 3e-4 \
    --weight-decay 0.05 \
    --clip-grad 5 \
    --waypoint-weight 0.5 \
    --traffic-light-weight 0.1 \
    --lidar-axis-policy morai_flu_to_lmdrive_fru \
    --log-interval 100 \
    >"${LOG_FILE}" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"
echo "training_pid=${TRAIN_PID}"
echo "training_log=${LOG_FILE}"
