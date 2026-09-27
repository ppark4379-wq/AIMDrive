#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONNOUSERSITE=1
exec "${MORAI_TRAIN_PYTHON:-python3}" -u train.py vision-encoder "$@"
