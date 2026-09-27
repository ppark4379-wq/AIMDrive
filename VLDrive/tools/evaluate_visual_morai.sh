#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
사용법:
  bash tools/evaluate_visual_morai.sh CHECKPOINT [추가 옵션]

예:
  bash tools/evaluate_visual_morai.sh \
    /path/to/checkpoint-last.pth.tar

configs/morai_transfer.local.json의 manifest_indices를 publication별·lap별로
따로 평가합니다. 모델 가중치는 변경하지 않습니다.
EOF
  exit 2
fi

CHECKPOINT="$1"
shift
exec "${MORAI_TRAIN_PYTHON:-python3}" -u -m utils.evaluation.morai_transfer_evaluation \
  --checkpoint "$CHECKPOINT" "$@"
