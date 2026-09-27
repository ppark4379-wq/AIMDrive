#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONNOUSERSITE=1

# The six-lap GPS model has the strongest waypoint/light result.  Keep its
# shared 106-token encoder frozen for this first controlled diagnostic and
# refine only the LMDrive traffic/object head on balanced positive/empty rows.
PYTHON="${MORAI_TRAIN_PYTHON:-python3}"
CONFIG="${MORAI_TRANSFER_CONFIG:-$ROOT/configs/morai_transfer.local.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[FAIL] PC별 설정이 없습니다: $CONFIG" >&2
  echo "       configs/examples/morai_transfer.example.json을 복사해 데이터와 checkpoint 경로를 지정하세요." >&2
  exit 2
fi

mapfile -t LOCAL_OBJECT_SETTINGS < <("$PYTHON" - "$CONFIG" <<'PY'
import json,sys
config=json.load(open(sys.argv[1],encoding='utf-8'))
print(config.get('object_initial_checkpoint',''))
print(config.get('object_initial_checkpoint_sha256',''))
PY
)
CHECKPOINT="${MORAI_OBJECT_INITIAL_CHECKPOINT:-${LOCAL_OBJECT_SETTINGS[0]:-}}"
CHECKPOINT_SHA256="${MORAI_OBJECT_INITIAL_SHA256:-${LOCAL_OBJECT_SETTINGS[1]:-}}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "[FAIL] object_initial_checkpoint가 없거나 파일을 찾을 수 없습니다: $CHECKPOINT" >&2
  exit 2
fi
if [[ -z "$CHECKPOINT_SHA256" ]]; then
  echo "[FAIL] object_initial_checkpoint_sha256을 PC별 config에 지정하세요." >&2
  exit 2
fi

echo "[START] 현재 waypoint/light 모델에서 object head만 균형 미세조정합니다."
echo "        object-positive : verified-empty = 1 : 1"
echo "        shared 106-token encoder, waypoint head, light head는 고정됩니다."
echo "        동일 validation baseline과 epoch별 object AP/F1을 비교해 최적 checkpoint를 보존합니다."

exec "$PYTHON" -u train.py vision-encoder-morai \
  --config "$CONFIG" \
  --initialization morai_checkpoint \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --trainable-scope object_head \
  --frame-sampling object_balanced \
  --object-negative-ratio 1.0 \
  --detection-loss-policy lmdrive_lav_v1 \
  --learning-rate 1e-4 \
  --epochs 5 \
  "$@"
