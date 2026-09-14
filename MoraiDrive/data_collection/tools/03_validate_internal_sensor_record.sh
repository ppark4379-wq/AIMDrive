#!/usr/bin/env bash
# Keep this entrypoint LF-only; Windows Git checkouts are supported.
set -Eeuo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TOOLS/internal_sensor_record_env.sh"

echo "[PREPARE] 검증용 PC 설정과 Python 의존성을 자동 점검합니다."
morai_prepare_internal_sensor_record_environment runtime
morai_load_internal_sensor_record_environment
morai_require_capture_python
morai_require_analysis_python
/usr/bin/python3 \
    "$MORAI_INTERNAL_SENSOR_RECORD_ROOT/tools/preflight_local_storage.py" \
    --reserve-gib 0

exec /usr/bin/python3 "$TOOLS/03_validate_internal_sensor_record.py" "$@"
