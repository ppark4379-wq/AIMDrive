#!/usr/bin/env bash
set -Eeuo pipefail

SIGNAL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != "--check" ]]; then
    echo "사용법: bash $0 [--check]" >&2
    echo "--check: MORAI 없이 실제 Expert 경로·좌회전 신호 정책 검사" >&2
    echo "인자 없음: 우회전 → 빨강 대기 → 직진 초록 대기 → 좌회전 화살표 출발 1회" >&2
    exit 2
fi
source "$SIGNAL_PROJECT_ROOT/data_collection/tools/internal_sensor_record_env.sh"
cd "$SIGNAL_PROJECT_ROOT"
if [[ "${1:-}" == "--check" ]]; then
    # Offline validation needs Python dependencies, not ROS/IP discovery.
    SIGNAL_DEPS="${MORAI_ASYNC_DEPS:-$(morai_runtime_value ASYNC_DEPS)}"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$SIGNAL_DEPS:$SIGNAL_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec /usr/bin/python3 -u -m scenario_runner.main_runner \
        --scenario full_loop_traffic_light --once --no-sync-runtime --check
fi
morai_load_internal_sensor_record_environment
morai_require_capture_python
SIGNAL_OUTPUT="$SIGNAL_PROJECT_ROOT/scenario_runner/.runtime/traffic_light"
mkdir -p "$SIGNAL_OUTPUT"
SIGNAL_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec /usr/bin/python3 -u -m scenario_runner.main_runner \
    --scenario full_loop_traffic_light --once --no-sync-runtime \
    --ego-trace "$SIGNAL_OUTPUT/ego_trace_${SIGNAL_STAMP}.csv"
