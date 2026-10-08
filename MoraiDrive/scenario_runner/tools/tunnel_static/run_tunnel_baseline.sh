#!/usr/bin/env bash
# Use the same isolated dependencies as the running MORAI bridge.
set -Eeuo pipefail

TUNNEL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# != 0 )); then
    echo "사용법: $0 (장애물 없는 터널 구간 1회 검증)" >&2
    exit 2
fi

source "$TUNNEL_PROJECT_ROOT/data_collection/tools/internal_sensor_record_env.sh"
morai_load_internal_sensor_record_environment
morai_require_capture_python

TUNNEL_OUTPUT="$TUNNEL_PROJECT_ROOT/scenario_runner/.runtime/tunnel_baseline"
mkdir -p "$TUNNEL_OUTPUT"
TUNNEL_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cd "$TUNNEL_PROJECT_ROOT"
exec /usr/bin/python3 -u -m scenario_runner.main_runner \
    --scenario full_loop_tunnel_normal --once --no-sync-runtime \
    --ego-trace "$TUNNEL_OUTPUT/ego_trace_${TUNNEL_STAMP}.csv"
