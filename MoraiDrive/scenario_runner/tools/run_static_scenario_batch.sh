#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if pgrep -f '[s]cenario_runner.tools.static_scenario_editor' >/dev/null; then
  cat >&2 <<'EOF'
[FAIL] 수동 정적 시나리오 에디터가 실행 중입니다.
       에디터 터미널에서 Ctrl+C로 종료한 뒤 다시 실행하세요.
       에디터의 제동 relay와 자동 Expert 제어를 동시에 실행할 수 없습니다.
EOF
  exit 2
fi

if ! ss -H -ltn 'sport = :9090' 2>/dev/null | grep -q .; then
  cat >&2 <<'EOF'
[FAIL] 9090 MORAI bridge가 실행 중이지 않습니다.
       별도 터미널에서 센서 연결이 필요 없는 다음 모드를 유지하세요.
       ./data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only
EOF
  exit 2
fi

set +e
/usr/bin/python3 - <<'PYCTRL'
import rosgraph

try:
    _, subscribers, _ = rosgraph.Master(
        "/static_scenario_batch_preflight"
    ).getSystemState()
except Exception:
    raise SystemExit(2)

raise SystemExit(
    0 if dict(subscribers).get("/ctrl_cmd", []) else 3
)
PYCTRL
ctrl_cmd_code=$?
set -e
if (( ctrl_cmd_code == 2 )); then
  cat >&2 <<'EOF'
[FAIL] ROS master 상태를 읽을 수 없습니다.
       별도 터미널에서 다음 명령을 다시 실행해 준비 완료 메시지를 확인하세요.
       ./data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only
EOF
  exit 2
fi
if (( ctrl_cmd_code != 0 )); then
  cat >&2 <<'EOF'
[FAIL] MORAI bridge에 /ctrl_cmd 구독자가 없습니다.
       MORAI Vehicle Information을 01번에 표시된 주소(이 PC는 보통 127.0.0.1:9090)로
       연결하고, 01번 터미널의 /ctrl_cmd 준비 완료 메시지를 확인한 뒤 다시 실행하세요.
       이 상태에서는 모든 후보가 주행 시작 전에 동일하게 실패합니다.
EOF
  exit 2
fi
echo "[STATIC BATCH] MORAI vehicle control ready: /ctrl_cmd subscriber connected"

PYTHON_BIN="${SCENARIO_PYTHON:-/usr/bin/python3}"
# Run the runner and Expert in the same environment as the collection
# entrypoints (ROS, catkin morai_msgs, pinned grpc/protobuf); the calling
# shell may be a conda env that hides the user site or lacks the catkin path.
source "$ROOT/data_collection/tools/internal_sensor_record_env.sh"
morai_load_internal_sensor_record_environment
echo "[STATIC BATCH] Python: $PYTHON_BIN"
exec "$PYTHON_BIN" -m scenario_runner.tools.generate_static_scenario_suite "$@"
