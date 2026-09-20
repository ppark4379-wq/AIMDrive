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

PYTHON_BIN="${SCENARIO_PYTHON:-/usr/bin/python3}"
echo "[STATIC BATCH] Python: $PYTHON_BIN"
exec "$PYTHON_BIN" -m scenario_runner.tools.generate_static_scenario_suite "$@"
