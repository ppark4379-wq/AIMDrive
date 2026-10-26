#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"

ROOT="$SCENARIO_AIM_ROOT"
cd "$ROOT"

if pgrep -f '[s]cenario_runner.tools.static_scenario_editor' >/dev/null; then
  echo "[FAIL] 수동 웹 에디터를 먼저 종료하세요." >&2
  exit 2
fi
if ! ss -H -ltn 'sport = :9090' 2>/dev/null | grep -q .; then
  echo "[FAIL] 01번 --scenario-only bridge를 먼저 실행하세요." >&2
  exit 2
fi

PYTHON_BIN="${SCENARIO_PYTHON:-/usr/bin/python3}"
exec "$PYTHON_BIN" -m scenario_runner.tools.review_static_scenario_candidate "$@"
