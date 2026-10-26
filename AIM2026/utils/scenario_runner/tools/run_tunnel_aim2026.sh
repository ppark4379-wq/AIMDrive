#!/usr/bin/env bash
# Tunnel-only scenario runs with the integrated AIM2026 runtime (vision encoder + Expert + virtual GPS).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GRPC_POLL_STRATEGY=epoll1
exec /usr/bin/python3 -B - "$SCENARIO_AIM_ROOT" "$@" <<'PY'
import os
import sys
from scenario_runner.utils.environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/utils/scenario_runner/tools/run_tunnel_aim2026.py', *sys.argv[2:]], build_capture_environment())
PY
