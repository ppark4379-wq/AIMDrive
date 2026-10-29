#!/usr/bin/env bash
# Full-course competition runs (tunnel scenes rotating) with the integrated AIM2026 runtime (vision encoder + Expert + virtual GPS).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GRPC_POLL_STRATEGY=epoll1
# The runner keeps pip --user packages out of its interpreter (PYTHONNOUSERSITE). When no package directory is
# configured (AIM2026_PYTHON_DEPS / MORAI_ASYNC_DEPS) and the system interpreter cannot import what the run needs,
# use the user's own package directory, so the command works without exporting anything first.
if [[ -z "${AIM2026_PYTHON_DEPS:-}${MORAI_ASYNC_DEPS:-}" ]] \
        && ! /usr/bin/python3 -c 'import google.protobuf, grpc, yaml, numpy, scipy' 2>/dev/null; then
    user_deps="$(/usr/bin/python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
    if [[ -n "$user_deps" && -d "$user_deps" ]]; then
        export AIM2026_PYTHON_DEPS="$user_deps"
    fi
fi
exec /usr/bin/python3 -B - "$SCENARIO_AIM_ROOT" "$@" <<'PY'
import os
import sys
from scenario_runner.utils.environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/utils/scenario_runner/tools/run_competition_aim2026.py', *sys.argv[2:]], build_capture_environment())
PY
