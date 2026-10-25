#!/usr/bin/env bash
set -euo pipefail
roundabout_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export GRPC_POLL_STRATEGY=epoll1
export PYTHONPATH="$roundabout_repo:$roundabout_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$roundabout_repo" "$@" <<'PY'
import os
import sys
from python_environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/AIM2026/Expert/run_roundabout.py', *sys.argv[2:]], build_capture_environment())
PY
