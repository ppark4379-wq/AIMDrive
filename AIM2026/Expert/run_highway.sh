#!/usr/bin/env bash
set -euo pipefail
highway_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export GRPC_POLL_STRATEGY=epoll1
export PYTHONPATH="$highway_repo:$highway_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$highway_repo" "$@" <<'PY'
import os
import sys
from python_environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/AIM2026/Expert/run_highway.py', *sys.argv[2:]], build_capture_environment())
PY
