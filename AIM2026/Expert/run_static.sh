#!/usr/bin/env bash
set -euo pipefail
static_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The runner uses system Python for its ROS/gRPC environment, while the Expert
# and its inference child need the Python selected in the caller's shell.
export AIM_STATIC_EXPERT_PYTHON="${AIM_STATIC_EXPERT_PYTHON:-${AIM_EXPERT_PYTHON:-$(command -v python3)}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export GRPC_POLL_STRATEGY=epoll1
export PYTHONPATH="$static_repo:$static_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$static_repo" "$@" <<'PY'
import os
import sys
from python_environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/AIM2026/Expert/run_static.py', *sys.argv[2:]], build_capture_environment())
PY
