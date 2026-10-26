#!/usr/bin/env bash
set -euo pipefail
static_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep the runner on system Python for ROS/gRPC. Let run_static.py select the
# Expert through runtime_python(): Windows lmdrive/vldrive receives MORAI's
# Windows localhost UDP. Do not force the caller's WSL/Conda interpreter here.
# AIM_STATIC_EXPERT_PYTHON, AIM_EXPERT_PYTHON and --expert-python remain overrides.
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
