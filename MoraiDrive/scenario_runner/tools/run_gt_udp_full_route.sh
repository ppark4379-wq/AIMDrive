#!/usr/bin/env bash
set -euo pipefail
gt_udp_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$gt_udp_repo:$gt_udp_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$gt_udp_repo" "$@" <<'PY'
import os,sys
from python_environment import build_capture_environment
repo=sys.argv[1]
env=build_capture_environment()
os.execve('/usr/bin/python3', ['/usr/bin/python3','-u','-B',
    repo+'/MoraiDrive/scenario_runner/tools/run_gt_udp_full_route.py',*sys.argv[2:]],env)
PY
